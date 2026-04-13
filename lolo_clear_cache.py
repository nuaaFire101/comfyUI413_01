"""
LoLolClearCache - 透明缓存清理节点（增强版）
=============================================

功能：
- 接收任意输入
- 输出相同的输入（透传）
- 执行后清理 ComfyUI 的缓存（显存、内存、未使用的模型等）
- 【新增】可清理 ComfyUI 的 prompt 执行缓存（解决循环生成中内存持续增长问题）
- 保持工作流其他节点不受影响

核心改进：
  ComfyUI 内部会缓存每个节点的执行结果（存储在 PromptExecutor 的 caches 中），
  在循环（for loop）场景下，前一轮的 VAEDecode、ImageFromBatch 等节点的输出
  （每轮 ~1.4 GB）不会自动释放，导致内存随循环次数线性增长。
  
  本节点通过访问 PromptServer.instance.prompt_queue 获取当前执行器，
  并清理其中的节点输出缓存，确保只保留当前轮需要的数据。

使用场景：
- 在循环工作流的 forLoopEnd 之前插入此节点
- 开启 clean_comfy_cache 选项来释放历史轮次的节点输出
"""

import torch
import gc
import time
import logging

# 尝试导入 psutil 以打印详细内存日志
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# 尝试导入 ComfyUI 的 model_management，以便清理未使用的模型
try:
    import comfy.model_management
except ImportError:
    comfy = None

# 尝试导入 ComfyUI 的 PromptServer，以便访问执行缓存
try:
    from server import PromptServer
    HAS_PROMPT_SERVER = True
except ImportError:
    HAS_PROMPT_SERVER = False

logger = logging.getLogger("LoLolClearCache")


def _get_mem_info():
    """获取当前内存使用信息的字符串"""
    lines = []
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        lines.append(f"RAM: {mem.used/1024**3:.2f}/{mem.total/1024**3:.2f} GB ({mem.percent:.1f}%)")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        lines.append(f"VRAM: alloc {allocated:.2f} GB, reserved {reserved:.2f} GB")
    return " | ".join(lines) if lines else "N/A"


class LoLolClearCache:
    """
    透明缓存清理节点（增强版）
    功能：透传输入，可清理 CUDA 缓存、执行垃圾回收、清理未使用的模型、
          以及 ComfyUI 的 prompt 执行缓存。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clean_cuda": ("BOOLEAN", {"default": True, "label": "清空 CUDA 缓存"}),
                "clean_memory": ("BOOLEAN", {"default": True, "label": "强制垃圾回收"}),
                "clean_unused_models": ("BOOLEAN", {"default": False, "label": "清理未使用的模型"}),
                "clean_comfy_cache": ("BOOLEAN", {"default": True, "label": "清理节点执行缓存（循环必开）"}),
            },
            "optional": {
                "input_1": ("*",),
                "input_2": ("*",),
                "input_3": ("*",),
                "input_4": ("*",),
                "input_5": ("*",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            }
        }

    RETURN_TYPES = ("*",) * 5
    RETURN_NAMES = ("output_1", "output_2", "output_3", "output_4", "output_5")
    FUNCTION = "process"
    CATEGORY = "LoLoNodes"
    DESCRIPTION = "透传节点，执行后可清理显存、内存、未使用的模型和 ComfyUI 节点执行缓存。在循环工作流中务必开启 clean_comfy_cache 以防内存溢出。"

    def _clear_comfy_execution_cache(self, unique_id, prompt):
        """
        清理 ComfyUI 的 prompt 执行缓存。
        
        策略：遍历执行缓存中的所有节点输出，只保留当前节点的直接输入来源，
        清理其他所有中间节点的缓存输出。
        
        这样能确保：
        - 当前节点接收的 input 数据不会被意外清理
        - VAEDecode、采样器等大量占用内存的中间结果会被释放
        """
        if not HAS_PROMPT_SERVER:
            logger.warning("[LoLolClearCache] PromptServer 不可用，无法清理执行缓存")
            return 0

        cleared_count = 0
        try:
            prompt_queue = PromptServer.instance.prompt_queue

            # 获取当前正在执行的 executor
            current_executor = None
            if hasattr(prompt_queue, 'currently_running'):
                for item_id, value in list(prompt_queue.currently_running.items()):
                    # currently_running 的值结构可能因版本而异
                    if isinstance(value, tuple) and len(value) >= 2:
                        current_executor = value[1]
                    else:
                        current_executor = value
                    break

            if current_executor is None:
                logger.info("[LoLolClearCache] 未找到当前执行器")
                return 0

            # 找到需要保护的节点ID（当前节点 + 直接输入来源）
            protected_nodes = set()
            protected_nodes.add(str(unique_id))

            if prompt and str(unique_id) in prompt:
                node_info = prompt[str(unique_id)]
                inputs = node_info.get("inputs", {})
                for key, value in inputs.items():
                    if isinstance(value, list) and len(value) >= 2:
                        # [node_id, slot_index] 表示一个连接
                        protected_nodes.add(str(value[0]))

            # 尝试多种方式访问执行缓存（适配不同 ComfyUI 版本）
            cache_cleared = False

            # 方式1: executor.caches.outputs（较新版本）
            if hasattr(current_executor, 'caches'):
                caches = current_executor.caches
                cache_dict = None

                if hasattr(caches, 'outputs'):
                    outputs = caches.outputs
                    # DynamicCache 或 LRUCache 内部可能有 .cache 属性
                    if hasattr(outputs, 'cache'):
                        cache_dict = outputs.cache
                    elif isinstance(outputs, dict):
                        cache_dict = outputs

                if cache_dict is not None:
                    keys_to_delete = [
                        nid for nid in list(cache_dict.keys())
                        if str(nid) not in protected_nodes
                    ]
                    for nid in keys_to_delete:
                        try:
                            del cache_dict[nid]
                            cleared_count += 1
                        except (KeyError, RuntimeError):
                            pass
                    cache_cleared = True

            # 方式2: executor.outputs（旧版本）
            if not cache_cleared and hasattr(current_executor, 'outputs'):
                outputs = current_executor.outputs
                if isinstance(outputs, dict):
                    keys_to_delete = [
                        nid for nid in list(outputs.keys())
                        if str(nid) not in protected_nodes
                    ]
                    for nid in keys_to_delete:
                        try:
                            del outputs[nid]
                            cleared_count += 1
                        except (KeyError, RuntimeError):
                            pass
                    cache_cleared = True

            # 方式3: 通用回退 - soft_empty_cache
            if not cache_cleared:
                if comfy is not None and hasattr(comfy.model_management, 'soft_empty_cache'):
                    comfy.model_management.soft_empty_cache()
                    logger.info("[LoLolClearCache] 回退: 已执行 soft_empty_cache")

            if cleared_count > 0:
                logger.info(
                    f"[LoLolClearCache] 已清理 {cleared_count} 个节点的执行缓存"
                    f"（保护了 {len(protected_nodes)} 个节点: {protected_nodes}）"
                )

        except Exception as e:
            logger.error(f"[LoLolClearCache] 清理执行缓存时出错: {e}")
            import traceback
            traceback.print_exc()

        return cleared_count

    def clear_cache(self, clean_cuda, clean_memory, clean_unused_models,
                    clean_comfy_cache=False, unique_id=None, prompt=None):
        """执行缓存清理"""
        try:
            mem_before = _get_mem_info()
            logger.info(f"[LoLolClearCache] 开始清理缓存... 清理前: {mem_before}")

            # 1. 清理 ComfyUI 执行缓存（最重要，先释放大对象引用）
            if clean_comfy_cache:
                self._clear_comfy_execution_cache(unique_id, prompt)

            # 2. 清理未使用的模型
            if clean_unused_models and comfy is not None:
                if hasattr(comfy.model_management, 'cleanup_models'):
                    comfy.model_management.cleanup_models()
                    logger.info("  - 未使用的模型已清理")
                if hasattr(comfy.model_management, 'soft_empty_cache'):
                    comfy.model_management.soft_empty_cache()
                    logger.info("  - soft_empty_cache 已执行")

            # 3. Python 垃圾回收（多次确保彻底，必须在删除缓存引用之后）
            if clean_memory:
                for _ in range(3):
                    gc.collect()
                logger.info("  - 垃圾回收已执行")

            # 4. 清空 CUDA 缓存（必须在 gc 之后，此时无引用的显存才能真正释放）
            if clean_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("  - CUDA 缓存已清空")

            mem_after = _get_mem_info()
            logger.info(f"[LoLolClearCache] 缓存清理完成. 清理后: {mem_after}")
            return True

        except Exception as e:
            logger.error(f"[LoLolClearCache] 清理缓存时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process(self, clean_cuda, clean_memory, clean_unused_models, clean_comfy_cache=True,
                unique_id=None, prompt=None, **kwargs):
        """
        处理函数：
        - 接收可选输入（input_1 ~ input_5）
        - 执行清理（包括 ComfyUI 执行缓存）
        - 返回与输入顺序对应的5个输出（未提供的输入对应 None）
        """
        inputs = [kwargs.get(f"input_{i}") for i in range(1, 6)]
        non_none_inputs = [f"input_{i}" for i, v in enumerate(inputs, start=1) if v is not None]
        logger.info(f"[LoLolClearCache] 收到输入: {', '.join(non_none_inputs) or '无'}")

        self.clear_cache(clean_cuda, clean_memory, clean_unused_models,
                         clean_comfy_cache, unique_id, prompt)

        return tuple(inputs)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """每次执行都视为变化，避免被缓存"""
        return float(time.time())


class LoLolClearCacheWithLabel(LoLolClearCache):
    """
    带标签的缓存清理节点
    增加一个标签参数，用于在日志中标识清理操作。
    """

    @classmethod
    def INPUT_TYPES(cls):
        parent_input = super().INPUT_TYPES()
        parent_input["required"] = {
            "label": ("STRING", {"default": "Cache Cleared", "multiline": False}),
            **parent_input["required"]
        }
        return parent_input

    RETURN_TYPES = ("*",) * 5
    RETURN_NAMES = ("output_1", "output_2", "output_3", "output_4", "output_5")
    FUNCTION = "process"
    CATEGORY = "LoLoNodes"
    DESCRIPTION = "透传节点，带自定义标签，执行后可清理显存、内存、未使用的模型和执行缓存。"

    def process(self, label, clean_cuda, clean_memory, clean_unused_models, clean_comfy_cache=True,
                unique_id=None, prompt=None, **kwargs):
        inputs = [kwargs.get(f"input_{i}") for i in range(1, 6)]
        non_none_inputs = [f"input_{i}" for i, v in enumerate(inputs, start=1) if v is not None]
        logger.info(f"[LoLolClearCache] ({label}) 收到输入: {', '.join(non_none_inputs) or '无'}")

        self.clear_cache(clean_cuda, clean_memory, clean_unused_models,
                         clean_comfy_cache, unique_id, prompt)
        logger.info(f"[LoLolClearCache] ({label}) 清理完成")

        return tuple(inputs)
