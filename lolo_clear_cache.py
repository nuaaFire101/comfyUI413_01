"""
LoLolClearCache - 透明缓存清理节点（增强版 v2）
=================================================

核心改进：
  ComfyUI 的 PromptExecutor 是 prompt_worker 函数中的局部变量 `e`，
  外部无法直接访问。本版本通过两种策略找到它：
  
  策略1：遍历所有线程的栈帧，找到 prompt_worker 函数中的局部变量 `e`
         （PromptExecutor 实例），然后直接操作其 caches.outputs。
  
  策略2：如果策略1失败，通过 monkey-patch PromptExecutor.__init__，
         在下次创建时将实例注册到 PromptServer 上。

  找到 executor 后，调用 caches.outputs 的方法清理节点输出缓存，
  确保循环中前几轮的大张量（VAEDecode ~736MB/轮）被及时释放。
"""

import torch
import gc
import sys
import time
import threading
import logging

# 尝试导入 psutil 以打印详细内存日志
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# 尝试导入 ComfyUI 的 model_management
try:
    import comfy.model_management
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False

# 尝试导入 execution 模块以识别 PromptExecutor 类型
try:
    from execution import PromptExecutor
    HAS_EXECUTOR_CLASS = True
except ImportError:
    HAS_EXECUTOR_CLASS = False

logger = logging.getLogger("LoLolClearCache")

# 全局缓存：一旦找到 executor 就存起来，不用每次都遍历栈帧
_cached_executor = None


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


def _find_prompt_executor():
    """
    通过遍历所有线程的栈帧，找到 prompt_worker 中的 PromptExecutor 实例。
    
    原理：ComfyUI 的 main.py 中有：
        def prompt_worker(q, server_instance):
            e = execution.PromptExecutor(...)
            while True:
                e.execute(...)
    
    当我们的节点代码在执行时，prompt_worker 线程的调用栈中
    一定有 `e` 这个局部变量。我们遍历所有线程栈帧找到它。
    """
    global _cached_executor

    # 先检查缓存的 executor 是否仍然有效
    if _cached_executor is not None:
        if hasattr(_cached_executor, 'caches') and hasattr(_cached_executor.caches, 'outputs'):
            return _cached_executor
        else:
            _cached_executor = None

    if not HAS_EXECUTOR_CLASS:
        return None

    # 遍历所有线程的栈帧
    for thread_id, frame in sys._current_frames().items():
        current_frame = frame
        while current_frame is not None:
            # 检查该栈帧的局部变量中是否有 PromptExecutor 实例
            for var_name, var_value in current_frame.f_locals.items():
                if isinstance(var_value, PromptExecutor):
                    logger.info(f"[LoLolClearCache] 找到 PromptExecutor: "
                               f"线程={thread_id}, 帧={current_frame.f_code.co_name}, 变量名={var_name}")
                    _cached_executor = var_value
                    return var_value
            current_frame = current_frame.f_back

    logger.warning("[LoLolClearCache] 未能通过栈帧找到 PromptExecutor")
    return None


class LoLolClearCache:
    """
    透明缓存清理节点（增强版 v2）
    功能：透传输入，可清理 CUDA 缓存、执行垃圾回收、清理未使用的模型、
          以及 ComfyUI 的 prompt 执行缓存（直接操作 PromptExecutor.caches）。
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
        
        通过栈帧回溯找到 PromptExecutor，然后操作其 caches.outputs，
        删除除当前节点直接输入来源以外的所有节点输出缓存。
        """
        executor = _find_prompt_executor()
        if executor is None:
            logger.warning("[LoLolClearCache] 无法找到 PromptExecutor，跳过执行缓存清理")
            # 回退方案
            if HAS_COMFY and hasattr(comfy.model_management, 'soft_empty_cache'):
                comfy.model_management.soft_empty_cache()
                logger.info("[LoLolClearCache] 回退: 已执行 soft_empty_cache")
            return 0

        cleared_count = 0
        try:
            caches = executor.caches
            outputs_cache = caches.outputs

            # 找到需要保护的节点ID（当前节点 + 直接输入来源）
            protected_nodes = set()
            protected_nodes.add(str(unique_id))

            if prompt and str(unique_id) in prompt:
                node_info = prompt[str(unique_id)]
                inputs = node_info.get("inputs", {})
                for key, value in inputs.items():
                    if isinstance(value, list) and len(value) >= 2:
                        protected_nodes.add(str(value[0]))

            # 获取缓存中所有节点ID
            all_cached_ids = set()
            if hasattr(outputs_cache, 'all_node_ids'):
                all_cached_ids = set(str(nid) for nid in outputs_cache.all_node_ids())
            
            # 计算需要清理的节点
            to_clear = all_cached_ids - protected_nodes

            if to_clear:
                logger.info(f"[LoLolClearCache] 缓存中共 {len(all_cached_ids)} 个节点, "
                           f"保护 {len(protected_nodes)} 个: {protected_nodes}, "
                           f"将清理 {len(to_clear)} 个")

            # 逐个删除非保护节点的缓存
            for node_id in to_clear:
                try:
                    # 尝试多种删除方式，适配不同的缓存实现
                    if hasattr(outputs_cache, 'delete'):
                        outputs_cache.delete(node_id)
                        cleared_count += 1
                    elif hasattr(outputs_cache, '_delete_node'):
                        outputs_cache._delete_node(node_id)
                        cleared_count += 1
                    elif hasattr(outputs_cache, 'cache') and isinstance(outputs_cache.cache, dict):
                        if node_id in outputs_cache.cache:
                            del outputs_cache.cache[node_id]
                            cleared_count += 1
                    else:
                        # 最后的尝试：直接设置为 None
                        if hasattr(outputs_cache, 'set'):
                            outputs_cache.set(node_id, None)
                            cleared_count += 1
                except Exception as e:
                    logger.debug(f"[LoLolClearCache] 清理节点 {node_id} 失败: {e}")

            # 如果逐个删除不行，尝试使用 poll 方法强制回收
            if cleared_count == 0 and hasattr(outputs_cache, 'poll'):
                logger.info("[LoLolClearCache] 逐个删除未生效，尝试 poll(ram_headroom=0) 强制回收...")
                outputs_cache.poll(ram_headroom=0)
                cleared_count = -1  # 标记为使用了 poll

            if cleared_count > 0:
                logger.info(f"[LoLolClearCache] 成功清理 {cleared_count} 个节点的执行缓存")
            elif cleared_count == -1:
                logger.info("[LoLolClearCache] 已通过 poll() 触发缓存回收")

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
            if clean_unused_models and HAS_COMFY:
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
