"""
LoLolClearCache - 透明缓存清理节点（增强版 v3）
=================================================

核心改进 v3：
  直接操作 PromptExecutor.caches.outputs.cache 字典来删除缓存条目。
  
  之前的版本尝试了：
  - v1: 通过 PromptQueue.currently_running 找 executor → 失败（executor 不在里面）
  - v2: 通过栈帧找到 executor，尝试 delete/poll → 找到了 executor 但：
        - delete 方法不存在
        - poll(ram_headroom=0) 因为可用内存充足不会触发清理
  
  v3 的做法：
  找到 executor 后，递归遍历 caches.outputs 的 cache 字典和 subcaches，
  直接 del 掉不需要保留的 cache_key 对应的条目。对于 for-loop 产生的
  深层 subcache，也会递归清理。
"""

import torch
import gc
import sys
import time
import logging

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import comfy.model_management
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False

try:
    from execution import PromptExecutor
    HAS_EXECUTOR_CLASS = True
except ImportError:
    HAS_EXECUTOR_CLASS = False

logger = logging.getLogger("LoLolClearCache")

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
    """通过遍历所有线程的栈帧，找到 prompt_worker 中的 PromptExecutor 实例。"""
    global _cached_executor

    if _cached_executor is not None:
        if hasattr(_cached_executor, 'caches') and hasattr(_cached_executor.caches, 'outputs'):
            return _cached_executor
        else:
            _cached_executor = None

    if not HAS_EXECUTOR_CLASS:
        return None

    for thread_id, frame in sys._current_frames().items():
        current_frame = frame
        while current_frame is not None:
            for var_name, var_value in current_frame.f_locals.items():
                if isinstance(var_value, PromptExecutor):
                    logger.info(f"[LoLolClearCache] 找到 PromptExecutor: "
                                f"线程={thread_id}, 帧={current_frame.f_code.co_name}, 变量名={var_name}")
                    _cached_executor = var_value
                    return var_value
            current_frame = current_frame.f_back

    logger.warning("[LoLolClearCache] 未能通过栈帧找到 PromptExecutor")
    return None


def _recursive_clear_cache(cache_obj, protected_node_ids, depth=0):
    """
    递归清理 BasicCache / HierarchicalCache / LRUCache / RAMPressureCache 的
    cache 字典和 subcaches。
    
    参数：
        cache_obj: BasicCache 或其子类实例
        protected_node_ids: 需要保护的 node_id 集合（这些节点的缓存不删除）
        depth: 递归深度（用于日志）
    
    返回：
        (cleared_count, total_count) 清理数和总数
    """
    cleared = 0
    total = 0
    prefix = "  " * depth

    # 清理当前层级的 cache 字典
    if hasattr(cache_obj, 'cache') and isinstance(cache_obj.cache, dict):
        keys_to_delete = []
        for cache_key in list(cache_obj.cache.keys()):
            total += 1
            # 检查这个 cache_key 是否对应一个受保护的节点
            # cache_key 可能是 node_id 本身，也可能是签名哈希
            # 我们无法反向映射 cache_key -> node_id，所以采用保守策略：
            # 如果 cache_key_set 存在，尝试通过它判断；否则全部清理
            is_protected = False
            if hasattr(cache_obj, 'cache_key_set'):
                for node_id in protected_node_ids:
                    try:
                        protected_key = cache_obj.cache_key_set.get_data_key(node_id)
                        if protected_key == cache_key:
                            is_protected = True
                            break
                    except Exception:
                        pass

            if not is_protected:
                keys_to_delete.append(cache_key)

        for key in keys_to_delete:
            try:
                del cache_obj.cache[key]
                cleared += 1
                # 同时清理 LRUCache 的辅助字典
                if hasattr(cache_obj, 'used_generation') and key in cache_obj.used_generation:
                    del cache_obj.used_generation[key]
                if hasattr(cache_obj, 'timestamps') and key in cache_obj.timestamps:
                    del cache_obj.timestamps[key]
                if hasattr(cache_obj, 'children') and key in cache_obj.children:
                    del cache_obj.children[key]
            except Exception as e:
                logger.debug(f"{prefix}删除 cache key 失败: {e}")

        if keys_to_delete:
            logger.info(f"{prefix}[depth={depth}] 清理了 {len(keys_to_delete)}/{total} 个 cache 条目")

    # 递归清理 subcaches
    if hasattr(cache_obj, 'subcaches') and isinstance(cache_obj.subcaches, dict):
        subcache_count = len(cache_obj.subcaches)
        if subcache_count > 0:
            # 对于 for-loop，subcaches 中保存了每轮迭代的子缓存
            # 我们需要保留当前轮次的 subcache，清理历史轮次的
            # 策略：只保留最后一个 subcache（当前轮次），清理其余
            if subcache_count > 1:
                sorted_keys = sorted(cache_obj.subcaches.keys())
                # 保留最后一个（当前轮次）
                keys_to_remove = sorted_keys[:-1]
                for key in keys_to_remove:
                    del cache_obj.subcaches[key]
                    cleared += 1
                if keys_to_remove:
                    logger.info(f"{prefix}[depth={depth}] 清理了 {len(keys_to_remove)}/{subcache_count} 个历史 subcache")

            # 递归清理剩余的 subcache
            for subcache_key, subcache in list(cache_obj.subcaches.items()):
                sub_cleared, sub_total = _recursive_clear_cache(subcache, protected_node_ids, depth + 1)
                cleared += sub_cleared
                total += sub_total

    return cleared, total


class LoLolClearCache:
    """
    透明缓存清理节点（增强版 v3）
    直接操作 PromptExecutor 的缓存字典，彻底释放循环中历史轮次的节点输出。
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
        直接操作 cache 字典和 subcaches 来释放历史数据。
        """
        executor = _find_prompt_executor()
        if executor is None:
            logger.warning("[LoLolClearCache] 无法找到 PromptExecutor，跳过执行缓存清理")
            if HAS_COMFY and hasattr(comfy.model_management, 'soft_empty_cache'):
                comfy.model_management.soft_empty_cache()
                logger.info("[LoLolClearCache] 回退: 已执行 soft_empty_cache")
            return 0

        try:
            outputs_cache = executor.caches.outputs

            # 找到需要保护的节点ID
            protected_nodes = set()
            protected_nodes.add(str(unique_id))
            if prompt and str(unique_id) in prompt:
                node_info = prompt[str(unique_id)]
                inputs = node_info.get("inputs", {})
                for key, value in inputs.items():
                    if isinstance(value, list) and len(value) >= 2:
                        protected_nodes.add(str(value[0]))

            logger.info(f"[LoLolClearCache] 保护节点: {protected_nodes}")
            logger.info(f"[LoLolClearCache] 缓存类型: {type(outputs_cache).__name__}")

            # 递归清理缓存
            cleared, total = _recursive_clear_cache(outputs_cache, protected_nodes)

            if cleared > 0:
                logger.info(f"[LoLolClearCache] 总计清理 {cleared} 个缓存条目（共 {total} 个）")
            else:
                logger.info(f"[LoLolClearCache] 无需清理（共 {total} 个缓存条目）")

            return cleared

        except Exception as e:
            logger.error(f"[LoLolClearCache] 清理执行缓存时出错: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def clear_cache(self, clean_cuda, clean_memory, clean_unused_models,
                    clean_comfy_cache=False, unique_id=None, prompt=None):
        """执行缓存清理"""
        try:
            mem_before = _get_mem_info()
            logger.info(f"[LoLolClearCache] 开始清理缓存... 清理前: {mem_before}")

            # 1. 清理 ComfyUI 执行缓存（最重要）
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

            # 3. Python 垃圾回收
            if clean_memory:
                for _ in range(3):
                    gc.collect()
                logger.info("  - 垃圾回收已执行")

            # 4. 清空 CUDA 缓存
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
        inputs = [kwargs.get(f"input_{i}") for i in range(1, 6)]
        non_none_inputs = [f"input_{i}" for i, v in enumerate(inputs, start=1) if v is not None]
        logger.info(f"[LoLolClearCache] 收到输入: {', '.join(non_none_inputs) or '无'}")

        self.clear_cache(clean_cuda, clean_memory, clean_unused_models,
                         clean_comfy_cache, unique_id, prompt)

        return tuple(inputs)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float(time.time())


class LoLolClearCacheWithLabel(LoLolClearCache):
    """带标签的缓存清理节点"""

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
