"""
LoLolClearCache - 透明缓存清理节点（增强版 v5）
=================================================

v5 改进：
  v4 只能清理 depth=0 层的大内存节点，因为在递归进入 subcache 后
  用顶层 prompt 的 node_id 无法匹配 subcache 内部的 cache_key。

  v5 的做法：在每一层 subcache 中，通过 cache_key_set.keys 字典
  获取该层所有注册的 node_id，再从 dynprompt 查询其 class_type，
  判断是否属于大内存节点类型，然后用 get_data_key 获取正确的
  cache_key 去删除。这样无论循环嵌套多深都能正确清理。
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

# 默认需要清理的大内存节点类型
DEFAULT_HEAVY_NODE_TYPES = {
    "VAEDecode",
    "VAEDecodeTiled",
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "SamplerCustomAdvanced",
    "ImageFromBatch",
    "ImageBatch",
    "LoloVideoSaveOutput",
    "LoloColorMatch",
    "WanInfiniteTalkToVideo",
    "WanInfiniteTalkToVideoEx",
}


def _get_mem_info():
    """获取当前内存使用信息的字符串"""
    lines = []
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        lines.append(f"RAM: {mem.used / 1024 ** 3:.2f}/{mem.total / 1024 ** 3:.2f} GB ({mem.percent:.1f}%)")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
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


def _find_heavy_keys_in_cache(cache_obj, heavy_types):
    """
    在当前缓存层级中找到属于大内存类型的 cache_key 列表。

    通过 cache_key_set.subcache_keys 获取 (node_id, class_type) 映射，
    或通过 cache_key_set.keys 获取 node_id 然后查 dynprompt。

    返回: 需要删除的 cache_key 列表
    """
    keys_to_delete = []

    if not hasattr(cache_obj, 'cache_key_set') or not hasattr(cache_obj, 'cache'):
        return keys_to_delete

    cache_key_set = cache_obj.cache_key_set

    # 方法1: 通过 subcache_keys 获取 class_type（最可靠）
    # subcache_keys 的值是 (node_id, class_type)
    if hasattr(cache_key_set, 'subcache_keys'):
        node_to_class = {}
        for node_id, subcache_key_val in cache_key_set.subcache_keys.items():
            if isinstance(subcache_key_val, tuple) and len(subcache_key_val) >= 2:
                node_to_class[node_id] = subcache_key_val[1]  # class_type

        # 对于在 subcache_keys 中找到 class_type 的节点
        for node_id, class_type in node_to_class.items():
            if class_type in heavy_types:
                cache_key = cache_key_set.get_data_key(node_id)
                if cache_key is not None and cache_key in cache_obj.cache:
                    keys_to_delete.append((cache_key, node_id, class_type))

    # 方法2: 对于在 keys 中注册但不在 subcache_keys 中的节点，
    # 尝试通过 dynprompt 查 class_type
    if hasattr(cache_key_set, 'keys') and hasattr(cache_obj, 'dynprompt'):
        dynprompt = cache_obj.dynprompt
        for node_id in cache_key_set.keys:
            # 跳过已处理的
            cache_key = cache_key_set.get_data_key(node_id)
            if cache_key is None or cache_key not in cache_obj.cache:
                continue
            # 检查是否已在 keys_to_delete 中
            if any(k[0] == cache_key for k in keys_to_delete):
                continue

            try:
                if hasattr(dynprompt, 'has_node') and dynprompt.has_node(node_id):
                    node = dynprompt.get_node(node_id)
                    class_type = node.get("class_type", "")
                    if class_type in heavy_types:
                        keys_to_delete.append((cache_key, node_id, class_type))
            except Exception:
                pass

    return keys_to_delete


def _recursive_selective_clear(cache_obj, heavy_types, depth=0):
    """
    递归遍历缓存层级，在每一层中：
    1. 找到属于大内存类型的 cache_key 并删除
    2. 清理历史 subcache（for-loop 的旧轮次）
    3. 递归进入剩余的 subcache 继续清理

    返回: (cleared_count, total_count)
    """
    cleared = 0
    total = 0
    prefix = "  " * depth

    # 步骤1: 在当前层级清理大内存节点的缓存
    if hasattr(cache_obj, 'cache') and isinstance(cache_obj.cache, dict):
        total += len(cache_obj.cache)

        heavy_keys = _find_heavy_keys_in_cache(cache_obj, heavy_types)

        for cache_key, node_id, class_type in heavy_keys:
            try:
                del cache_obj.cache[cache_key]
                cleared += 1
                # 清理辅助字典
                if hasattr(cache_obj, 'used_generation') and cache_key in cache_obj.used_generation:
                    del cache_obj.used_generation[cache_key]
                if hasattr(cache_obj, 'timestamps') and cache_key in cache_obj.timestamps:
                    del cache_obj.timestamps[cache_key]
                if hasattr(cache_obj, 'children') and cache_key in cache_obj.children:
                    del cache_obj.children[cache_key]
            except Exception as e:
                logger.debug(f"{prefix}删除失败 node={node_id} type={class_type}: {e}")

        if heavy_keys:
            logger.info(f"{prefix}[depth={depth}] 清理了 {len(heavy_keys)} 个大内存条目 "
                        f"(类型: {set(k[2] for k in heavy_keys)})")

    # 步骤2: 处理 subcaches
    if hasattr(cache_obj, 'subcaches') and isinstance(cache_obj.subcaches, dict):
        subcache_count = len(cache_obj.subcaches)

        if subcache_count > 1:
            # 清理历史 subcache，只保留最后一个（当前轮次）
            sorted_keys = sorted(cache_obj.subcaches.keys())
            keys_to_remove = sorted_keys[:-1]
            for key in keys_to_remove:
                del cache_obj.subcaches[key]
                cleared += 1
            if keys_to_remove:
                logger.info(f"{prefix}[depth={depth}] 清理了 {len(keys_to_remove)}/{subcache_count} 个历史 subcache")

        # 步骤3: 递归进入剩余的 subcache
        for subcache_key, subcache in list(cache_obj.subcaches.items()):
            sub_cleared, sub_total = _recursive_selective_clear(subcache, heavy_types, depth + 1)
            cleared += sub_cleared
            total += sub_total

    return cleared, total


class LoLolClearCache:
    """
    透明缓存清理节点（增强版 v5）
    在每一层缓存中通过 class_type 识别并清理大内存节点，
    同时清理 for-loop 的历史 subcache。
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
    DESCRIPTION = "透传节点，在每一层缓存中清理大内存节点（VAEDecode、采样器等），保留轻量节点缓存。在循环工作流中务必开启 clean_comfy_cache。"

    def _clear_comfy_execution_cache(self, unique_id, prompt):
        """递归清理所有层级中的大内存节点缓存。"""
        executor = _find_prompt_executor()
        if executor is None:
            logger.warning("[LoLolClearCache] 无法找到 PromptExecutor，跳过执行缓存清理")
            if HAS_COMFY and hasattr(comfy.model_management, 'soft_empty_cache'):
                comfy.model_management.soft_empty_cache()
                logger.info("[LoLolClearCache] 回退: 已执行 soft_empty_cache")
            return 0

        try:
            outputs_cache = executor.caches.outputs
            logger.info(f"[LoLolClearCache] 缓存类型: {type(outputs_cache).__name__}")

            # 递归清理所有层级
            cleared, total = _recursive_selective_clear(outputs_cache, DEFAULT_HEAVY_NODE_TYPES)

            if cleared > 0:
                logger.info(f"[LoLolClearCache] 总计清理 {cleared} 个缓存条目（共 {total} 个）")
            else:
                logger.info(f"[LoLolClearCache] 未清理任何条目（共 {total} 个缓存条目）")

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

            # 1. 清理 ComfyUI 执行缓存
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
    DESCRIPTION = "透传节点，带自定义标签，在每一层缓存中清理大内存节点。"

    def process(self, label, clean_cuda, clean_memory, clean_unused_models, clean_comfy_cache=True,
                unique_id=None, prompt=None, **kwargs):
        inputs = [kwargs.get(f"input_{i}") for i in range(1, 6)]
        non_none_inputs = [f"input_{i}" for i, v in enumerate(inputs, start=1) if v is not None]
        logger.info(f"[LoLolClearCache] ({label}) 收到输入: {', '.join(non_none_inputs) or '无'}")

        self.clear_cache(clean_cuda, clean_memory, clean_unused_models,
                         clean_comfy_cache, unique_id, prompt)
        logger.info(f"[LoLolClearCache] ({label}) 清理完成")

        return tuple(inputs)