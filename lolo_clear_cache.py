"""
LoLolClearCache - 透明缓存清理节点（增强版 v4）
=================================================

v4 改进：
  v3 清理太彻底，把 LoloGenerateFilename、SetNode 等轻量节点的缓存也删了，
  导致每轮循环重新生成文件名 → 每段视频创建新文件夹。
  
  v4 改为「白名单清理」策略：
  只清理已知的大内存节点（VAEDecode、采样器、ImageFromBatch 等），
  保留所有轻量节点的缓存。用户可以通过 heavy_node_types 参数自定义
  需要清理的节点类型。
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
# 这些节点的输出是大张量（数百MB~数GB），必须在循环中及时释放
DEFAULT_HEAVY_NODE_TYPES = {
    # VAE 解码输出（109帧 × 1024×576×3 × float32 ≈ 736 MB）
    "VAEDecode",
    "VAEDecodeTiled",
    # 采样器输出（latent 也不小）
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "SamplerCustomAdvanced",
    # 图像处理
    "ImageFromBatch",
    "ImageBatch",
    # 视频保存（内部已做清理但缓存仍会保留引用）
    "LoloVideoSaveOutput",
    # 色彩校正
    "LoloColorMatch",
    # Wan 视频生成（条件和模型补丁，单个不大但包含大量引用）
    "WanInfiniteTalkToVideo",
    "WanInfiniteTalkToVideoEx",
}


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


def _get_heavy_node_ids(prompt, heavy_types):
    """
    从 prompt 中找出所有属于大内存类型的节点ID。
    
    参数：
        prompt: ComfyUI 的 prompt 字典 {node_id: {class_type, inputs, ...}}
        heavy_types: 需要清理的节点类型集合
    
    返回：
        set: 需要清理的节点ID集合
    """
    heavy_ids = set()
    if not prompt:
        return heavy_ids

    for node_id, node_info in prompt.items():
        class_type = node_info.get("class_type", "")
        if class_type in heavy_types:
            heavy_ids.add(str(node_id))

    return heavy_ids


def _selective_clear_cache(cache_obj, heavy_node_ids, prompt, depth=0):
    """
    选择性清理缓存：只删除大内存节点的缓存条目，保留轻量节点。
    同时递归处理 subcaches（for-loop 的历史轮次）。
    
    参数：
        cache_obj: BasicCache 或其子类实例
        heavy_node_ids: 需要清理的节点ID集合
        prompt: ComfyUI 的 prompt 字典
        depth: 递归深度
    
    返回：
        (cleared_count, total_count)
    """
    cleared = 0
    total = 0
    prefix = "  " * depth

    # 清理当前层级的 cache 字典中属于大内存节点的条目
    if hasattr(cache_obj, 'cache') and isinstance(cache_obj.cache, dict) and \
       hasattr(cache_obj, 'cache_key_set'):
        keys_to_delete = []

        for cache_key in list(cache_obj.cache.keys()):
            total += 1
            # 检查这个 cache_key 是否对应一个大内存节点
            should_delete = False
            for node_id in heavy_node_ids:
                try:
                    node_cache_key = cache_obj.cache_key_set.get_data_key(node_id)
                    if node_cache_key == cache_key:
                        should_delete = True
                        break
                except Exception:
                    pass

            if should_delete:
                keys_to_delete.append(cache_key)

        for key in keys_to_delete:
            try:
                del cache_obj.cache[key]
                cleared += 1
                if hasattr(cache_obj, 'used_generation') and key in cache_obj.used_generation:
                    del cache_obj.used_generation[key]
                if hasattr(cache_obj, 'timestamps') and key in cache_obj.timestamps:
                    del cache_obj.timestamps[key]
                if hasattr(cache_obj, 'children') and key in cache_obj.children:
                    del cache_obj.children[key]
            except Exception as e:
                logger.debug(f"{prefix}删除 cache key 失败: {e}")

        if keys_to_delete:
            logger.info(f"{prefix}[depth={depth}] 选择性清理了 {len(keys_to_delete)}/{total} 个大内存缓存条目")

    # 处理 subcaches（for-loop 产生的子缓存）
    if hasattr(cache_obj, 'subcaches') and isinstance(cache_obj.subcaches, dict):
        subcache_count = len(cache_obj.subcaches)
        if subcache_count > 1:
            # 只保留最后一个 subcache（当前轮次），清理历史轮次
            sorted_keys = sorted(cache_obj.subcaches.keys())
            keys_to_remove = sorted_keys[:-1]
            for key in keys_to_remove:
                del cache_obj.subcaches[key]
                cleared += 1
            if keys_to_remove:
                logger.info(f"{prefix}[depth={depth}] 清理了 {len(keys_to_remove)}/{subcache_count} 个历史 subcache")

        # 递归清理剩余的 subcache
        for subcache_key, subcache in list(cache_obj.subcaches.items()):
            sub_cleared, sub_total = _selective_clear_cache(subcache, heavy_node_ids, prompt, depth + 1)
            cleared += sub_cleared
            total += sub_total

    return cleared, total


class LoLolClearCache:
    """
    透明缓存清理节点（增强版 v4）
    只清理大内存节点的缓存，保留文件名生成、SetNode 等轻量节点的缓存。
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
    DESCRIPTION = "透传节点，只清理大内存节点的执行缓存（VAEDecode、采样器等），保留文件名等轻量缓存。在循环工作流中务必开启 clean_comfy_cache。"

    def _clear_comfy_execution_cache(self, unique_id, prompt):
        """选择性清理大内存节点的执行缓存。"""
        executor = _find_prompt_executor()
        if executor is None:
            logger.warning("[LoLolClearCache] 无法找到 PromptExecutor，跳过执行缓存清理")
            if HAS_COMFY and hasattr(comfy.model_management, 'soft_empty_cache'):
                comfy.model_management.soft_empty_cache()
                logger.info("[LoLolClearCache] 回退: 已执行 soft_empty_cache")
            return 0

        try:
            outputs_cache = executor.caches.outputs

            # 从 prompt 中识别大内存节点
            heavy_node_ids = _get_heavy_node_ids(prompt, DEFAULT_HEAVY_NODE_TYPES)
            logger.info(f"[LoLolClearCache] 缓存类型: {type(outputs_cache).__name__}")
            logger.info(f"[LoLolClearCache] 识别到 {len(heavy_node_ids)} 个大内存节点需要清理: "
                       f"{heavy_node_ids}")

            if not heavy_node_ids:
                logger.info("[LoLolClearCache] 未找到需要清理的大内存节点")
                return 0

            # 选择性清理
            cleared, total = _selective_clear_cache(outputs_cache, heavy_node_ids, prompt)

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

            # 1. 选择性清理 ComfyUI 执行缓存
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
    DESCRIPTION = "透传节点，带自定义标签，只清理大内存节点的执行缓存。"

    def process(self, label, clean_cuda, clean_memory, clean_unused_models, clean_comfy_cache=True,
                unique_id=None, prompt=None, **kwargs):
        inputs = [kwargs.get(f"input_{i}") for i in range(1, 6)]
        non_none_inputs = [f"input_{i}" for i, v in enumerate(inputs, start=1) if v is not None]
        logger.info(f"[LoLolClearCache] ({label}) 收到输入: {', '.join(non_none_inputs) or '无'}")

        self.clear_cache(clean_cuda, clean_memory, clean_unused_models,
                         clean_comfy_cache, unique_id, prompt)
        logger.info(f"[LoLolClearCache] ({label}) 清理完成")

        return tuple(inputs)
