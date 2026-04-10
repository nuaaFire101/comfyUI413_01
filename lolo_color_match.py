import torch
import numpy as np
import logging

logger = logging.getLogger("LoloColorMatch")


class LoloColorMatch:
    """
    色彩校正节点：将生成的视频帧的色彩分布向参考图像对齐。

    原理：
    使用 Reinhard 色彩迁移算法，在 LAB 色彩空间中将生成帧的
    均值和标准差对齐到参考图像。这能有效解决生成视频中
    面部惨白、色彩偏移等问题，同时保持画面的自然过渡。

    用法：
    将此节点插入 VAEDecode 之后、VideoSaveOutput 之前。
    - reference_image: 接入原始参考图（和 start_image 相同）
    - images: 接入 VAEDecode 输出的视频帧
    - strength: 控制色彩校正强度，0=不校正，1=完全匹配参考图色彩
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "色彩校正强度。0.0=不校正，0.5~0.8=推荐范围，1.0=完全匹配参考图色彩分布"
                }),
                "mode": (["reinhard_lab", "mean_std_rgb", "histogram_match"], {
                    "default": "reinhard_lab",
                    "tooltip": "校正算法。reinhard_lab=LAB空间色彩迁移(推荐)，mean_std_rgb=RGB均值标准差匹配，histogram_match=直方图匹配"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "可选遮罩，仅对遮罩区域（如面部）进行色彩校正"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "color_match"
    CATEGORY = "LoLoNodes"
    DESCRIPTION = "将生成视频帧的色彩分布向参考图像对齐，解决面部惨白、色彩偏移等问题。"

    def _rgb_to_lab(self, rgb):
        """RGB [0,1] -> LAB，使用近似转换"""
        # RGB -> linear RGB (去gamma)
        linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

        # linear RGB -> XYZ (sRGB D65)
        x = linear[..., 0] * 0.4124564 + linear[..., 1] * 0.3575761 + linear[..., 2] * 0.1804375
        y = linear[..., 0] * 0.2126729 + linear[..., 1] * 0.7151522 + linear[..., 2] * 0.0721750
        z = linear[..., 0] * 0.0193339 + linear[..., 1] * 0.1191920 + linear[..., 2] * 0.9503041

        # XYZ -> LAB (D65 白点)
        x /= 0.95047
        z /= 1.08883

        def f(t):
            delta = 6.0 / 29.0
            return np.where(t > delta ** 3, t ** (1.0 / 3.0), t / (3 * delta ** 2) + 4.0 / 29.0)

        fx, fy, fz = f(x), f(y), f(z)
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return np.stack([L, a, b], axis=-1)

    def _lab_to_rgb(self, lab):
        """LAB -> RGB [0,1]"""
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200

        delta = 6.0 / 29.0

        def f_inv(t):
            return np.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4.0 / 29.0))

        x = 0.95047 * f_inv(fx)
        y = f_inv(fy)
        z = 1.08883 * f_inv(fz)

        # XYZ -> linear RGB
        r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
        g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
        b_ch = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

        linear = np.stack([r, g, b_ch], axis=-1)
        linear = np.clip(linear, 0, None)

        # linear RGB -> sRGB (加gamma)
        rgb = np.where(linear > 0.0031308, 1.055 * (linear ** (1.0 / 2.4)) - 0.055, 12.92 * linear)
        return np.clip(rgb, 0, 1)

    def _reinhard_lab(self, source, reference, strength):
        """
        Reinhard 色彩迁移：在 LAB 空间中匹配均值和标准差。
        这是最经典也是效果最自然的色彩迁移算法。
        """
        src_lab = self._rgb_to_lab(source)
        ref_lab = self._rgb_to_lab(reference)

        # 计算参考图的统计量（只算一次）
        ref_mean = ref_lab.mean(axis=(0, 1))
        ref_std = ref_lab.std(axis=(0, 1)) + 1e-6

        # 对源图的每个通道做标准化后重映射
        src_mean = src_lab.mean(axis=(0, 1))
        src_std = src_lab.std(axis=(0, 1)) + 1e-6

        result_lab = (src_lab - src_mean) / src_std * ref_std + ref_mean

        # 按 strength 混合
        result_lab = src_lab * (1 - strength) + result_lab * strength

        return self._lab_to_rgb(result_lab)

    def _mean_std_rgb(self, source, reference, strength):
        """RGB 空间的均值-标准差匹配，更简单但有时够用"""
        ref_mean = reference.mean(axis=(0, 1))
        ref_std = reference.std(axis=(0, 1)) + 1e-6
        src_mean = source.mean(axis=(0, 1))
        src_std = source.std(axis=(0, 1)) + 1e-6

        result = (source - src_mean) / src_std * ref_std + ref_mean
        result = source * (1 - strength) + result * strength
        return np.clip(result, 0, 1)

    def _histogram_match(self, source, reference, strength):
        """逐通道直方图匹配"""
        result = np.zeros_like(source)
        for c in range(3):
            src_ch = source[..., c].ravel()
            ref_ch = reference[..., c].ravel()

            # 计算 CDF
            src_vals, src_counts = np.unique(
                (src_ch * 255).astype(np.uint8), return_counts=True
            )
            ref_vals, ref_counts = np.unique(
                (ref_ch * 255).astype(np.uint8), return_counts=True
            )

            src_cdf = np.zeros(256)
            ref_cdf = np.zeros(256)
            src_cdf[src_vals] = np.cumsum(src_counts).astype(float) / src_counts.sum()
            ref_cdf[ref_vals] = np.cumsum(ref_counts).astype(float) / ref_counts.sum()

            # 填充空值
            for i in range(1, 256):
                if src_cdf[i] == 0:
                    src_cdf[i] = src_cdf[i - 1]
                if ref_cdf[i] == 0:
                    ref_cdf[i] = ref_cdf[i - 1]

            # 建立映射
            mapping = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                mapping[i] = np.argmin(np.abs(src_cdf[i] - ref_cdf))

            matched = mapping[(src_ch * 255).astype(np.uint8)].reshape(source[..., c].shape)
            result[..., c] = matched / 255.0

        result = source * (1 - strength) + result * strength
        return np.clip(result, 0, 1)

    def color_match(self, images, reference_image, strength=0.7, mode="reinhard_lab", mask=None):
        """
        对批量图像进行色彩校正。（内存优化版：逐帧处理 + 原地写入）
        """
        import gc

        if strength == 0.0:
            return (images,)

        # 只转换参考图到 numpy（很小，只有1帧）
        ref_np = reference_image[0].cpu().numpy()  # [H, W, C]

        # 预处理 mask（只做一次）
        mask_3d = None
        if mask is not None:
            mask_np = mask[0].cpu().numpy()
            if mask_np.shape != (images.shape[1], images.shape[2]):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((images.shape[2], images.shape[1]), PILImage.BILINEAR)
                mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            mask_3d = mask_np[..., np.newaxis]

        # 选择算法
        match_fn = {
            "reinhard_lab": self._reinhard_lab,
            "mean_std_rgb": self._mean_std_rgb,
            "histogram_match": self._histogram_match,
        }[mode]

        batch_size = images.shape[0]

        # 逐帧处理，避免同时持有整个批次的两份副本
        result_frames = []
        for i in range(batch_size):
            frame_np = images[i].cpu().numpy()  # 单帧 ~6.8 MB
            corrected = match_fn(frame_np, ref_np, strength).astype(np.float32)

            if mask_3d is not None:
                corrected = frame_np * (1 - mask_3d) + corrected * mask_3d

            result_frames.append(torch.from_numpy(corrected))
            del frame_np, corrected  # 立即释放

        result_tensor = torch.stack(result_frames).to(images.device, dtype=images.dtype)
        del result_frames
        gc.collect()

        logger.info(
            f"[LoloColorMatch] 已对 {batch_size} 帧进行色彩校正 "
            f"(mode={mode}, strength={strength:.2f})"
        )

        return (result_tensor,)