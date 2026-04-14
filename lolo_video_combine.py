import os
import subprocess
import tempfile
import re
import struct
import numpy as np
import torch
import folder_paths
from .lolo_ffmpeg_utils import get_ffmpeg_path


class LoloVideoCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_dir": ("STRING", {"default": "segments", "multiline": False}),
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "combined_video"}),
                "blend_frames": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 30,
                    "step": 1,
                    "tooltip": "相邻片段衔接处的帧混合数。两段视频的尾部/头部各取这么多帧做像素级线性混合，实现自然过渡。0=不混合（直接拼接）。推荐4~8帧。"
                }),
            },
            "optional": {
                "any": ("*",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_path",)
    FUNCTION = "combine"
    CATEGORY = "LoLo Nodes/video"

    def __init__(self):
        self.ffmpeg_path = None
        self._init_ffmpeg()

    def _init_ffmpeg(self):
        try:
            self.ffmpeg_path = get_ffmpeg_path()
            print(f"[LoloVideoCombine] ffmpeg: {self.ffmpeg_path}")
        except RuntimeError as e:
            raise RuntimeError(f"节点初始化失败: {e}")

    def _get_video_info(self, video_path):
        """获取视频的帧率、宽高"""
        cmd = [self.ffmpeg_path, "-i", video_path, "-f", "null", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = result.stderr

        fps = 25.0
        width, height = 0, 0

        fps_match = re.search(r"(\d+(?:\.\d+)?)\s+fps", output)
        if fps_match:
            fps = float(fps_match.group(1))

        # 解析分辨率，如 "736x1024"
        res_match = re.search(r"(\d{2,5})x(\d{2,5})", output)
        if res_match:
            width = int(res_match.group(1))
            height = int(res_match.group(2))

        return fps, width, height

    def _decode_video_to_frames(self, video_path, width, height):
        """
        用 ffmpeg 将视频解码为 numpy 帧数组。
        返回: np.ndarray, shape = (num_frames, height, width, 3), dtype=uint8
        """
        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-v", "error",
            "-"
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"解码视频失败: {proc.stderr.decode('utf-8', errors='ignore')}")

        raw = proc.stdout
        frame_size = width * height * 3
        num_frames = len(raw) // frame_size
        if num_frames == 0:
            raise RuntimeError(f"视频解码后无帧: {video_path}")

        frames = np.frombuffer(raw[:num_frames * frame_size], dtype=np.uint8)
        frames = frames.reshape(num_frames, height, width, 3)
        return frames

    def _encode_frames_to_video(self, frames, output_path, fps, width, height):
        """
        用 ffmpeg 将 numpy 帧数组编码为视频。
        frames: np.ndarray, shape = (num_frames, height, width, 3), dtype=uint8
        """
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            output_path
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(input=frames.tobytes())
        if proc.returncode != 0:
            raise RuntimeError(f"编码视频失败:\n{stderr.decode('utf-8', errors='ignore')}")

    def _combine_with_blend(self, video_files, blend_frames, temp_video):
        """
        像素级帧混合拼接。
        
        对于相邻的两段视频 A 和 B：
        - 取 A 的最后 blend_frames 帧
        - 取 B 的前 blend_frames 帧
        - 在重叠区域按帧做线性混合：
            blended[i] = A[i] * (1 - alpha) + B[i] * alpha
            其中 alpha 从 0 线性增加到 1
        - 最终序列：A[:-blend] + blended + B[blend:]
        
        这比 crossfade（淡入淡出）更自然，因为混合区域的每一帧
        都同时包含两段的纹理细节，而不是简单的透明度叠加。
        """
        fps, width, height = self._get_video_info(video_files[0])
        print(f"[LoloVideoCombine] 帧混合模式: {blend_frames}帧, {fps}fps, {width}x{height}")

        # 解码所有视频片段
        all_segments = []
        for i, vf in enumerate(video_files):
            frames = self._decode_video_to_frames(vf, width, height)
            print(f"[LoloVideoCombine]   片段{i}: {frames.shape[0]}帧")
            all_segments.append(frames)

        # 逐步合并
        result_frames = all_segments[0]

        for i in range(1, len(all_segments)):
            next_segment = all_segments[i]

            # 确保blend_frames不超过任何一段的帧数
            actual_blend = min(blend_frames, result_frames.shape[0], next_segment.shape[0])

            if actual_blend <= 0:
                # 无法混合，直接拼接
                result_frames = np.concatenate([result_frames, next_segment], axis=0)
                continue

            # 分离各部分
            head = result_frames[:-actual_blend]          # A的非重叠部分
            overlap_a = result_frames[-actual_blend:]     # A的重叠帧
            overlap_b = next_segment[:actual_blend]       # B的重叠帧
            tail = next_segment[actual_blend:]            # B的非重叠部分

            # 逐帧线性混合
            blended = np.empty_like(overlap_a)
            for j in range(actual_blend):
                alpha = (j + 1) / (actual_blend + 1)  # 从接近0到接近1，不包含端点
                blended[j] = (overlap_a[j].astype(np.float32) * (1 - alpha) +
                              overlap_b[j].astype(np.float32) * alpha).astype(np.uint8)

            # 拼合
            result_frames = np.concatenate([head, blended, tail], axis=0)

            # 释放已处理的片段
            del next_segment, overlap_a, overlap_b, blended

            print(f"[LoloVideoCombine]   合并片段{i}: 混合{actual_blend}帧, "
                  f"当前总帧数={result_frames.shape[0]}")

        # 编码为视频
        print(f"[LoloVideoCombine] 编码最终视频: {result_frames.shape[0]}帧")
        self._encode_frames_to_video(result_frames, temp_video, fps, width, height)
        del result_frames

    def combine(self, video_dir, audio, filename_prefix, blend_frames=6, any=None):
        # ---------- 智能路径解析 ----------
        if not os.path.isabs(video_dir):
            video_dir = os.path.join(folder_paths.get_output_directory(), video_dir)
            print(f"[LoloVideoCombine] 解析相对路径为: {video_dir}")

        if not os.path.isdir(video_dir):
            raise NotADirectoryError(f"目录不存在: {video_dir}")

        # ---------- 获取视频文件列表 ----------
        files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
        files.sort()
        if not files:
            raise RuntimeError(f"目录中没有视频文件: {video_dir}")

        video_files = [os.path.join(video_dir, f) for f in files]
        print(f"[LoloVideoCombine] 找到 {len(files)} 个视频片段")

        # ---------- 自动生成输出路径 ----------
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        counter = 1
        while True:
            out_path = os.path.join(output_dir, f"{filename_prefix}_{counter:05d}.mp4")
            if not os.path.exists(out_path):
                break
            counter += 1

        # ---------- 临时文件变量预定义 ----------
        temp_video = None
        audio_file = None

        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                temp_video = tmp.name

            # ---------- 拼接视频 ----------
            if blend_frames > 0 and len(video_files) > 1:
                print(f"[LoloVideoCombine] 使用帧混合模式 ({blend_frames}帧重叠混合)")
                self._combine_with_blend(video_files, blend_frames, temp_video)
            else:
                # 原有的直接拼接逻辑
                print(f"[LoloVideoCombine] 使用直接拼接模式")
                list_file = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                        list_file = f.name
                        for file in files:
                            full_path = os.path.join(video_dir, file)
                            full_path = full_path.replace('\\', '/')
                            if ' ' in full_path:
                                f.write(f'file "{full_path}"\n')
                            else:
                                f.write(f'file {full_path}\n')

                    try:
                        subprocess.run([self.ffmpeg_path, "-f", "concat", "-safe", "0",
                                       "-i", list_file, "-c", "copy", "-y", temp_video],
                                       check=True, capture_output=True)
                        print(f"[LoloVideoCombine] 流复制成功")
                    except subprocess.CalledProcessError:
                        subprocess.run([self.ffmpeg_path, "-f", "concat", "-safe", "0",
                                       "-i", list_file,
                                       "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                                       "-pix_fmt", "yuv420p",
                                       "-c:a", "aac", "-b:a", "192k",
                                       "-y", temp_video],
                                       check=True, capture_output=True)
                        print(f"[LoloVideoCombine] 重新编码成功")
                finally:
                    if list_file and os.path.exists(list_file):
                        os.remove(list_file)

            # ---------- 处理音频 ----------
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            samples = waveform.shape[1]
            channels = waveform.shape[0]
            audio_data = waveform.t().contiguous().numpy().astype(np.float32)

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                audio_file = tmp_audio.name

            ffmpeg_audio_cmd = [
                self.ffmpeg_path,
                "-y",
                "-f", "f32le",
                "-ar", str(sample_rate),
                "-ac", str(channels),
                "-i", "-",
                "-c:a", "pcm_s16le",
                "-f", "wav",
                audio_file
            ]
            proc_audio = subprocess.Popen(ffmpeg_audio_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            proc_audio.stdin.write(audio_data.tobytes())
            proc_audio.stdin.close()
            proc_audio.wait()
            if proc_audio.returncode != 0:
                stderr = proc_audio.stderr.read().decode('utf-8', errors='ignore')
                raise RuntimeError(f"ffmpeg 音频编码失败 (返回码 {proc_audio.returncode}):\n{stderr}")

            # ---------- 合并音频到最终视频 ----------
            subprocess.run([self.ffmpeg_path, "-i", temp_video, "-i", audio_file,
                           "-c:v", "copy", "-c:a", "aac",
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-shortest", "-y", out_path],
                           check=True, capture_output=True)

        except Exception as e:
            print(f"[LoloVideoCombine] 处理失败: {e}")
            raise e
        finally:
            for f in [temp_video, audio_file]:
                if f is not None and os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"[LoloVideoCombine] 临时文件删除失败（可忽略）: {e}")

        return (out_path,)
