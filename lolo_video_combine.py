import os
import subprocess
import tempfile
import re
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
                "crossfade_frames": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 30,
                    "step": 1,
                    "tooltip": "相邻片段衔接处的交叉淡入淡出帧数。0=不做crossfade（直接拼接），4~8帧推荐。"
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

    def _get_video_fps(self, video_path):
        """用 ffprobe/ffmpeg 获取视频帧率"""
        try:
            cmd = [self.ffmpeg_path, "-i", video_path, "-f", "null", "-"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            output = result.stderr
            fps_match = re.search(r"(\d+(?:\.\d+)?)\s+fps", output)
            if fps_match:
                return float(fps_match.group(1))
        except Exception:
            pass
        return 25.0  # 默认帧率

    def _get_video_duration(self, video_path):
        """获取视频时长（秒）"""
        try:
            cmd = [self.ffmpeg_path, "-i", video_path, "-f", "null", "-"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            output = result.stderr
            duration_match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", output)
            if duration_match:
                h, m, s = duration_match.groups()
                return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            pass
        return 0.0

    def _combine_with_crossfade(self, video_files, crossfade_frames, temp_video):
        """
        使用 ffmpeg 的 xfade 滤镜逐步拼接视频片段，在衔接处做交叉淡入淡出。
        
        原理：
        - 先获取每段视频的帧率和时长
        - 用 xfade 滤镜在每两段之间做 crossfade
        - xfade 的 offset = 第一段视频时长 - crossfade时长
        - 多段视频需要链式应用 xfade
        """
        if len(video_files) < 2:
            # 只有一个文件，直接复制
            subprocess.run([self.ffmpeg_path, "-i", video_files[0],
                           "-c", "copy", "-y", temp_video],
                          check=True, capture_output=True)
            return

        fps = self._get_video_fps(video_files[0])
        crossfade_duration = crossfade_frames / fps
        print(f"[LoloVideoCombine] Crossfade: {crossfade_frames}帧 = {crossfade_duration:.3f}秒 @ {fps}fps")

        # 获取每段视频的时长
        durations = []
        for f in video_files:
            d = self._get_video_duration(f)
            durations.append(d)
            print(f"[LoloVideoCombine]   {os.path.basename(f)}: {d:.3f}秒")

        # 逐步拼接：先拼前两个，再把结果和第三个拼，以此类推
        # 这样避免 ffmpeg filter_complex 过于复杂
        current_input = video_files[0]
        current_duration = durations[0]
        temp_files = []

        try:
            for i in range(1, len(video_files)):
                next_video = video_files[i]
                next_duration = durations[i]

                # 计算 xfade 的 offset（在第一段结束前 crossfade_duration 秒开始过渡）
                offset = max(0, current_duration - crossfade_duration)

                # 创建临时输出
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    step_output = tmp.name
                    temp_files.append(step_output)

                # 最后一步直接输出到 temp_video
                if i == len(video_files) - 1:
                    step_output = temp_video

                cmd = [
                    self.ffmpeg_path,
                    "-i", current_input,
                    "-i", next_video,
                    "-filter_complex",
                    f"[0:v][1:v]xfade=transition=fade:duration={crossfade_duration:.4f}:offset={offset:.4f},format=yuv420p[v]",
                    "-map", "[v]",
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-an",  # 音频后面再加
                    "-y", step_output
                ]

                print(f"[LoloVideoCombine] 拼接步骤 {i}/{len(video_files)-1}: "
                      f"offset={offset:.3f}s, crossfade={crossfade_duration:.3f}s")

                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    stderr = result.stderr.decode('utf-8', errors='ignore')
                    raise RuntimeError(f"xfade 拼接失败 (步骤 {i}):\n{stderr}")

                # 更新当前输入和时长
                current_input = step_output
                # xfade 后的时长 = 两段时长之和 - crossfade时长
                current_duration = current_duration + next_duration - crossfade_duration

        finally:
            # 清理中间临时文件（但不清理最后输出的 temp_video）
            for f in temp_files:
                if f != temp_video and os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass

        print(f"[LoloVideoCombine] Crossfade 拼接完成，总时长: {current_duration:.3f}秒")

    def combine(self, video_dir, audio, filename_prefix, crossfade_frames=4, any=None):
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
            # ---------- 临时无音频视频 ----------
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                temp_video = tmp.name

            # ---------- 拼接视频 ----------
            if crossfade_frames > 0 and len(video_files) > 1:
                # 使用 crossfade 拼接
                print(f"[LoloVideoCombine] 使用 crossfade 模式拼接 ({crossfade_frames} 帧过渡)")
                self._combine_with_crossfade(video_files, crossfade_frames, temp_video)
            else:
                # 原有的直接拼接逻辑
                print(f"[LoloVideoCombine] 使用直接拼接模式（无 crossfade）")
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

            # ---------- 处理音频（使用 ffmpeg 直接编码为 wav）----------
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
