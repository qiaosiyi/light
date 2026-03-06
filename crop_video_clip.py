"""
根据 LabelMe 标注的矩形区域，用 ffmpeg 将每个原始视频裁剪为红绿灯区域的小视频。
帧率不变，全时长输出，用于模型推理输入。

用法:
    python crop_video_clip.py [--video_root PATH] [--label_root PATH]
                              [--output_dir PATH] [--crf N]
"""

import json
import shutil
import subprocess
import argparse
from pathlib import Path


def get_ffmpeg_exe() -> str:
    """查找 ffmpeg 可执行文件：优先 PATH，其次 imageio-ffmpeg 内置。"""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return ""

# ======================== 默认配置 ========================
# Windows 测试（视频平放在 orgin-video/ 下）
DEFAULT_VIDEO_ROOT = Path("orgin-video")
# Linux 正式运行时改为:
# DEFAULT_VIDEO_ROOT = Path("/media/zekai/Expansion/Experiment data CHAO_MAI")

DEFAULT_LABEL_ROOT = Path("Label_prepare")
DEFAULT_OUTPUT_DIR = Path("cropped-video")
DEFAULT_CRF = 23

VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"}


def parse_args():
    parser = argparse.ArgumentParser(description="用 ffmpeg 裁剪视频中的红绿灯区域")
    parser.add_argument("--video_root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--label_root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF,
                        help="编码质量 (越小质量越高, 默认 23)")
    parser.add_argument("--gpu", action="store_true",
                        help="使用 NVIDIA GPU 硬件编码 (h264_nvenc)")
    return parser.parse_args()


def check_nvenc(ffmpeg: str) -> bool:
    """检测 ffmpeg 是否支持 h264_nvenc。"""
    try:
        r = subprocess.run([ffmpeg, "-encoders"], capture_output=True, text=True, timeout=10)
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


def find_labels(label_root: Path) -> dict:
    """遍历 Label_prepare 下所有 JSON，返回 {video_stem: rect} 字典。"""
    labels = {}
    for json_path in label_root.rglob("*.json"):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        shapes = data.get("shapes", [])
        if not shapes:
            continue

        for shape in shapes:
            if shape.get("shape_type") != "rectangle" or shape.get("label") != "Lamp":
                continue
            pts = shape["points"]
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            rect = (int(min(x1, x2)), int(min(y1, y2)),
                    int(max(x1, x2)), int(max(y1, y2)))

            stem = json_path.stem
            video_stem = stem.rsplit("_t", 1)[0]
            labels[video_stem] = rect
            break

    return labels


def find_videos(video_root: Path) -> list:
    """递归查找所有视频文件。"""
    videos = []
    for p in video_root.rglob("*"):
        if p.is_file() and p.suffix in VIDEO_EXTS:
            videos.append(p)
    return videos


NVENC_MIN_SIZE = 160


def calc_crop(rect: tuple, img_w: int, img_h: int, use_gpu: bool) -> tuple:
    """计算裁剪参数 (x, y, w, h)。GPU 模式下扩展到 NVENC 最小尺寸，CPU 模式保持原始大小。"""
    x1, y1, x2, y2 = rect
    min_size = NVENC_MIN_SIZE if use_gpu else 0

    if (x2 - x1) < min_size:
        pad = min_size - (x2 - x1)
        x1 -= pad // 2
        x2 = x1 + min_size
    if (y2 - y1) < min_size:
        pad = min_size - (y2 - y1)
        y1 -= pad // 2
        y2 = y1 + min_size

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h

    x1 = max(0, x1)
    y1 = max(0, y1)

    # 偶数对齐
    w = (x2 - x1) // 2 * 2
    h = (y2 - y1) // 2 * 2
    return x1, y1, w, h


def crop_video_ffmpeg(ffmpeg: str, video_path: Path, rect: tuple,
                      output_path: Path, crf: int, use_gpu: bool = False,
                      img_w: int = 1920, img_h: int = 1080) -> bool:
    """用 ffmpeg crop 滤镜裁剪视频。"""
    x1, y1, w, h = calc_crop(rect, img_w, img_h, use_gpu)
    if w <= 0 or h <= 0:
        print(f"  [错误] 裁剪尺寸无效: w={w}, h={h}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_gpu:
        codec_args = ["-c:v", "h264_nvenc", "-cq", str(crf), "-preset", "p4"]
    else:
        codec_args = ["-c:v", "libx264", "-crf", str(crf)]

    cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-vf", f"crop={w}:{h}:{x1}:{y1}",
        *codec_args,
        "-c:a", "copy",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            print(f"  [错误] ffmpeg 失败 (code={result.returncode})")
            stderr_tail = result.stderr[-500:] if result.stderr else ""
            if stderr_tail:
                print(f"  {stderr_tail}")
            return False
        return True
    except FileNotFoundError:
        print("  [错误] 未找到 ffmpeg，请确认已安装并在 PATH 中")
        return False
    except subprocess.TimeoutExpired:
        print("  [错误] ffmpeg 超时 (>3600s)")
        return False


def main():
    args = parse_args()

    ffmpeg = get_ffmpeg_exe()
    if not ffmpeg:
        print("错误: 未找到 ffmpeg。请安装: pip install imageio-ffmpeg")
        return
    print(f"ffmpeg:   {ffmpeg}")

    use_gpu = args.gpu
    if use_gpu:
        if check_nvenc(ffmpeg):
            print("编码器:   h264_nvenc (GPU)")
        else:
            print("警告: h264_nvenc 不可用，回退到 CPU (libx264)")
            use_gpu = False
    if not use_gpu:
        print("编码器:   libx264 (CPU)")

    print(f"视频目录: {args.video_root.resolve()}")
    print(f"标注目录: {args.label_root.resolve()}")
    print(f"输出目录: {args.output_dir.resolve()}")
    print(f"CRF/CQ:   {args.crf}")
    print()

    print("正在扫描标注文件...")
    labels = find_labels(args.label_root)
    print(f"找到 {len(labels)} 个有效标注\n")

    print("正在扫描视频文件...")
    videos = find_videos(args.video_root)
    print(f"找到 {len(videos)} 个视频文件\n")

    stats = {"success": 0, "skipped": 0, "failed": 0}

    for i, video_path in enumerate(videos, 1):
        video_stem = video_path.stem
        if video_stem not in labels:
            print(f"[{i}/{len(videos)}] [跳过] {video_stem}: 无标注")
            stats["skipped"] += 1
            continue

        rect = labels[video_stem]
        output_path = args.output_dir / f"{video_stem}.mp4"

        if output_path.exists():
            print(f"[{i}/{len(videos)}] [跳过] {video_stem}: 输出已存在")
            stats["skipped"] += 1
            continue

        cx, cy, cw, ch = calc_crop(rect, 1920, 1080, use_gpu)
        print(f"[{i}/{len(videos)}] [处理] {video_stem}  crop={cw}x{ch}+{cx}+{cy}")

        ok = crop_video_ffmpeg(ffmpeg, video_path, rect, output_path, args.crf, use_gpu)
        if ok:
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"  -> {output_path.name} ({size_mb:.1f} MB)")
            stats["success"] += 1
        else:
            stats["failed"] += 1

    print("\n" + "=" * 50)
    print("处理完成!")
    print(f"  成功: {stats['success']} 个视频")
    print(f"  跳过: {stats['skipped']} 个视频")
    print(f"  失败: {stats['failed']} 个视频")


if __name__ == "__main__":
    main()
