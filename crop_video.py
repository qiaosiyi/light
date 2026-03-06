"""
从原始视频中根据 LabelMe 标注的矩形区域，随机截取帧并裁剪出红绿灯小图片。

用法:
    python crop_video.py [--video_root PATH] [--label_root PATH]
                         [--output_dir PATH] [--frames_per_video N]
"""

import json
import random
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

# ======================== 默认配置 ========================
# Windows 测试（视频平放在 orgin_video/ 下）
DEFAULT_VIDEO_ROOT = Path("orgin-video")
# Linux 正式运行时改为:
# DEFAULT_VIDEO_ROOT = Path("/media/zekai/Expansion/Experiment data CHAO_MAI")

DEFAULT_LABEL_ROOT = Path("Label_prepare")
DEFAULT_OUTPUT_DIR = Path("cropped_data")
DEFAULT_FRAMES_PER_VIDEO = 50

VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"}


def parse_args():
    parser = argparse.ArgumentParser(description="从视频中裁剪红绿灯区域")
    parser.add_argument("--video_root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--label_root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames_per_video", type=int, default=DEFAULT_FRAMES_PER_VIDEO)
    return parser.parse_args()


def find_labels(label_root: Path) -> dict:
    """遍历 Label_prepare 下所有 JSON，返回 {video_stem: (json_path, rect)} 字典。"""
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

            # DJI_0297_t300.000s.json -> DJI_0297
            stem = json_path.stem
            video_stem = stem.rsplit("_t", 1)[0]
            labels[video_stem] = {"json_path": json_path, "rect": rect}
            break  # 每个 JSON 只取第一个 Lamp 矩形

    return labels


def find_videos(video_root: Path) -> list:
    """递归查找所有视频文件。"""
    videos = []
    for p in video_root.rglob("*"):
        if p.is_file() and p.suffix in VIDEO_EXTS:
            videos.append(p)
    return videos


def extract_crops(video_path: Path, rect: tuple, output_dir: Path, n_frames: int):
    """从视频随机采 n_frames 帧，裁剪 rect 区域并保存。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [错误] 无法打开视频: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"  [错误] 无法获取帧数: {video_path}")
        cap.release()
        return 0

    n = min(n_frames, total_frames)
    frame_indices = sorted(random.sample(range(total_frames), n))

    x1, y1, x2, y2 = rect
    video_stem = video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx in tqdm(frame_indices, desc=f"  {video_stem}", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(w, x2)
        cy2 = min(h, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = frame[cy1:cy2, cx1:cx2]
        out_path = output_dir / f"{video_stem}_frame{idx:06d}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    return saved


def main():
    args = parse_args()
    print(f"视频目录:   {args.video_root.resolve()}")
    print(f"标注目录:   {args.label_root.resolve()}")
    print(f"输出目录:   {args.output_dir.resolve()}")
    print(f"每视频帧数: {args.frames_per_video}")
    print()

    # 1. 解析标注
    print("正在扫描标注文件...")
    labels = find_labels(args.label_root)
    print(f"找到 {len(labels)} 个有效标注\n")

    # 2. 查找视频
    print("正在扫描视频文件...")
    videos = find_videos(args.video_root)
    print(f"找到 {len(videos)} 个视频文件\n")

    # 3. 逐视频裁剪
    stats = {"success": 0, "skipped": 0, "failed": 0, "total_crops": 0}

    for video_path in videos:
        video_stem = video_path.stem
        if video_stem not in labels:
            print(f"[跳过] {video_stem}: 未找到对应标注")
            stats["skipped"] += 1
            continue

        rect = labels[video_stem]["rect"]
        out_dir = args.output_dir / video_stem
        print(f"[处理] {video_stem}  裁剪区域={rect}")

        saved = extract_crops(video_path, rect, out_dir, args.frames_per_video)
        if saved > 0:
            stats["success"] += 1
            stats["total_crops"] += saved
            print(f"  -> 保存 {saved} 张图片到 {out_dir}")
        else:
            stats["failed"] += 1
            print(f"  -> 失败，未保存任何图片")

    # 4. 汇总
    print("\n" + "=" * 50)
    print("处理完成!")
    print(f"  成功: {stats['success']} 个视频")
    print(f"  跳过: {stats['skipped']} 个视频 (无标注)")
    print(f"  失败: {stats['failed']} 个视频")
    print(f"  共生成 {stats['total_crops']} 张裁剪图片")


if __name__ == "__main__":
    main()
