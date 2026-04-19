"""
对 orgin-video 里的原始视频进行推理并实时显示：
1. 读取同名 JSON 获取 Lamp 裁剪区域
2. 每帧裁出该区域，插值放大到宽度 640
3. 用 best.pt 推理，得到信号灯分类
4. 在原始视频上画 Lamp 框，视频最下方显示双色圆指示灯状态

标签字符颜色规则：
  R → 红色圆  G → 绿色圆  Y → 黄色圆  0 → 黑色圆

操作说明：
  空格  —— 暂停 / 继续
  N     —— 下一个视频
  Q/ESC —— 退出
"""

import json
import sqlite3
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ───── 配置 ─────────────────────────────────────────────────
MODEL_PATH   = Path("best.pt")
VIDEO_DIR    = Path("orgin-video")
TARGET_WIDTH = 640          # 裁剪区域插值放大到的宽度
CONF_THRESH  = 0.25
IOU_THRESH   = 0.45
SPEED_MULT   = 10            # 播放倍速（跳帧数），5 = 5倍速
INFER_EVERY  = 1            # 每显示 N 帧推理一次（在跳帧基础上再稀疏推理）
MAX_SPEED    = True         # True = 以最大处理速度显示，不限帧率；False = 按倍速限速
SHOW_VIDEO   = False        # True = 每帧都显示；False = 限速 PREVIEW_FPS 显示
PREVIEW_FPS  = 10          # SHOW_VIDEO=False 时的预览帧率（帧/秒）
USE_HALF     = True         # FP16 半精度推理（需要 NVIDIA GPU），速度约提升 1.5-2x
DB_BATCH     = 4096           # 积攒多少条推理结果后批量写入数据库（减少 I/O 阻塞）
DB_PATH      = Path("infer_results.db")   # SQLite 数据库路径
INFER_BATCH  = 32           # GPU 批量推理大小（>1 启用批推理，每积攒 N 张 crop 统一推理；1=逐帧推理）
# ─────────────────────────────────────────────────────────────

# 单个字符 → BGR 颜色
CHAR_COLOR = {
    "R": (0,   0,   220),    # 红
    "G": (0,   200,  0),     # 绿
    "Y": (0,   210, 220),    # 黄
    "0": (30,  30,  30),     # 黑
}
DEFAULT_CHAR_COLOR = (160, 160, 160)

# Lamp 框颜色
LAMP_BOX_COLOR = (255, 180, 0)   # 蓝橙色


def init_db(db_path: Path) -> sqlite3.Connection:
    """初始化 SQLite 数据库，创建推理结果表（已存在则跳过）"""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")   # 提升并发写入性能
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            video       TEXT    NOT NULL,   -- 视频文件名（不含扩展名）
            frame_idx   INTEGER NOT NULL,   -- 视频原始帧号
            time_s      REAL    NOT NULL,   -- 相对视频起点的时间（秒），起点=0
            region_idx  INTEGER NOT NULL,   -- Lamp 区域编号（0起）
            label       TEXT,               -- 分类标签，如 RR / G0，无检测则 NULL
            confidence  REAL,               -- 置信度，无检测则 NULL
            box_x1      REAL,               -- 在放大图上的检测框坐标（无检测则 NULL）
            box_y1      REAL,
            box_x2      REAL,
            box_y2      REAL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_video_time
        ON detections (video, time_s)
    """)
    conn.commit()
    return conn


def collect_rows(video_stem: str, frame_idx: int, time_s: float,
                 all_detections: list) -> list:
    """将当前帧检测结果转为 DB 行列表（不立即写入）"""
    rows = []
    for region_idx, dets in enumerate(all_detections):
        if dets:
            for label, conf, bx1, by1, bx2, by2 in dets:
                rows.append((video_stem, frame_idx, time_s,
                              region_idx, label, conf,
                              bx1, by1, bx2, by2))
        else:
            rows.append((video_stem, frame_idx, time_s,
                          region_idx, None, None,
                          None, None, None, None))
    return rows


def flush_db(conn: sqlite3.Connection, pending: list):
    """将积攒的行批量写入数据库并清空 pending"""
    if not pending:
        return
    conn.executemany("""
        INSERT INTO detections
            (video, frame_idx, time_s, region_idx,
             label, confidence, box_x1, box_y1, box_x2, box_y2)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, pending)
    conn.commit()
    pending.clear()


def run_batch_infer(batch_buf: list, model, class_names: dict,
                    regions: list, video_stem: str,
                    db_pending: list) -> dict:
    """
    对 batch_buf 中积攒的 crops 执行一次批量推理。

    batch_buf 每项格式：(crop_img_or_None, frame_idx, time_s, region_idx)
      - crop_img_or_None 为 None 表示该区域裁剪为空，跳过推理直接记空检测。

    返回 {frame_idx: [dets_for_region0, dets_for_region1, ...]}
    并将结果追加到 db_pending（不立即写库）。
    """
    if not batch_buf:
        return {}

    # 收集有效 crop 及其在 batch_buf 中的下标
    crops, crop_indices = [], []
    for i, (img, _fi, _ts, _ri) in enumerate(batch_buf):
        if img is not None:
            crops.append(img)
            crop_indices.append(i)

    # 批量推理
    infer_map: dict[int, list] = {}   # buf_index -> dets
    if crops:
        res_list = model.predict(
            source=crops,
            imgsz=TARGET_WIDTH,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            half=USE_HALF,
            agnostic_nms=True,
            verbose=False,
        )
        for buf_i, res in zip(crop_indices, res_list):
            dets = []
            boxes = res.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    dets.append((class_names[cls_id], conf, bx1, by1, bx2, by2))
            infer_map[buf_i] = dets

    # 按 frame_idx 分组
    frame_data: dict[int, dict] = {}
    for i, (_img, fi, ts, ri) in enumerate(batch_buf):
        fd = frame_data.setdefault(fi, {"time_s": ts, "regions": {}})
        fd["regions"][ri] = infer_map.get(i, [])

    # 整理成 [dets_per_region, ...] 并写入 db_pending
    frame_dets: dict[int, list] = {}
    for fi, fd in frame_data.items():
        all_dets = [fd["regions"].get(ri, []) for ri in range(len(regions))]
        frame_dets[fi] = all_dets
        db_pending.extend(collect_rows(video_stem, fi, fd["time_s"], all_dets))

    return frame_dets


def load_crop_regions(json_path: Path):
    """从 labelme JSON 读取所有 Lamp 矩形区域，返回 [(x1,y1,x2,y2), ...]"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    regions = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        pts = shape["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        regions.append((x1, y1, x2, y2))
    return regions


def draw_indicator_bar(frame, all_detections):
    """
    在视频最下方绘制半透明黑色状态栏，
    每组检测结果用两个彩色圆表示标签的两个字符。
    多个区域的结果从左到右排列，圆之间留间距。
    """
    h, w = frame.shape[:2]
    bar_h  = 80       # 状态栏高度（像素）
    radius = 24       # 圆半径
    gap    = 20       # 两圆之间的间距
    margin = 30       # 左边起始边距
    group_gap = 50    # 多区域之间的间距

    # 半透明黑色背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cy = h - bar_h // 2   # 圆心纵坐标（栏中央）
    cx = margin + radius

    for region_dets in all_detections:
        if not region_dets:
            # 无检测：画两个灰色空心圆
            for _ in range(2):
                cv2.circle(frame, (cx, cy), radius, (100, 100, 100), 2)
                cx += radius * 2 + gap
        else:
            # 取置信度最高的检测结果
            best = max(region_dets, key=lambda d: d[1])
            label = best[0]   # e.g. "RR", "G0"

            for char in label[:2]:
                color = CHAR_COLOR.get(char, DEFAULT_CHAR_COLOR)
                cv2.circle(frame, (cx, cy), radius, color, -1)
                # 白色边框增加辨识度
                cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)
                # 圆内显示字符
                cv2.putText(frame, char,
                            (cx - 8, cy + 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cx += radius * 2 + gap

            # 在圆右侧显示标签和置信度
            conf_text = f"{label} {best[1]:.2f}"
            cv2.putText(frame, conf_text,
                        (cx, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (220, 220, 220), 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(
                conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cx += tw + group_gap


def draw_lamp_box(frame, region, detections):
    """在原始帧上标出 Lamp 区域框（不再在框内写文字）"""
    rx1, ry1, rx2, ry2 = region
    rw = rx2 - rx1
    rh = ry2 - ry1

    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), LAMP_BOX_COLOR, 2)

    if detections:
        scale_x = rw / TARGET_WIDTH
        scale_y = rh / (rh * TARGET_WIDTH / rw)

        best = max(detections, key=lambda d: d[1])
        label, conf, bx1, by1, bx2, by2 = best

        ox1 = rx1 + int(bx1 * scale_x)
        oy1 = ry1 + int(by1 * scale_y)
        ox2 = rx1 + int(bx2 * scale_x)
        oy2 = ry1 + int(by2 * scale_y)

        # 用标签第一个字符的颜色画检测框
        box_color = CHAR_COLOR.get(label[0], DEFAULT_CHAR_COLOR) if label else DEFAULT_CHAR_COLOR
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), box_color, 2)

    return frame


# ── 主程序 ───────────────────────────────────────────────────

model = YOLO(str(MODEL_PATH))
class_names = model.names
print(f"已加载模型: {MODEL_PATH}  类别: {class_names}")

db_conn = init_db(DB_PATH)
print(f"数据库: {DB_PATH.resolve()}\n")
print("操作说明: [空格] 暂停/继续  [N] 下一个视频  [Q/ESC] 退出\n")

video_files = sorted(set(VIDEO_DIR.glob("*.MP4")) | set(VIDEO_DIR.glob("*.mp4")))
if not video_files:
    print(f"未找到视频文件：{VIDEO_DIR.resolve()}")
    raise SystemExit

WINDOW = "原始视频推理 (空格暂停 | N下一个 | Q退出)"
if SHOW_VIDEO:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
else:
    print("SHOW_VIDEO=False，不显示视频窗口，以最大速度处理。")

for video_path in video_files:
    # 匹配同目录下的 JSON（文件名前缀相同即可）
    json_candidates = sorted(VIDEO_DIR.glob(f"{video_path.stem}_*.json"))
    if not json_candidates:
        print(f"[跳过] 找不到对应 JSON：{video_path.name}")
        continue
    json_path = json_candidates[0]

    regions = load_crop_regions(json_path)
    if not regions:
        print(f"[跳过] JSON 中没有有效区域：{json_path.name}")
        continue

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[跳过] 无法打开视频：{video_path.name}")
        continue

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # MAX_SPEED=True 时 waitKey 只等 1ms，让推理跑多快显多快
    # MAX_SPEED=False 时按原始帧率 × 倍速控制显示节奏
    delay_ms = 1 if MAX_SPEED else max(1, int(1000 / fps / SPEED_MULT))

    print(f"视频: {video_path.name}  JSON: {json_path.name}")
    print(f"  帧率: {fps:.1f}fps  总帧: {total}  倍速: {SPEED_MULT}x  "
          f"跳帧: {SPEED_MULT}帧/步  推理间隔: 每{INFER_EVERY}显示帧  区域数: {len(regions)}")
    for i, r in enumerate(regions):
        print(f"  区域{i}: {r}")

    paused          = False
    skip            = False
    quit_all        = False
    display_idx     = 0          # 实际显示的帧计数（用于控制推理频率）
    last_detections = [[] for _ in regions]
    last_frame_idx  = 0          # 上次推理时的原始帧号（用于数据库记录）
    db_pending      = []         # 待写入数据库的行缓冲
    batch_buf       = []         # 批量推理缓冲：[(crop_or_None, frame_idx, time_s, region_idx), ...]

    # FPS 计量：用滑动窗口平均，避免数值剧烈跳动
    fps_window      = 30         # 滑动窗口大小（帧数）
    ts_history      = []         # 存放最近 fps_window 帧的时间戳
    display_fps     = 0.0        # 当前显示的 FPS 值

    # 终端进度输出
    PRINT_EVERY     = 50         # 每处理 N 个显示帧打印一次进度
    video_start_ts  = time.perf_counter()   # 视频开始处理的时刻

    # SHOW_VIDEO=True 时按 PREVIEW_FPS 节流显示
    preview_interval = 1.0 / PREVIEW_FPS
    last_preview_ts  = -preview_interval   # 初始设为负数，确保第一帧就显示

    while True:
        if not paused:
            # 记录读帧前的帧位置（= 本帧的原始帧号）
            cur_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 读取当前帧
            ret, frame = cap.read()
            if not ret:
                break

            # 跳过后续 (SPEED_MULT-1) 帧：grab 只解封装不解码，速度快
            for _ in range(SPEED_MULT - 1):
                cap.grab()

            # 每 INFER_EVERY 个显示帧才做一次推理
            if display_idx % INFER_EVERY == 0:
                cur_time_s = cur_frame_idx / fps

                if INFER_BATCH > 1:
                    # ── 批量模式：积攒 crops，满 INFER_BATCH 条时统一推理 ──
                    for r_idx, region in enumerate(regions):
                        rx1, ry1, rx2, ry2 = region
                        crop = frame[ry1:ry2, rx1:rx2]
                        if crop.size == 0:
                            batch_buf.append((None, cur_frame_idx, cur_time_s, r_idx))
                            continue
                        rh_orig = ry2 - ry1
                        rw_orig = rx2 - rx1
                        new_h = round(rh_orig * TARGET_WIDTH / rw_orig)
                        upscaled = cv2.resize(crop, (TARGET_WIDTH, new_h),
                                              interpolation=cv2.INTER_LINEAR)
                        batch_buf.append((upscaled, cur_frame_idx, cur_time_s, r_idx))

                    if len(batch_buf) >= INFER_BATCH:
                        frame_dets = run_batch_infer(
                            batch_buf, model, class_names, regions,
                            video_path.stem, db_pending)
                        batch_buf.clear()
                        if frame_dets:
                            last_detections = frame_dets[max(frame_dets)]
                        if len(db_pending) >= DB_BATCH:
                            flush_db(db_conn, db_pending)

                else:
                    # ── 单帧模式（原有逻辑） ──
                    last_detections = []
                    last_frame_idx  = cur_frame_idx
                    for region in regions:
                        rx1, ry1, rx2, ry2 = region
                        crop = frame[ry1:ry2, rx1:rx2]
                        if crop.size == 0:
                            last_detections.append([])
                            continue
                        rh_orig = ry2 - ry1
                        rw_orig = rx2 - rx1
                        new_h = round(rh_orig * TARGET_WIDTH / rw_orig)
                        # INTER_LINEAR 比 INTER_CUBIC 快约 3-5x，推理精度几乎无差异
                        upscaled = cv2.resize(crop, (TARGET_WIDTH, new_h),
                                              interpolation=cv2.INTER_LINEAR)
                        results = model.predict(
                            source=upscaled,
                            imgsz=TARGET_WIDTH,
                            conf=CONF_THRESH,
                            iou=IOU_THRESH,
                            half=USE_HALF,       # FP16 推理（GPU 加速）
                            agnostic_nms=True,   # 类无关 NMS，略减 NMS 耗时
                            verbose=False,
                        )
                        dets = []
                        boxes = results[0].boxes
                        if boxes is not None:
                            for box in boxes:
                                cls_id = int(box.cls[0])
                                conf   = float(box.conf[0])
                                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                                dets.append((class_names[cls_id], conf,
                                             bx1, by1, bx2, by2))
                        last_detections.append(dets)

                    # 积攒到 db_pending，达到 DB_BATCH 条后批量写入
                    db_pending.extend(collect_rows(
                        video_path.stem, last_frame_idx, cur_time_s, last_detections))
                    if len(db_pending) >= DB_BATCH:
                        flush_db(db_conn, db_pending)

            # 更新 FPS（滑动窗口平均）
            now = time.perf_counter()
            ts_history.append(now)
            if len(ts_history) > fps_window:
                ts_history.pop(0)
            if len(ts_history) >= 2:
                display_fps = (len(ts_history) - 1) / (ts_history[-1] - ts_history[0])

            real_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # SHOW_VIDEO=True 且距上次显示超过 preview_interval 时才渲染
            if SHOW_VIDEO and (now - last_preview_ts >= preview_interval):
                display = frame.copy()
                for region, dets in zip(regions, last_detections):
                    draw_lamp_box(display, region, dets)
                draw_indicator_bar(display, last_detections)

                info = (f"{video_path.stem}  frame {real_frame}/{total}"
                        f"  {SPEED_MULT}x  {display_fps:.1f} fps")
                cv2.putText(display, info, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (230, 230, 230), 1, cv2.LINE_AA)

                cv2.imshow(WINDOW, display)
                last_preview_ts = now
            display_idx += 1

            # 终端进度打印（每 PRINT_EVERY 显示帧输出一次，用 \r 原地刷新）
            if display_idx % PRINT_EVERY == 0:
                elapsed   = time.perf_counter() - video_start_ts
                pct       = real_frame / total * 100 if total > 0 else 0
                bar_len   = 30
                filled    = int(bar_len * real_frame / total) if total > 0 else 0
                bar       = "█" * filled + "░" * (bar_len - filled)
                print(
                    f"\r  [{bar}] {pct:5.1f}%  "
                    f"frame {real_frame:>6}/{total}  "
                    f"{display_fps:5.1f} fps  "
                    f"elapsed {elapsed:6.1f}s",
                    end="", flush=True
                )

        if SHOW_VIDEO:
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord(" "):
                paused = not paused
            elif key in (ord("n"), ord("N")):
                skip = True
                break
            elif key in (ord("q"), ord("Q"), 27):
                quit_all = True
                break

    # 批量模式：视频结束时刷新剩余未满一批的 crops
    if INFER_BATCH > 1 and batch_buf:
        frame_dets = run_batch_infer(
            batch_buf, model, class_names, regions,
            video_path.stem, db_pending)
        batch_buf.clear()
        if frame_dets:
            last_detections = frame_dets[max(frame_dets)]
    flush_db(db_conn, db_pending)   # 确保本视频所有结果都写入库

    cap.release()

    elapsed_total = time.perf_counter() - video_start_ts
    avg_fps = display_idx / elapsed_total if elapsed_total > 0 else 0
    # 进度条结束后换行，再打印汇总
    print(
        f"\r  [{'█' * 30}] 100.0%  "
        f"frame {total:>6}/{total}  "
        f"{avg_fps:5.1f} fps  "
        f"elapsed {elapsed_total:6.1f}s"
    )

    if quit_all:
        print("用户退出。")
        break
    print(f"  → {'Skipped' if skip else 'Done'}  "
          f"({display_idx} frames displayed, avg {avg_fps:.1f} fps)\n")

if SHOW_VIDEO:
    cv2.destroyAllWindows()
db_conn.close()
print(f"结束。数据库已保存到: {DB_PATH.resolve()}")
