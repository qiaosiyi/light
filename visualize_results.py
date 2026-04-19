"""
Read infer_results.db and visualize traffic-light detection timelines.
  X axis : time (seconds)
  Y axis : classification label
  Colors : RR = red, others = various greens
  Left panel buttons to switch between videos
"""

import sqlite3
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

# ───── 配置 ─────────────────────────────────────────────────
DB_PATH     = Path("infer_results.db")
WINDOW_TITLE = "Traffic Light Detection Timeline"
# ─────────────────────────────────────────────────────────────

# Label display order on the Y axis (top → bottom)
LABEL_ORDER = ["RR", "GR", "GY", "G0", "Y0", "None"]

LABEL_COLOR = {
    "RR":   "#E53935",   # red
    "GR":   "#2E7D32",   # dark green
    "GY":   "#43A047",   # medium green
    "G0":   "#66BB6A",   # light green
    "Y0":   "#9CCC65",   # yellow-green
    "None": "#BDBDBD",   # grey (no detection)
}
LABEL_DESC = {
    "RR":   "RR  Red",
    "GR":   "GR  Green-Red",
    "GY":   "GY  Green",
    "G0":   "G0  Go-Green",
    "Y0":   "Y0  Yellow",
    "None": "No detection",
}


def load_videos(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    videos = [r[0] for r in conn.execute(
        "SELECT DISTINCT video FROM detections ORDER BY video").fetchall()]
    conn.close()
    return videos


def load_data(db_path: Path, video: str):
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("""
        SELECT time_s, COALESCE(label, 'None'), MAX(confidence)
        FROM detections
        WHERE video = ?
        GROUP BY time_s, region_idx
        ORDER BY time_s
    """, (video,)).fetchall()
    conn.close()
    return rows   # [(time_s, label, conf), ...]


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1280x700")
        self.root.configure(bg="#1E1E2E")

        self.db_path = DB_PATH
        self.videos  = load_videos(self.db_path)
        self.current_video = tk.StringVar(value=self.videos[0] if self.videos else "")

        self._build_ui()
        if self.videos:
            self.plot(self.videos[0])

    def _build_ui(self):
        # ── 左侧面板（视频选择）──────────────────────────────
        left = tk.Frame(self.root, bg="#181825", width=200)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        left.pack_propagate(False)

        tk.Label(left, text="Select Video", bg="#181825", fg="#CDD6F4",
                 font=("Segoe UI", 12, "bold")).pack(pady=(16, 8))

        self.btn_refs = []
        for v in self.videos:
            btn = tk.Button(
                left, text=v, bg="#313244", fg="#CDD6F4",
                activebackground="#45475A", activeforeground="#CDD6F4",
                relief=tk.FLAT, font=("Segoe UI", 10),
                padx=8, pady=6, cursor="hand2",
                command=lambda name=v: self.on_video_select(name),
            )
            btn.pack(fill=tk.X, padx=10, pady=3)
            self.btn_refs.append((v, btn))

        # ── 统计信息区 ──────────────────────────────────────
        self.stat_label = tk.Label(
            left, text="", bg="#181825", fg="#A6ADC8",
            font=("Segoe UI", 9), justify=tk.LEFT, wraplength=180)
        self.stat_label.pack(pady=12, padx=10, anchor="w")

        # ── 右侧图表区 ──────────────────────────────────────
        right = tk.Frame(self.root, bg="#1E1E2E")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(10, 5.5))
        self.fig.patch.set_facecolor("#1E1E2E")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(right, bg="#181825")
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.config(bg="#181825")
        toolbar.update()

    def on_video_select(self, name: str):
        self.current_video.set(name)
        # 高亮选中按钮
        for v, btn in self.btn_refs:
            btn.configure(bg="#89B4FA" if v == name else "#313244",
                          fg="#1E1E2E" if v == name else "#CDD6F4")
        self.plot(name)

    def plot(self, video: str):
        rows = load_data(self.db_path, video)

        self.ax.cla()
        self.ax.set_facecolor("#181825")
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#45475A")
        self.ax.tick_params(colors="#CDD6F4")
        self.ax.xaxis.label.set_color("#CDD6F4")
        self.ax.yaxis.label.set_color("#CDD6F4")
        self.ax.title.set_color("#CDD6F4")

        if not rows:
            self.ax.text(0.5, 0.5, "No data available", transform=self.ax.transAxes,
                         ha="center", va="center", color="#CDD6F4", fontsize=14)
            self.canvas.draw()
            return

        # 按标签分组绘制散点
        from collections import defaultdict
        groups = defaultdict(lambda: {"t": [], "c": []})
        for time_s, label, conf in rows:
            groups[label]["t"].append(time_s)
            groups[label]["c"].append(conf if conf is not None else 0.0)

        label_to_y = {lb: i for i, lb in enumerate(LABEL_ORDER)}

        plotted_labels = []
        for label in LABEL_ORDER:
            if label not in groups:
                continue
            ts   = np.array(groups[label]["t"])
            conf = np.array(groups[label]["c"])
            y    = label_to_y[label]
            color = LABEL_COLOR[label]

            # 散点：大小 = 置信度，alpha 固定便于密集区域辨认
            sizes = np.clip(conf * 60, 10, 60)
            self.ax.scatter(ts, [y] * len(ts),
                            c=color, s=sizes, alpha=0.7,
                            linewidths=0, zorder=3, label=label)
            plotted_labels.append(label)

        # Y axis ticks
        present = [lb for lb in LABEL_ORDER if lb in groups]
        self.ax.set_yticks([label_to_y[lb] for lb in present])
        self.ax.set_yticklabels(
            [LABEL_DESC.get(lb, lb) for lb in present],
            color="#CDD6F4", fontsize=10)

        self.ax.yaxis.grid(True, color="#313244", linewidth=0.8, zorder=0)
        self.ax.set_axisbelow(True)

        self.ax.set_xlabel("Time (s)", color="#CDD6F4", fontsize=11)
        self.ax.set_title(f"{video}  —  Detection Timeline", color="#CDD6F4",
                          fontsize=13, fontweight="bold", pad=10)

        # Legend
        legend_handles = [
            mpatches.Patch(color=LABEL_COLOR[lb],
                           label=LABEL_DESC.get(lb, lb))
            for lb in plotted_labels
        ]
        self.ax.legend(
            handles=legend_handles,
            loc="upper right", framealpha=0.3,
            facecolor="#313244", edgecolor="#45475A",
            labelcolor="#CDD6F4", fontsize=9,
        )

        self.fig.tight_layout()
        self.canvas.draw()

        # Stats panel
        total = len(rows)
        label_counts = {}
        for _, label, _ in rows:
            label_counts[label] = label_counts.get(label, 0) + 1
        stats = f"Total: {total}\n\n"
        for lb in LABEL_ORDER:
            if lb in label_counts:
                stats += f"{LABEL_DESC.get(lb, lb)}: {label_counts[lb]}\n"
        self.stat_label.configure(text=stats)

        # 高亮当前按钮
        for v, btn in self.btn_refs:
            btn.configure(bg="#89B4FA" if v == video else "#313244",
                          fg="#1E1E2E" if v == video else "#CDD6F4")


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH.resolve()}")
        print("Please run infer_origin.py first to generate the data.")
        return

    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
