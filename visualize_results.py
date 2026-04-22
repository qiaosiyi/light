"""
Read infer_results.db and visualize traffic-light detection timelines.
  X axis : time (seconds)
  Y axis : fixed height bars, color = label (R red / G green / None grey)
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
LABEL_ORDER = ["R", "G", "None"]

LABEL_COLOR = {
    "R":    "#E53935",   # red
    "G":    "#43A047",   # green
    "None": "#BDBDBD",   # grey (no detection)
}
LABEL_DESC = {
    "R":    "R  Red",
    "G":    "G  Green",
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
        # ── 左侧面板（视频选择，双列 + 可滚动）────────────────
        left_outer = tk.Frame(self.root, bg="#181825", width=300)
        left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        left_outer.pack_propagate(False)

        tk.Label(left_outer, text="Select Video", bg="#181825", fg="#CDD6F4",
                 font=("Segoe UI", 12, "bold")).pack(pady=(16, 6))

        # 可滚动区域
        scroll_frame = tk.Frame(left_outer, bg="#181825")
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=4)

        canvas_scroll = tk.Canvas(scroll_frame, bg="#181825",
                                   highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(scroll_frame, orient=tk.VERTICAL,
                                   command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas_scroll, bg="#181825")
        inner_win = canvas_scroll.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(event):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))

        def _on_canvas_configure(event):
            canvas_scroll.itemconfig(inner_win, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas_scroll.bind("<Configure>", _on_canvas_configure)

        # 鼠标滚轮支持
        def _on_mousewheel(event):
            canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)

        # 双列布局放按钮
        self.btn_refs = []
        for i, v in enumerate(self.videos):
            col = i % 2
            row = i // 2
            btn = tk.Button(
                inner, text=v, bg="#313244", fg="#CDD6F4",
                activebackground="#45475A", activeforeground="#CDD6F4",
                relief=tk.FLAT, font=("Segoe UI", 9),
                padx=4, pady=5, cursor="hand2", wraplength=120,
                command=lambda name=v: self.on_video_select(name),
            )
            btn.grid(row=row, column=col, padx=4, pady=3, sticky="ew")
            self.btn_refs.append((v, btn))

        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)

        # ── 统计信息区 ──────────────────────────────────────
        self.stat_label = tk.Label(
            left_outer, text="", bg="#181825", fg="#A6ADC8",
            font=("Segoe UI", 9), justify=tk.LEFT, wraplength=280)
        self.stat_label.pack(pady=8, padx=10, anchor="w")

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
        self.ax.title.set_color("#CDD6F4")

        if not rows:
            self.ax.text(0.5, 0.5, "No data available", transform=self.ax.transAxes,
                         ha="center", va="center", color="#CDD6F4", fontsize=14)
            self.canvas.draw()
            return

        times  = np.array([r[0] for r in rows])
        labels = [r[1] for r in rows]

        # 柱宽 = 相邻帧时间间隔的中位数（避免过宽或过窄）
        if len(times) > 1:
            bar_w = float(np.median(np.diff(np.sort(np.unique(times)))))
            bar_w = max(bar_w, 0.01)
        else:
            bar_w = 0.1

        colors = [LABEL_COLOR.get(lb, LABEL_COLOR["None"]) for lb in labels]

        # 所有柱子高度固定为 1，用颜色区分标签
        self.ax.bar(times, height=1.0, width=bar_w,
                    color=colors, align="center",
                    linewidth=0, zorder=3)

        # 隐藏 Y 轴（高度无实际含义）
        self.ax.set_yticks([])
        self.ax.set_ylim(0, 1.2)
        for spine in ["left", "right", "top"]:
            self.ax.spines[spine].set_visible(False)

        self.ax.set_xlabel("Time (s)", color="#CDD6F4", fontsize=11)
        self.ax.set_title(f"{video}  —  Detection Timeline",
                          color="#CDD6F4", fontsize=13, fontweight="bold", pad=10)

        # 图例：只显示出现过的标签
        present_labels = [lb for lb in LABEL_ORDER if lb in set(labels)]
        legend_handles = [
            mpatches.Patch(color=LABEL_COLOR[lb], label=LABEL_DESC.get(lb, lb))
            for lb in present_labels
        ]
        self.ax.legend(
            handles=legend_handles,
            loc="upper right", framealpha=0.4,
            facecolor="#313244", edgecolor="#45475A",
            labelcolor="#CDD6F4", fontsize=10,
        )

        self.fig.tight_layout()
        self.canvas.draw()

        # Stats panel
        total = len(rows)
        label_counts: dict = {}
        for _, lb, _ in rows:
            label_counts[lb] = label_counts.get(lb, 0) + 1
        stats = f"Total: {total}\n\n"
        for lb in LABEL_ORDER:
            if lb in label_counts:
                pct = label_counts[lb] / total * 100
                stats += f"{LABEL_DESC.get(lb, lb)}: {label_counts[lb]}  ({pct:.1f}%)\n"
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
