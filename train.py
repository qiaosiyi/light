"""
使用 yolo26n.pt 预训练模型在 yolo_dataset_big 数据集上进行训练
"""

from pathlib import Path
from ultralytics import YOLO

# ───── 配置区 ─────────────────────────────────────────────
DATA_YAML   = Path("yolo_dataset_big/data.yaml")   # 数据集配置
MODEL       = "yolo26n.pt"                          # 预训练权重
EPOCHS      = 100                                   # 训练轮数
IMG_SIZE    = 640                                   # 输入图片尺寸（与放大后的图片宽度一致）
BATCH       = 16                                    # 批大小，显存不足时改为 8 或 4
PROJECT     = "runs/train"                          # 结果保存的父目录
NAME        = "yolo26n_light_100epochs"                       # 本次训练的子目录名
WORKERS     = 4                                     # 数据加载线程数
DEVICE      = 0                                     # GPU 编号，CPU 则改为 "cpu"
# ──────────────────────────────────────────────────────────


def main():
    # 加载预训练模型
    model = YOLO(MODEL)

    # 开始训练
    results = model.train(
        data=str(DATA_YAML.resolve()),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        workers=WORKERS,
        device=DEVICE,
        # 以下为可选调优参数，按需取消注释
        # lr0=0.01,           # 初始学习率
        # lrf=0.01,           # 最终学习率比例
        # momentum=0.937,
        # weight_decay=0.0005,
        # warmup_epochs=3,
        # patience=50,        # Early stopping 轮数，0 表示禁用
        # amp=True,           # 混合精度训练（节省显存）
        # cache=True,         # 图片预缓存到内存（加快速度，需要较大内存）
        # cos_lr=True,        # 余弦学习率调度
    )

    print("\n训练完成！")
    print(f"最佳权重保存在: {results.save_dir}/weights/best.pt")

    # ── 可选：训练结束后立即在验证集上评估 ──
    metrics = model.val()
    print(f"\n验证结果：")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()