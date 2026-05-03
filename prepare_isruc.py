import argparse
import os
import glob
import numpy as np
import pyedflib
from scipy.signal import resample_poly
from logger import get_logger

# ISRUC 原始标签 → 标准五分类 (W, N1, N2, N3, REM)
ISRUC_LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/isruc/isruc-1",
                        help="Path to ISRUC SG1 root folder.")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/isruc_npz",
                        help="Directory to save .npz files.")
    parser.add_argument("--select_ch", type=str, default="C3-A2",
                        help="Preferred EEG channel (e.g., C3-A2). Will auto-fallback if not found.")
    parser.add_argument("--target_fs", type=int, default=100,
                        help="Target sampling rate (default 100 Hz).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, "prepare_isruc.log"), level="info")

    # 查找所有子文件夹（受试者目录），不限定名字格式
    subj_dirs = [d for d in glob.glob(os.path.join(args.data_dir, "*"))
                 if os.path.isdir(d)]
    logger.info(f"Found {len(subj_dirs)} subject directories.")

    for subj_dir in subj_dirs:
        subj_id = os.path.basename(subj_dir)          # 直接用文件夹名作为ID
        logger.info(f"Processing subject: {subj_id}")

        # 查找 .rec 文件（取第一个）
        rec_files = glob.glob(os.path.join(subj_dir, "*.rec"))
        if not rec_files:
            logger.warning(f"No .rec file in {subj_dir}, skip.")
            continue
        rec_path = rec_files[0]

        # 查找注释 .txt 文件（取第一个）
        txt_files = glob.glob(os.path.join(subj_dir, "*.txt"))
        if not txt_files:
            logger.warning(f"No annotation .txt file in {subj_dir}, skip.")
            continue
        ann_path = txt_files[0]

        # ---------- 读取 EEG 信号 ----------
        try:
            f = pyedflib.EdfReader(rec_path)
        except Exception as e:
            logger.error(f"Failed to open {rec_path}: {e}")
            continue

        ch_names = f.getSignalLabels()

        # ---------- 智能通道选择（多级备选） ----------
        select_ch_idx = None
        best_ch_name = None

        # 优先级列表：C3+A2 > C3 > C4+A1 > C4
        priority_patterns = [
            ('C3', 'A2'),
            ('C3', None),
            ('C4', 'A1'),
            ('C4', None),
        ]

        for pat1, pat2 in priority_patterns:
            for idx, ch in enumerate(ch_names):
                if pat1 in ch:
                    if pat2 is None or pat2 in ch:
                        select_ch_idx = idx
                        best_ch_name = ch
                        break
            if select_ch_idx is not None:
                break

        if select_ch_idx is None:
            logger.warning(f"No suitable EEG channel found for subject {subj_id}. Available: {ch_names}")
            f.close()
            continue

        logger.info(f"Selected channel for {subj_id}: '{best_ch_name}' (preferred: {args.select_ch})")

        fs = f.getSampleFrequency(select_ch_idx)
        signal = f.readSignal(select_ch_idx).astype(np.float32)
        f.close()

        # ---------- 读取标签 ----------
        try:
            with open(ann_path, 'r') as fh:
                lines = fh.read().strip().split()
            raw_labels = [int(x) for x in lines if x.strip() != '']
        except Exception as e:
            logger.error(f"Failed to read {ann_path}: {e}")
            continue

        # 标签映射（5 → 4）
        labels = np.array([ISRUC_LABEL_MAP.get(l, -1) for l in raw_labels])
        valid_mask = labels >= 0
        labels = labels[valid_mask]
        n_epochs = len(labels)

        # ---------- 信号切片为 30 秒 epochs ----------
        epoch_len_samples = int(30 * fs)
        total_needed = n_epochs * epoch_len_samples
        if len(signal) < total_needed:
            logger.warning(f"Signal shorter than expected for {subj_id}. Truncating labels.")
            n_epochs = len(signal) // epoch_len_samples
            labels = labels[:n_epochs]
            total_needed = n_epochs * epoch_len_samples

        signal = signal[:total_needed]
        x = signal.reshape(n_epochs, epoch_len_samples)

        # ---------- 重采样到 100 Hz ----------
        if fs != args.target_fs:
            # 使用 scipy 多相滤波重采样
            x = resample_poly(x, args.target_fs, fs, axis=1)
            epoch_len_samples = x.shape[1]

        # 增加通道维 (n_epochs, samples, 1)
        x = np.expand_dims(x, axis=2)

        # ---------- 逐受试者 Z‑score 归一化 ----------
        mean = x.mean()
        std = x.std()
        if std > 0:
            x = (x - mean) / std

        # ---------- 保存 .npz ----------
        save_dict = {
            "x": x.astype(np.float32),
            "y": labels.astype(np.int32),
            "fs": args.target_fs,
            "ch_label": best_ch_name,
        }
        npz_path = os.path.join(args.output_dir, f"{subj_id}.npz")
        np.savez(npz_path, **save_dict)
        logger.info(f"Saved {npz_path} -> x{x.shape}, y{labels.shape}")

    logger.info("All subjects processed.")

    # ---------- 生成交叉验证用的 subjects.txt ----------
    npz_files = glob.glob(os.path.join(args.output_dir, "*.npz"))
    subj_ids = sorted([os.path.splitext(os.path.basename(f))[0] for f in npz_files])
    txt_path = os.path.join(args.output_dir, "isruc.txt")
    with open(txt_path, "w") as f:
        for sid in subj_ids:
            f.write(sid + "\n")
    logger.info(f"Generated {txt_path} with {len(subj_ids)} subjects.")

if __name__ == "__main__":
    main()