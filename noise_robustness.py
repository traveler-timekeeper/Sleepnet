import sys, os, glob
import numpy as np
import torch
import argparse
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
import importlib

sys.path.insert(0, os.path.dirname(__file__))
from model import Model
from data import load_data, get_subject_files
from minibatching import iterate_batch_multiple_seq_minibatches
from utils import load_seq_ids
from logger import get_logger

def add_gaussian_noise(signal, snr_db):
    """向信号添加高斯白噪声，返回带噪信号。
       signal: numpy 数组，形状 (3000,)
       snr_db: 信噪比(dB)，若为 None 则不加噪声。
    """
    if snr_db is None or snr_db == np.inf:
        return signal.copy()
    signal_power = np.mean(signal ** 2)
    # 避免除零
    if signal_power == 0:
        return signal.copy()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def evaluate_with_noise(config, model, test_x, test_y, snr_db, device):
    trues, preds = [], []
    for night_x, night_y in zip(test_x, test_y):
        # 对整夜数据逐 epoch 添加噪声
        noisy_x = np.zeros_like(night_x)
        for i in range(len(night_x)):
            # night_x[i] 形状 (3000, 1, 1) 或 (3000,)
            epoch = night_x[i].squeeze()
            noisy_epoch = add_gaussian_noise(epoch, snr_db)
            noisy_x[i] = noisy_epoch.reshape(night_x[i].shape)
        minibatch_fn = iterate_batch_multiple_seq_minibatches(
            [noisy_x], [night_y],
            batch_size=config["batch_size"],
            seq_length=config["seq_length"],
            shuffle_idx=None, augment_seq=False)
        outs = model.evaluate_with_dataloader(minibatch_fn)
        trues.extend(outs["test/trues"])
        preds.extend(outs["test/preds"])
    acc = skmetrics.accuracy_score(trues, preds) * 100
    mf1 = skmetrics.f1_score(trues, preds, average='macro') * 100
    return acc, mf1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--model_dir", required=True, help="如 output/bilstm_attn_tune_fold4")
    parser.add_argument("--fold_idx", type=int, required=True)
    parser.add_argument("--output_dir", default="./output/noise_robustness")
    parser.add_argument("--gpu", type=int, default=-1, help="-1 表示使用 CPU")
    args = parser.parse_args()

    # 加载配置
    spec = importlib.util.spec_from_file_location("*", args.config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, "noise_robustness.log"), level="info")

    # 设备
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # 获取测试集
    subject_files = glob.glob(os.path.join(config["data_dir"], "*.npz"))
    seq_sids = load_seq_ids(f"{config['dataset']}.txt")
    fold_pids = np.array_split(seq_sids, config["n_folds"])
    test_sids = fold_pids[args.fold_idx]
    logger.info(f"Test SIDs: {test_sids}")

    test_files = []
    for sid in test_sids:
        test_files.append(get_subject_files(config["dataset"], subject_files, sid))
    test_files = np.hstack(test_files)
    test_x, test_y, _ = load_data(test_files)

    # 加载最佳模型（与 eval_single_fold 相同的安全加载方式）
    config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)
    model = Model(config=config, output_dir=os.path.join(args.model_dir, str(args.fold_idx)),
                  use_rnn=True, testing=True, use_best=False, device=device)
    best_ckpt = os.path.join(args.model_dir, str(args.fold_idx), "best_ckpt", "best_model.ckpt")
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.tsn.load_state_dict(checkpoint)
    model.tsn.to(device)
    logger.info(f"Loaded model from {best_ckpt}")

    # 测试不同信噪比
    snr_levels = [None, 20, 10, 5, 0, -5]   # None 表示无噪声（原始）
    results = {"SNR": [], "Accuracy": [], "Macro_F1": []}

    for snr in snr_levels:
        label = "Original" if snr is None else f"{snr} dB"
        logger.info(f"Evaluating with SNR = {label} ...")
        acc, mf1 = evaluate_with_noise(config, model, test_x, test_y, snr, device)
        logger.info(f"  Accuracy = {acc:.2f}%, Macro F1 = {mf1:.2f}%")
        results["SNR"].append(label)
        results["Accuracy"].append(acc)
        results["Macro_F1"].append(mf1)

    # 保存结果为 CSV
    csv_file = os.path.join(args.output_dir, "noise_results.csv")
    with open(csv_file, "w") as f:
        f.write("SNR,Accuracy,Macro_F1\n")
        for snr, acc, mf1 in zip(results["SNR"], results["Accuracy"], results["Macro_F1"]):
            f.write(f"{snr},{acc:.2f},{mf1:.2f}\n")
    logger.info(f"Results saved to {csv_file}")

    # 绘制曲线
    x_labels = results["SNR"]
    acc_vals = results["Accuracy"]
    mf1_vals = results["Macro_F1"]
    plt.figure(figsize=(8, 5))
    plt.plot(x_labels, acc_vals, marker='o', label='Accuracy')
    plt.plot(x_labels, mf1_vals, marker='s', label='Macro F1')
    plt.xlabel("Noise Level (SNR)")
    plt.ylabel("Score (%)")
    plt.title("Robustness to Gaussian Noise")
    plt.legend()
    plt.grid(True)
    plot_file = os.path.join(args.output_dir, "noise_robustness_curve.png")
    plt.savefig(plot_file, dpi=150)
    logger.info(f"Curve saved to {plot_file}")
    plt.close()

if __name__ == "__main__":
    main()