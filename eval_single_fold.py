import sys, os, glob
import numpy as np
import torch
import argparse
import sklearn.metrics as skmetrics

sys.path.insert(0, os.path.dirname(__file__))
from model import Model
from data import load_data, get_subject_files
from minibatching import iterate_batch_multiple_seq_minibatches
from utils import load_seq_ids
from logger import get_logger
import importlib

def evaluate_fold(config_file, model_dir, fold_idx, output_dir, gpu=0):
    # 加载配置
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 自动判断设备，无 GPU 则用 CPU
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    logger = get_logger(os.path.join(output_dir, "predict.log"), level="info")

    subject_files = glob.glob(os.path.join(config["data_dir"], "*.npz"))
    fname = f"{config['dataset']}.txt"
    seq_sids = load_seq_ids(fname)

    # 划分测试集
    fold_pids = np.array_split(seq_sids, config["n_folds"])
    test_sids = fold_pids[fold_idx]
    logger.info("Test SIDs: ({}) {}".format(len(test_sids), test_sids))

    # 加载测试集数据（仅该折受试者）
    test_files = []
    for sid in test_sids:
        test_files.append(get_subject_files(config["dataset"], subject_files, sid))
    test_files = np.hstack(test_files)
    test_x, test_y, _ = load_data(test_files)

    # 创建模型，先不加载最佳模型，稍后手动加载
    config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)
    model = Model(config=config, output_dir=os.path.join(model_dir, str(fold_idx)),
                  use_rnn=True, testing=True, use_best=False, device=device)

    # 手动加载最佳模型权重，并强制映射到 CPU（兼容 GPU 训练的模型在 CPU 上加载）
    best_ckpt_path = os.path.join(model_dir, str(fold_idx), "best_ckpt", "best_model.ckpt")
    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {best_ckpt_path}")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.tsn.load_state_dict(checkpoint)
    model.tsn.to(device)
    logger.info(f"Loaded best model from {best_ckpt_path}")

    # 评估
    trues, preds = [], []
    for night_x, night_y in zip(test_x, test_y):
        minibatch_fn = iterate_batch_multiple_seq_minibatches(
            [night_x], [night_y],
            batch_size=config["batch_size"],
            seq_length=config["seq_length"],
            shuffle_idx=None, augment_seq=False)
        outs = model.evaluate_with_dataloader(minibatch_fn)
        trues.extend(outs["test/trues"])
        preds.extend(outs["test/preds"])

    acc = skmetrics.accuracy_score(trues, preds) * 100
    mf1 = skmetrics.f1_score(trues, preds, average='macro') * 100
    cm = skmetrics.confusion_matrix(trues, preds, labels=[0,1,2,3,4])

    logger.info("n={}, acc={:.1f}, mf1={:.1f}".format(len(trues), acc, mf1))
    logger.info(">> Confusion Matrix")
    logger.info(cm)

    # 保存预测结果
    np.savez(os.path.join(output_dir, "predictions.npz"), y_true=trues, y_pred=preds)
    logger.info("Saved predictions to {}".format(output_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--fold_idx", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    evaluate_fold(args.config_file, args.model_dir, args.fold_idx, args.output_dir, args.gpu)