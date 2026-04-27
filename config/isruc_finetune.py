# config/isruc_finetune.py
params = {
    # ---------- 微调训练参数 ----------
    "n_epochs": 10,            # 少量 epoch 防止过拟合
    "learning_rate": 1e-5,     # 极小学习率，稳定预训练权重
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "clip_grad_value": 1.0,    # 梯度裁剪，稳定训练

    # 评估与早停（微调时评估间隔可缩短）
    "evaluate_span": 5,
    "checkpoint_span": 5,
    "no_improve_epochs": 5,    # 早停耐心设置为 5 个 epoch

    # ---------- 模型结构（必须与预训练模型一致） ----------
    "model": "model-mod-8",
    "n_rnn_layers": 1,
    "n_rnn_units": 128,
    "sampling_rate": 100.0,
    "input_size": 3000,
    "n_classes": 5,
    "l2_weight_decay": 1e-3,
    "dropout_rnn": 0.6,        # 保留原设置，但不影响微调

    # ---------- ISRUC 数据集 ----------
    "dataset": "isruc",
    "data_dir": "./data/isruc",    # 预处理后的 ISRUC .npz 文件目录
    "n_folds": 1,              # 不使用交叉验证，由外部文件划分
    "n_subjects": 1,           # 占位

    # ---------- 关闭数据增强与加权损失 ----------
    "augment_seq": False,      # 微调时关闭序列增强
    "augment_signal_full": False, # 关闭信号偏移增强
    "weighted_cross_ent": False,  # 使用标准交叉熵
}

# 训练时使用的 batch_size 和序列长度
train = params.copy()
train.update({
    "seq_length": 20,          # 必须与预训练模型一致
    "batch_size": 10,          # 微调时可稍大，GPU 内存允许可设 16
    "num_heads": 2,            # 注意力头数，与最终模型一致
})

# 评估时的配置（通常 batch_size=1 或 10，seq_length=20）
predict = params.copy()
predict.update({
    "batch_size": 1,           # 评估时建议使用 1，避免状态对齐问题
    "seq_length": 1,           # 评估时序列长度由 minibatch 控制，这里设为 1 不影响
})