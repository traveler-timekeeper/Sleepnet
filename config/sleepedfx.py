params = {
    # Train
    "n_epochs": 200,


#原
    #"learning_rate": 1e-4,
    #改动，双向 LSTM 参数更多，收敛可能变慢
    "learning_rate": 5e-5,




    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,

    #"clip_grad_value": 5.0,
    "clip_grad_value": 1.0,
    "dropout_rnn": 0.6, 


    "evaluate_span": 50,
    "checkpoint_span": 50,

    # Early-stopping
    #"no_improve_epochs": 50,
    "no_improve_epochs": 30,


    # Model
    "model": "model-mod-8",
    "n_rnn_layers": 1,
    "n_rnn_units": 128,
    "sampling_rate": 100.0,
    "input_size": 3000,
    "n_classes": 5,
    "l2_weight_decay": 1e-3,

    # Dataset
    "dataset": "sleepedfx",
    
    "data_dir": "./data/sleepedf/sleep-cassette/eeg_fpz_cz",
    "n_folds": 10,
    "n_subjects": 78,

    # Data Augmentation
    "augment_seq": True,
    "augment_signal_full": True,
    "weighted_cross_ent": True,
}



# train = params.copy()
# #原
# # train.update({
# #     "seq_length": 20,
# #     "batch_size": 15,

# #改动
#     "bidirectional": True,   # 启用双向 LSTM

# })


#-----------------注意力-----------------
train = params.copy()
train.update({
    "seq_length": 20,          # 可尝试改为 25 或 30
    "batch_size": 10,          # 从 15 降下来，稳定梯度
    "num_heads": 2,            # 从头数 4 降为 2
    "learning_rate": 5e-5,     # 降低学习率
})
#-----------------注意力-----------------


predict = params.copy()
predict.update({
    "batch_size": 1,
    "seq_length": 1,
})
