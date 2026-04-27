import sys, os
import numpy as np
import torch
import sklearn.metrics as skmetrics
import argparse
import importlib
import copy

sys.path.insert(0, os.path.dirname(__file__))
from model import Model
from minibatching import iterate_batch_multiple_seq_minibatches

def load_nights(file_list, data_dir):
    xs, ys = [], []
    with open(file_list) as f:
        for name in f:
            name = name.strip()
            if not name: continue
            d = np.load(os.path.join(data_dir, name))
            xs.append(d['x'])
            ys.append(d['y'])
    return xs, ys

def compute_mf1(model, device, x_list, y_list, config):
    model.tsn.eval()
    trues, preds = [], []
    with torch.no_grad():
        for x, y in zip(x_list, y_list):
            it = iterate_batch_multiple_seq_minibatches(
                [x], [y], batch_size=config['batch_size'],
                seq_length=config['seq_length'],
                shuffle_idx=None, augment_seq=False)
            for bx, by, w, sl, re in it:
                bx = torch.from_numpy(bx).float().to(device).view(-1, 1, 3000)
                by = torch.from_numpy(by).long().to(device)
                if re:
                    state = (torch.zeros(2, config['batch_size'], config['n_rnn_units']).to(device),
                             torch.zeros(2, config['batch_size'], config['n_rnn_units']).to(device))
                y_pred, state = model.tsn(bx, state)
                preds.extend(y_pred.argmax(1).cpu().numpy())
                trues.extend(by.cpu().numpy())
    mf1 = skmetrics.f1_score(trues, preds, average='macro')
    return mf1, trues, preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config/isruc_finetune.py')
    parser.add_argument('--pretrained_dir', required=True)
    parser.add_argument('--fold_idx', type=int, default=4)
    parser.add_argument('--output_dir', default='./output/isruc_finetune')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train_list', default='isruc_train.txt')
    parser.add_argument('--val_list', default='isruc_val.txt')
    parser.add_argument('--test_list', default='isruc_test.txt')
    parser.add_argument('--data_dir', default='./data/isruc')
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("*", args.config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict
    config['seq_length'] = 20
    config['batch_size'] = 10

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    train_x, train_y = load_nights(args.train_list, args.data_dir)
    val_x, val_y = load_nights(args.val_list, args.data_dir)
    test_x, test_y = load_nights(args.test_list, args.data_dir)
    print(f'Train nights: {len(train_x)}, Val: {len(val_x)}, Test: {len(test_x)}')

    config['class_weights'] = np.ones(config['n_classes'], dtype=np.float32)
    model = Model(config=config, output_dir=os.path.join(args.pretrained_dir, str(args.fold_idx)),
                  use_rnn=True, testing=False, use_best=False, device=device)
    ckpt = os.path.join(args.pretrained_dir, str(args.fold_idx), 'best_ckpt', 'best_model.ckpt')
    model.tsn.load_state_dict(torch.load(ckpt, map_location=device))
    model.tsn.to(device)

    optimizer = torch.optim.Adam(model.tsn.parameters(), lr=config.get('learning_rate', 1e-5))
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val_mf1 = -1
    best_weights = None

    for epoch in range(config.get('n_epochs', 10)):
        model.tsn.train()
        idx = np.random.permutation(len(train_x))
        stat = None
        losses = []
        minibatch_fn = iterate_batch_multiple_seq_minibatches(
            [train_x[i] for i in idx], [train_y[i] for i in idx],
            batch_size=config['batch_size'], seq_length=config['seq_length'],
            shuffle_idx=idx, augment_seq=False)
        for bx, by, w, sl, re in minibatch_fn:
            bx = torch.from_numpy(bx).float().to(device).view(-1, 1, 3000)
            by = torch.from_numpy(by).long().to(device)
            if re or stat is None:
                stat = (torch.zeros(2, config['batch_size'], config['n_rnn_units']).to(device),
                        torch.zeros(2, config['batch_size'], config['n_rnn_units']).to(device))
            stat = (stat[0].detach(), stat[1].detach())
            optimizer.zero_grad()
            pred, stat = model.tsn(bx, stat)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_mf1, _, _ = compute_mf1(model, device, val_x, val_y, config)
        print(f'Epoch {epoch+1:2d}, loss: {np.mean(losses):.4f}, val MF1: {val_mf1*100:.2f}%')
        if val_mf1 > best_val_mf1:
            best_val_mf1 = val_mf1
            best_weights = copy.deepcopy(model.tsn.state_dict())

    # 最终测试
    model.tsn.load_state_dict(best_weights)
    test_mf1, trues, preds = compute_mf1(model, device, test_x, test_y, config)
    acc = skmetrics.accuracy_score(trues, preds) * 100
    cm = skmetrics.confusion_matrix(trues, preds, labels=[0,1,2,3,4])

    print(f'\nFinal test: Acc={acc:.2f}%, MF1={test_mf1*100:.2f}%')
    print(cm)

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(os.path.join(args.output_dir, 'finetune_results.npz'), acc=acc, mf1=test_mf1*100, cm=cm)
    with open(os.path.join(args.output_dir, 'finetune_results.txt'), 'w') as f:
        f.write(f'Accuracy: {acc:.2f}%\nMacro F1: {test_mf1*100:.2f}%\n{cm}\n')

if __name__ == '__main__':
    main()