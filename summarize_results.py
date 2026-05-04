import os
import re
import numpy as np

# ================== 配置区域（根据你的实际路径修改） ==================
BASE_DIR = "./output/bilstm_attn_tune_fold"   # 各折目录的前缀
FOLDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]        # 所有折号
LOG_FILENAME = "predict.log"                   # 每折的评估日志文件名
OUTPUT_DIR = "./output/"                       # 汇总日志的输出目录
# ===================================================================

def extract_data_from_log(log_path):
    """从单个 predict.log 中提取所需数据"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"⚠️ 警告：找不到 {log_path}，跳过该折")
        return None

    data = {}

    # 提取准确率
    match_acc = re.search(r'n=\d+,\s*acc=([\d.]+)', content)
    if match_acc:
        data['acc'] = float(match_acc.group(1))

    # 提取宏平均F1
    match_mf1 = re.search(r'mf1=([\d.]+)', content)
    if match_mf1:
        data['mf1'] = float(match_mf1.group(1))

    # 提取各类样本数
    match_n = re.search(r'W:\s*(\d+).*?N1:\s*(\d+).*?N2:\s*(\d+).*?N3:\s*(\d+).*?REM:\s*(\d+)', content, re.DOTALL)
    if match_n:
        data['n_per_class'] = [int(match_n.group(i)) for i in range(1, 6)]

    # 提取混淆矩阵
    match_cm = re.search(r'>> Confusion Matrix.*?\[\[(.*?)\]\]', content, re.DOTALL)
    if match_cm:
        cm_str = match_cm.group(1)
        rows = cm_str.strip().split('\n')
        cm = []
        for row in rows:
            row = row.strip().replace('[', '').replace(']', '')
            if row:
                cm.append([int(x) for x in row.split()])
        if cm:
            data['cm'] = np.array(cm)

    return data

def main():
    # 累积所有折的数据
    total_cm = None
    all_acc = []
    all_mf1 = []

    print("正在读取各折数据...")
    for fold in FOLDS:
        log_path = os.path.join(BASE_DIR + str(fold), LOG_FILENAME)
        data = extract_data_from_log(log_path)

        if data is None or 'cm' not in data:
            print(f"Fold {fold}: ❌ 数据不全，跳过")
            continue

        # 累加混淆矩阵
        if total_cm is None:
            total_cm = data['cm']
        else:
            total_cm += data['cm']

        all_acc.append(data.get('acc', 0))
        all_mf1.append(data.get('mf1', 0))
        print(f"Fold {fold}: Acc={data.get('acc', 0):.1f}%, MF1={data.get('mf1', 0):.1f}%")

    if total_cm is None:
        print("❌ 无有效数据！")
        return

    # 计算各类别总数
    n_per_class = total_cm.sum(axis=1)

    # 计算总体准确率
    correct = np.trace(total_cm)
    total = total_cm.sum()
    overall_acc = correct / total * 100

    # 计算每类 Precision, Recall, F1
    precision_list = []
    recall_list = []
    f1_list = []
    for i in range(5):
        tp = total_cm[i, i]
        fp = total_cm[:, i].sum() - tp
        fn = total_cm[i, :].sum() - tp
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # 宏平均F1（直接平均各类F1或取fold均值）
    mf1_mean = np.mean(all_mf1) if all_mf1 else 0
    acc_mean = np.mean(all_acc) if all_acc else 0

    # 计算Kappa
    p_o = correct / total
    row_sum = total_cm.sum(axis=1)
    col_sum = total_cm.sum(axis=0)
    p_e = np.dot(row_sum, col_sum) / (total * total)
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0

    # 打印最终结果（符合你的日志格式）
    print("\n" + "=" * 60)
    print("=" * 60)

    classes = ['W', 'N1', 'N2', 'N3', 'REM']
    log_lines = []
    log_lines.append("=== Overall ===")
    for i, cls in enumerate(classes):
        log_lines.append(f"  {cls}: {int(n_per_class[i])}")
    log_lines.append(f"  n={int(total)}, acc={acc_mean:.1f}, mf1={mf1_mean:.1f}")
    log_lines.append("  >> Confusion Matrix")
    for i, row in enumerate(total_cm):
        log_lines.append(f"  [{', '.join([str(x) for x in row])}]")
    log_lines.append(f"  Total: {int(total)}")
    log_lines.append(f"  Number of samples from each class: [{', '.join([str(int(x)) for x in n_per_class])}]")
    log_lines.append(f"  Accuracy: {acc_mean:.1f}")
    log_lines.append(f"  Macro F1-Score: {mf1_mean:.1f}")
    log_lines.append(f"  Cohen's Kappa: {kappa:.4f}")
    log_lines.append(f"  Per-class Precision: {', '.join([f'{p:.1f}' for p in precision_list])}")
    log_lines.append(f"  Per-class Recall: {', '.join([f'{r:.1f}' for r in recall_list])}")
    log_lines.append(f"  Per-class F1-Score: {', '.join([f'{f:.1f}' for f in f1_list])}")

    # 输出到终端
    for line in log_lines:
        print(line)

    # 保存到文件
    output_path = os.path.join(OUTPUT_DIR, "predict.log")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(log_lines) + '\n')

    return log_lines

if __name__ == "__main__":
    main()