import os
import re
import numpy as np

# ================== 配置 ==================
BASE_DIR = "./output/bilstm_attn_tune_fold"   # 输出根路径
FOLDS = [0, 1, 2, 3, 5, 6, 7, 8, 9]           # 需要汇总的折号
LOG_FILENAME = "predict.log"                   # 日志文件名
# ====================================================================

def extract_metrics(log_path):
    """从单个 predict.log 中提取 Acc, MF1, 混淆矩阵, Per-class F1"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"找不到 {log_path}")
        return None

    metrics = {}

    # 1. 总体准确率 (n=..., acc=xx.x)
    match_acc = re.search(r'n=\d+,\s*acc=([\d.]+)', content)
    if match_acc:
        metrics['acc'] = float(match_acc.group(1))

    # 2. 宏平均F1 (mf1=xx.x)
    match_mf1 = re.search(r'mf1=([\d.]+)', content)
    if match_mf1:
        metrics['mf1'] = float(match_mf1.group(1))

    # 3. 各类别F1（Per-class F1-Score: ...）
    match_f1s = re.search(r'Per-class F1-Score:\s*([\d.]+ [\d.]+ [\d.]+ [\d.]+ [\d.]+)', content)
    if match_f1s:
        metrics['f1_per_class'] = [float(x) for x in match_f1s.group(1).split()]

    # 4. 混淆矩阵 (>> Confusion Matrix 之后的 [[...]])
    match_cm = re.search(r'>> Confusion Matrix.*?\[\[(.*?)\]\]', content, re.DOTALL)
    if match_cm:
        cm_str = match_cm.group(1)
        # 将字符串转为二维数组
        rows = cm_str.strip().split('\n')
        cm = []
        for row in rows:
            row = row.strip().replace('[', '').replace(']', '')
            if row:
                cm.append([int(x) for x in row.split()])
        if cm:
            metrics['cm'] = np.array(cm)

    # 5. 计算 Cohen's Kappa
    if 'cm' in metrics and 'acc' in metrics:
        cm = metrics['cm']
        N = cm.sum()
        p_o = metrics['acc'] / 100.0   # 准确率是百分比，转为小数
        # 各类真实总数(行和)与预测总数(列和)
        row_sum = cm.sum(axis=1)
        col_sum = cm.sum(axis=0)
        p_e = np.dot(row_sum, col_sum) / (N * N)
        if (1 - p_e) != 0:
            metrics['kappa'] = (p_o - p_e) / (1 - p_e)
        else:
            metrics['kappa'] = 0.0

    return metrics

def main():
    acc_all, mf1_all, kappa_all = [], [], []
    f1_class_all = [[] for _ in range(5)]  # 5个类别的F1列表

    print("=" * 60)
    print("汇总结果...")
    print("=" * 60)

    for fold in FOLDS:
        log_path = os.path.join(BASE_DIR + str(fold), LOG_FILENAME)
        m = extract_metrics(log_path)

        if m is None:
            print(f"Fold {fold}: 提取失败，跳过")
            continue

        # 检查必需字段
        if all(k in m for k in ('acc', 'mf1', 'kappa', 'f1_per_class')):
            acc_all.append(m['acc'])
            mf1_all.append(m['mf1'])
            kappa_all.append(m['kappa'] * 100)   # 转为百分比便于显示
            for i in range(5):
                f1_class_all[i].append(m['f1_per_class'][i])
            print(f"Fold {fold}: Acc={m['acc']:.2f}%, MF1={m['mf1']:.2f}%, κ={m['kappa']:.4f}")
        else:
            print(f"Fold {fold}: 缺少部分指标，跳过")

    if len(acc_all) == 0:
        print("\n没有有效数据，请检查路径和日志。")
        return

    # 计算均值与标准差
    def get_mean_std(lst):
        return np.mean(lst), np.std(lst, ddof=1)

    acc_m, acc_s = get_mean_std(acc_all)
    mf1_m, mf1_s = get_mean_std(mf1_all)
    kappa_m, kappa_s = get_mean_std(kappa_all)   # 已经是百分比数值

    f1_class_m = [np.mean(f1_class_all[i]) for i in range(5)]
    f1_class_s = [np.std(f1_class_all[i], ddof=1) for i in range(5)]

    print("\n" + "=" * 60)
    print("10折交叉验证最终结果")
    print("=" * 60)
    print(f"总体准确率 (Acc)     : {acc_m:.2f}% ± {acc_s:.2f}%")
    print(f"宏平均F1   (MF1)     : {mf1_m:.2f}% ± {mf1_s:.2f}%")
    print(f"Cohen's Kappa (κ)    : {kappa_m:.2f}% ± {kappa_s:.2f}%")
    print("-" * 60)
    classes = ['W', 'N1', 'N2', 'N3', 'REM']
    for i, cls in enumerate(classes):
        print(f"  {cls} F1-Score        : {f1_class_m[i]:.2f}% ± {f1_class_s[i]:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()