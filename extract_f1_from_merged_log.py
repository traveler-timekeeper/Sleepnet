import re

log_file = "./output/output.log"

with open(log_file, "r") as f:
    lines = f.readlines()

folds = []          # 每个元素是一个 list，包含该 fold 所有 epoch 的行
current_fold = []   # 当前 fold 的行

for line in lines:
    # 当遇到 "Load generated SIDs" 时，表示新的一折开始
    if "Load generated SIDs" in line:
        if current_fold:   # 保存上一折
            folds.append(current_fold)
            current_fold = []
    current_fold.append(line)
if current_fold:    # 最后一折
    folds.append(current_fold)

print(f"检测到 {len(folds)} 个 fold")

# 对每个 fold，提取最后一次 "Saved best checkpoint" 前的 TE f1
for i, fold_lines in enumerate(folds):
    best_f1 = None
    # 从后往前找 "Saved best checkpoint"
    for j in range(len(fold_lines)-1, -1, -1):
        if "Saved best checkpoint" in fold_lines[j]:
            # 找到它之前的一行（或同行的上一行）来提取 TE f1
            # 标志行格式为: [INFO ] [eXX/200 ...] TR ... VA ... TE ...
            # 通常这行紧挨在 "Saved best checkpoint" 行前面
            if j > 0:
                prev_line = fold_lines[j-1]
                # 用正则提取 TE 的 f1=xx.x
                match = re.search(r'TE.*?f1=([0-9.]+)', prev_line)
                if match:
                    best_f1 = float(match.group(1))
                    break
    if best_f1 is not None:
        print(f"Fold {i}: TE F1 = {best_f1:.1f}")
    else:
        print(f"Fold {i}: NOT_FOUND")