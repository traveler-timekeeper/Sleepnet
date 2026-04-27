import os
import glob

def main(data_dir, dataset_name):
    files = glob.glob(os.path.join(data_dir, '*.npz'))
    files.sort()
    n = len(files)
    txt = dataset_name + '.txt'
    with open(txt, 'w') as f:
        for i in range(n):
            f.write(str(i) + '\n')
    print(f'Wrote {n} ids to {txt}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/isruc')
    parser.add_argument('--dataset', type=str, default='isruc')
    args = parser.parse_args()
    main(args.data_dir, args.dataset)
