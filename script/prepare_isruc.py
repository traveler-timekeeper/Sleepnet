import argparse
import glob
import os
import ntpath
import shutil
import numpy as np
try:
    import pyedflib
    _USE_PYEDFLIB = True
except Exception:
    _USE_PYEDFLIB = False
    import mne

from logger import get_logger


label_map = {
    'W': 0, 'Wake': 0,
    'N1': 1, '1': 1,
    'N2': 2, '2': 2,
    'N3': 3, '3': 3, 'N4': 3,
    'R': 4, 'REM': 4,
    'M': 5, 'MOVE': 5,
    '?': 6, 'UNK': 6
}


def read_annotation_file(ann_path):
    """Try to read annotation file that contains one label per epoch (txt/csv)."""
    with open(ann_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    # Try to parse as single-column labels
    labels = []
    for l in lines:
        if l in label_map:
            labels.append(label_map[l])
        else:
            # try numeric
            try:
                labels.append(int(l))
            except Exception:
                # try first token
                tok = l.split()[0]
                if tok in label_map:
                    labels.append(label_map[tok])
                else:
                    try:
                        labels.append(int(tok))
                    except Exception:
                        raise Exception(f"Unknown label token: {l} in {ann_path}")

    return np.array(labels, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/isruc/', help='ISRUC data root directory')
    parser.add_argument('--output_dir', type=str, default='./data/isruc', help='Output dir for npz files')
    parser.add_argument('--select_ch', type=str, default=None, help='Channel name to select (default: first EEG)')
    parser.add_argument('--use_mne', action='store_true', help='Force using mne instead of pyedflib')
    parser.add_argument('--log_file', type=str, default='info_isruc.log')
    args = parser.parse_args()

    # allow forcing mne fallback to avoid pyedflib heavy memory usage
    if getattr(args, 'use_mne', False):
        global _USE_PYEDFLIB
        _USE_PYEDFLIB = False

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, args.log_file)
    logger = get_logger(log_path, level='info')

    # Find EDF files (accept both .edf and .EDF)
    edf_files = glob.glob(os.path.join(args.data_dir, '**', '*.edf'), recursive=True)
    edf_files += glob.glob(os.path.join(args.data_dir, '**', '*.EDF'), recursive=True)
    edf_files += glob.glob(os.path.join(args.data_dir, '**', '*.rec'), recursive=True)
    edf_files += glob.glob(os.path.join(args.data_dir, '**', '*.REC'), recursive=True)
    edf_files.sort()

    if len(edf_files) == 0:
        raise Exception('No EDF files found under data_dir')

    for edf in edf_files:
        logger.info(f'Processing {edf}')
        base = ntpath.basename(edf)
        name_no_ext = os.path.splitext(base)[0]

        # common ann patterns
        ann_candidates = []
        for ext in ['.txt', '.csv', '.ann', '_ann.txt', '_hyp.txt', '-Hypnogram.edf', '_hypnogram.edf']:
            p = os.path.join(os.path.dirname(edf), name_no_ext + ext)
            if os.path.exists(p):
                ann_candidates.append(p)

        # If no same-base ann found, try find any annotation EDF in same folder
        if len(ann_candidates) == 0:
            for f in glob.glob(os.path.join(os.path.dirname(edf), '*')):
                if f.endswith('Hypnogram.edf') or f.endswith('hypnogram.edf'):
                    ann_candidates.append(f)

        # Also try patterns like name_1.txt, name-1.txt, etc.
        if len(ann_candidates) == 0:
            txts = glob.glob(os.path.join(os.path.dirname(edf), name_no_ext + '*.txt'))
            txts += glob.glob(os.path.join(os.path.dirname(edf), name_no_ext + '*.csv'))
            txts += glob.glob(os.path.join(os.path.dirname(edf), '*.txt'))
            txts.sort()
            if len(txts) > 0:
                ann_candidates.extend(txts)

        ann_labels = None

        if _USE_PYEDFLIB:
            try:
                edf_f = pyedflib.EdfReader(edf)
            except Exception as e:
                logger.info(f'  skip (not readable): {e}')
                continue

            epoch_duration = edf_f.datarecord_duration
            # read annotations from EDF if present
            if len(ann_candidates) > 0 and ann_candidates[0].endswith('.edf'):
                ann_f = pyedflib.EdfReader(ann_candidates[0])
                ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()
                labels_list = []
                for a in range(len(ann_stages)):
                    onset_sec = int(ann_onsets[a])
                    duration_sec = int(ann_durations[a])
                    ann_str = ''.join(ann_stages[a])
                    if ann_str in label_map:
                        lab = label_map[ann_str]
                    else:
                        tok = ann_str.split()[0]
                        lab = label_map.get(tok, None)
                        if lab is None:
                            raise Exception(f'Unknown ann label: {ann_str}')
                    n_epoch = int(duration_sec / epoch_duration)
                    labels_list.append(np.ones(n_epoch, dtype=np.int32) * lab)
                ann_labels = np.hstack(labels_list)
            elif len(ann_candidates) > 0:
                ann_labels = read_annotation_file(ann_candidates[0])

            # If no annotation found, skip
            if ann_labels is None:
                logger.info('  no annotation found, skip')
                edf_f.close()
                continue

            # select channel
            ch_idx = 0
            ch_names = edf_f.getSignalLabels()
            if args.select_ch is not None:
                for i, c in enumerate(ch_names):
                    if c == args.select_ch:
                        ch_idx = i
                        break
            sampling_rate = edf_f.getSampleFrequency(ch_idx)
            n_epoch_samples = int(epoch_duration * sampling_rate)
            sig = edf_f.readSignal(ch_idx).reshape(-1, n_epoch_samples)

            # Trim/pad labels to match epochs
            n_epochs = sig.shape[0]
            if len(ann_labels) > n_epochs:
                ann_labels = ann_labels[:n_epochs]
            elif len(ann_labels) < n_epochs:
                pad = np.ones(n_epochs - len(ann_labels), dtype=np.int32) * 0
                ann_labels = np.concatenate([ann_labels, pad])

            x = sig.astype(np.float32)
            y = ann_labels.astype(np.int32)

            filename = name_no_ext + '.npz'
            save_dict = {
                'x': x,
                'y': y,
                'fs': sampling_rate,
                'ch_label': ch_names[ch_idx],
                'start_datetime': None,
                'file_duration': edf_f.getFileDuration(),
                'epoch_duration': epoch_duration,
                'n_all_epochs': edf_f.datarecords_in_file,
                'n_epochs': len(x),
            }
            np.savez(os.path.join(args.output_dir, filename), **save_dict)
            logger.info(f'  saved -> {os.path.join(args.output_dir, filename)}')

            edf_f.close()

        else:
            # Use mne as fallback
            # mne may reject files with .rec extension even if they are EDF;
            # for .rec, create a temporary .edf copy and read that
            tmp_file = None
            try:
                read_path = edf
                if edf.lower().endswith('.rec'):
                    tmp_file = os.path.join('/tmp', name_no_ext + '.edf')
                    shutil.copy(edf, tmp_file)
                    read_path = tmp_file
                # Use preload=False to avoid loading entire file into memory
                raw = mne.io.read_raw_edf(read_path, preload=False, verbose=False)
            except Exception as e:
                logger.info(f'  skip (mne failed to read): {e}')
                if tmp_file is not None and os.path.exists(tmp_file):
                    os.remove(tmp_file)
                continue

            # Default epoch duration (ISRUC uses 30s epochs)
            epoch_duration = 30

            ch_names = raw.ch_names
            ch_idx = 0
            if args.select_ch is not None and args.select_ch in ch_names:
                ch_idx = ch_names.index(args.select_ch)

            sampling_rate = int(raw.info['sfreq'])
            n_epoch_samples = int(epoch_duration * sampling_rate)
            n_epochs = raw.n_times // n_epoch_samples

            # Use memmap to avoid large memory spikes: write epochs one-by-one
            tmp_x_path = os.path.join(args.output_dir, name_no_ext + '.npy.tmp')
            mm = np.memmap(tmp_x_path, dtype=np.float32, mode='w+', shape=(n_epochs, n_epoch_samples))
            for ei in range(n_epochs):
                start_samp = ei * n_epoch_samples
                stop_samp = start_samp + n_epoch_samples
                chunk = raw.get_data(picks=[ch_idx], start=start_samp, stop=stop_samp)
                mm[ei, :] = chunk.squeeze().astype(np.float32)
            sig = np.asarray(mm)
            # remove memmap backing file after copying to ndarray
            try:
                os.remove(tmp_x_path)
            except Exception:
                pass

            # read annotations if present
            if len(ann_candidates) > 0 and ann_candidates[0].endswith('.edf'):
                try:
                    ann = mne.read_annotations(ann_candidates[0])
                    onsets = ann.onset
                    durations = ann.duration
                    descriptions = ann.description
                    labels_list = []
                    for a in range(len(descriptions)):
                        desc = descriptions[a]
                        tok = desc.split()[0]
                        lab = label_map.get(tok, None)
                        if lab is None:
                            try:
                                lab = int(tok)
                            except Exception:
                                raise Exception(f'Unknown ann label: {desc}')
                        n_epoch = int(durations[a] / epoch_duration)
                        labels_list.append(np.ones(n_epoch, dtype=np.int32) * lab)
                    ann_labels = np.hstack(labels_list)
                except Exception:
                    ann_labels = None
            elif len(ann_candidates) > 0:
                ann_labels = read_annotation_file(ann_candidates[0])

            if ann_labels is None:
                logger.info('  no annotation found, skip')
                continue

            if len(ann_labels) > n_epochs:
                ann_labels = ann_labels[:n_epochs]
            elif len(ann_labels) < n_epochs:
                pad = np.ones(n_epochs - len(ann_labels), dtype=np.int32) * 0
                ann_labels = np.concatenate([ann_labels, pad])

            x = sig.astype(np.float32)
            y = ann_labels.astype(np.int32)

            filename = name_no_ext + '.npz'
            save_dict = {
                'x': x,
                'y': y,
                'fs': sampling_rate,
                'ch_label': ch_names[ch_idx],
                'start_datetime': None,
                'file_duration': raw.n_times / sampling_rate,
                'epoch_duration': epoch_duration,
                'n_all_epochs': n_epochs,
                'n_epochs': len(x),
            }
            np.savez(os.path.join(args.output_dir, filename), **save_dict)
            logger.info(f'  saved -> {os.path.join(args.output_dir, filename)}')

            # remove copied tmp file if created
            if tmp_file is not None and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass


if __name__ == '__main__':
    main()
