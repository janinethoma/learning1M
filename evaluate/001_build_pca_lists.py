import os

import numpy as np
from distloss.cold_helper import get_recursive_file_list, parse_file_list

from learnlarge.util.helper import mkdir, fs_root
from learnlarge.util.io import load_csv
from learnlarge.util.io import save_csv

out_root = os.path.join(fs_root(), 'lists')
N_SAMPLES = 5000

mkdir(out_root)

# Oxford
place = 'oxford'


def img_path(info):
    date = info[0]
    folder = info[1]
    t = info[2]
    return os.path.join('datasets/oxford_512', '{}_stereo_centre_{:02d}'.format(date, int(folder)), '{}.png'.format(t))


# Preselected reference
preselected_ref = os.path.join(fs_root(), 'data/learnlarge/shuffled/train_ref_000.csv')
p_meta = load_csv(preselected_ref)
p_meta['path'] = [img_path((d, f, t)) for d, f, t in
                  zip(p_meta['date'], p_meta['folder'], p_meta['t'])]
idxs_to_keep = np.linspace(0, len(p_meta['path']), num=N_SAMPLES, endpoint=False, dtype=int)
for key in p_meta.keys():
    p_meta[key] = [p_meta[key][i] for i in idxs_to_keep]
save_csv(p_meta, os.path.join(out_root, '{}_pca.csv'.format(place)))

# Cold
place = 'cold'


def parse_cold_folder(path, pattern):
    all_files = get_recursive_file_list(path, pattern)
    all_files, TXYA = parse_file_list(all_files)

    if len(all_files) > N_SAMPLES:
        idxs_to_keep = np.linspace(0, len(all_files), num=N_SAMPLES, endpoint=False, dtype=int)
    else:
        idxs_to_keep = np.arange(len(all_files))

    meta = dict()
    meta['path'] = [all_files[i] for i in idxs_to_keep]
    meta['yaw'] = [TXYA[i, 3] for i in idxs_to_keep]
    meta['easting'] = [TXYA[i, 1] for i in idxs_to_keep]
    meta['northing'] = [TXYA[i, 2] for i in idxs_to_keep]
    meta['t'] = [TXYA[i, 0] for i in idxs_to_keep]

    return meta


train_set = os.path.join(fs_root(), 'data/datasets/cold/freiburg')
train_ref_filter = '*seq[12]*[124]/std_cam/*.jpeg'
meta = parse_cold_folder(train_set, train_ref_filter)

ref_sort_i = np.argsort(meta['t'])
for key in meta.keys():
    meta[key] = [meta[key][i] for i in ref_sort_i]

meta['path'] = [os.path.join(train_set, p) for p in meta['path']]
save_csv(meta, os.path.join(out_root, '{}_{}.csv'.format(place, 'pca')))
