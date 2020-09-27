import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KDTree
from tqdm import tqdm

from learnlarge.util.helper import fs_root
from learnlarge.util.helper import mkdir
from learnlarge.util.io import load_csv, save_csv, load_pickle
from learnlarge.util.meta import get_xy

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

np.random.RandomState(seed=42)

out_root = os.path.join(fs_root(), 'list_plots')
list_out_root = os.path.join(fs_root(), 'lists')

mkdir(out_root)
mkdir(list_out_root)

rows = 1
cols = 4

f, axs = plt.subplots(rows, cols, constrained_layout=False)
if rows == 1:
    axs = np.expand_dims(axs, 0)
if cols == 1:
    axs = np.expand_dims(axs, 1)
f.tight_layout()
f.set_figheight(2.5)  # 8.875in textheight
f.set_figwidth(13)  # 6.875in textwidth

# ------------------------------------- Pittsburgh -------------------------------------
place = 'pittsburgh'
data = loadmat('pittsburgh.mat')
ref_paths = [i[0][0] for i in data['dbImageFns']]
query_paths = [i[0][0] for i in data['qImageFns']]
ref_xy = np.array(data['utmDb'], dtype=float).transpose()
query_xy = np.array(data['utmQ'], dtype=float).transpose()

query_x = [query_xy[i, 0] for i in range(len(query_paths))]
query_y = [query_xy[i, 1] for i in range(len(query_paths))]

# Remove References more than 100m away from query region
query_tree = KDTree(query_xy)
d_to_query, _ = query_tree.query(ref_xy, return_distance=True, k=1)

# Remove references far away from queries
max_dist = 100
remaining_ref_i = [i for i, d in enumerate(d_to_query) if d < max_dist]

# Remove sky-views
remaining_ref_i = [i for i in remaining_ref_i if 'pitch1' in ref_paths[i]]
remaining_query_i = [i for i in range(len(query_paths)) if 'pitch1' in query_paths[i]]

print('Remaining queries: {}\n'
      'Remaining ref: {}'.format(len(remaining_query_i), len(remaining_ref_i)))

ax = axs[0, 1]
s1 = ax.scatter([ref_xy[i, 0] for i in remaining_ref_i[::1]], [ref_xy[i, 1] for i in remaining_ref_i[::1]], s=1, c='k')
s2 = ax.scatter([query_xy[i, 0] for i in remaining_query_i[::1]], [query_xy[i, 1] for i in remaining_query_i[::1]], s=1)
s1.set_rasterized(True)
s2.set_rasterized(True)
ax.title.set_text('Pittsburgh test images')
ax.legend(['{} reference images'.format(len(remaining_ref_i)), '{} query images'.format(len(remaining_query_i))],
          markerscale=5)
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')

ref_meta = dict()
ref_meta['path'] = [os.path.join('datasets/pittsburgh_used/ref', ref_paths[i]) for i in remaining_ref_i]
ref_meta['easting'] = [ref_xy[i, 0] for i in remaining_ref_i]
ref_meta['northing'] = [ref_xy[i, 1] for i in remaining_ref_i]
save_csv(ref_meta, os.path.join(list_out_root, '{}_ref.csv'.format(place)))

query_meta = dict()
query_meta['path'] = [os.path.join('datasets/pittsburgh_used/query', query_paths[i]) for i in remaining_query_i]
query_meta['easting'] = [query_xy[i, 0] for i in remaining_query_i]
query_meta['northing'] = [query_xy[i, 1] for i in remaining_query_i]
save_csv(query_meta, os.path.join(list_out_root, '{}_query.csv'.format(place)))

# ------------------------------------- Oxford -------------------------------------
place = 'oxford'


def img_path(info):
    date = info[0]
    folder = info[1]
    t = info[2]
    return os.path.join('datasets/oxford', '{}_stereo_centre_{:02d}'.format(date, int(folder)), '{}.png'.format(t))


# Preselected reference
preselected_ref = os.path.join(fs_root(), 'data/learnlarge/clean_merged_parametrized/test.csv')
ref_date = '2014-12-02-15-30-08'
p_meta = load_csv(preselected_ref)
for key in p_meta.keys():
    p_meta[key] = [e for e, d in zip(p_meta[key], p_meta['date']) if d == ref_date]
p_meta['path'] = [img_path((d, f, t)) for d, f, t in
                  zip(p_meta['date'], p_meta['folder'], p_meta['t'])]
ref_xy = get_xy(p_meta)
save_csv(p_meta, os.path.join(list_out_root, '{}_ref.csv'.format(place)))

ax = axs[0, 2]
ax.plot(ref_xy[:, 0], ref_xy[:, 1], label='{} overcast reference images'.format(len(ref_xy)), c='k')

# Query sequences
sets = {'overcast': '2015-08-14-14-54-57', 'sunny': '2014-11-18-13-20-12',
        'night': '2014-12-17-18-18-43', 'snow': '2015-02-03-08-45-10'}

full_meta = load_csv(os.path.join(fs_root(), 'data/learnlarge/clean_merged_parametrized/test.csv'))
for name in sets.keys():
    date = sets[name]
    selected_meta = dict()

    for key in full_meta.keys():
        selected_meta[key] = [e for e, d in zip(full_meta[key], full_meta['date']) if d == date]

    selected_meta['path'] = [img_path((d, f, t)) for d, f, t in
                             zip(selected_meta['date'], selected_meta['folder'], selected_meta['t'])]

    query_xy = get_xy(selected_meta)

    ax.plot(query_xy[:, 0], query_xy[:, 1], label='{} {} query images'.format(len(query_xy), name), linewidth=0.8)
    save_csv(selected_meta, os.path.join(list_out_root, '{}_{}.csv'.format(place, name)))

ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.title.set_text('Oxford RobotCar test images')
ax.legend(markerscale=5, loc='upper right')

# ------------------------------------- Cold -------------------------------------
ref_file = 'datasets/cold/freiburg/sq3_cloudy1/std_cam'
queries = {'cloudy': 'datasets/cold/freiburg/seq3_cloudy2/std_cam',
           'sunny': 'datasets/cold/freiburg/seq3_sunny1/std_cam'}
place = 'freiburg'


def get_t_x_y_a_from_path(path):
    name = os.path.basename(path)

    t = re.findall(r'(?<=t)-?\d+[.]?\d*', name)
    x = re.findall(r'(?<=x)-?\d+[.]?\d*', name)
    y = re.findall(r'(?<=y)-?\d+[.]?\d*', name)
    a = re.findall(r'(?<=a)-?\d+[.]?\d*', name)

    if not (len(t) == 1 and len(x) == 1 and len(y) == 1 and len(a) == 1):
        return False, 0, 0, 0, 0
    else:
        return True, float(t[0]), float(x[0]), float(y[0]), float(a[0])


def parse_file_list(FILES):
    valid_files = []
    T = []
    X = []
    Y = []
    A = []
    for i in range(len(FILES)):
        is_valid, t, x, y, a = get_t_x_y_a_from_path(FILES[i])
        if is_valid:
            valid_files.append(FILES[i])
            T.append(t)
            X.append(x)
            Y.append(y)
            A.append(a)
    return {'path': valid_files, 'yaw': A, 'easting': X, 'northing': Y, 't': T}


ref_meta = parse_file_list(os.listdir(os.path.join(fs_root(), ref_file)))
ref_sort_i = np.argsort(ref_meta['t'])
for key in ref_meta.keys():
    ref_meta[key] = [ref_meta[key][i] for i in ref_sort_i]

# Subsample reference set
ref_xy = get_xy(ref_meta)
last_xy = ref_xy[0, :]
remaining_i = [0]
r = 0.0
for i in tqdm(range(len(ref_meta['t']))):
    if sum((ref_xy[i, :] - last_xy) ** 2) >= r ** 2:
        last_xy = ref_xy[i, :]
        remaining_i.append(i)

for key in ref_meta.keys():
    ref_meta[key] = [ref_meta[key][i] for i in remaining_i]
ref_xy = get_xy(ref_meta)

ref_meta['path'] = [os.path.join(ref_file, p) for p in ref_meta['path']]
save_csv(ref_meta, os.path.join(list_out_root, '{}_{}.csv'.format(place, 'ref')))

ax = axs[0, 0]
ax.plot(ref_xy[:, 0], ref_xy[:, 1], label='{} overcast reference images'.format(len(ref_xy)), c='k')
for name in queries:
    query_meta = parse_file_list(os.listdir(os.path.join(fs_root(), queries[name])))

    sort_i = np.argsort(query_meta['t'])
    for key in query_meta.keys():
        query_meta[key] = [query_meta[key][i] for i in sort_i]

    query_meta['path'] = [os.path.join(queries[name], p) for p in query_meta['path']]
    query_xy = get_xy(query_meta)
    ax.plot(query_xy[:, 0], query_xy[:, 1], label='{} {} query images'.format(len(query_xy), name), linewidth=0.8)
    save_csv(query_meta, os.path.join(list_out_root, '{}_{}.csv'.format(place, name)))

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.title.set_text('Cold Freiburg test images')
ax.legend(markerscale=5)

# ------------------------------------- Oxford Training -------------------------------------

data = load_pickle(os.path.join(fs_root(), 'data/beyond/queries/full-10-25/train_ref.pickle'))
small_x = [data[k]['x'] for k in data.keys()]
small_y = [data[k]['y'] for k in data.keys()]

ax = axs[0, 3]
full_meta = load_csv(os.path.join(fs_root(), 'data/learnlarge/clean_merged_parametrized/train_ref.csv'))
ref_xy = get_xy(full_meta)

s1 = ax.scatter(ref_xy[::1, 0], ref_xy[::1, 1], s=1, label='Large: {} training images'.format(len(ref_xy)))
s2 = ax.scatter(small_x[::1], small_y[::1], s=1, label='Small: {} training images'.format(len(small_x)))
s1.set_rasterized(True)
s2.set_rasterized(True)
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.legend(markerscale=5)
ax.title.set_text('Oxford RobotCar training images')

# ------------------------------------- Save -------------------------------------
out_name = os.path.join(out_root, 'lists.pdf')

left = 0.0  # the left side of the subplots of the figure
right = 1.0  # the right side of the subplots of the figure
bottom = 0.0  # the bottom of the subplots of the figure
top = 1.0  # the top of the subplots of the figure
wspace = 0.35  # the amount of width reserved for space between subplots,
# expressed as a fraction of the average axis width
hspace = 0.2  # the amount of height reserved for space between subplots,
# expressed as a fraction of the average axis height

# space = 0.2
plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)

plt.savefig(out_name, bbox_inches='tight',
            pad_inches=0)

plt.savefig(out_name.replace('.pdf', '.pgf'), bbox_inches='tight',
            pad_inches=0)
