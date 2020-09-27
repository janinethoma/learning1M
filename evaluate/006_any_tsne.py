#!/path/to/my/bin/python -u

# Call this script using 006_call_tsne.py

import argparse
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from learnlarge.util.helper import flags_to_globals
from learnlarge.util.helper import fs_root
from learnlarge.util.io import load_csv
from learnlarge.util.io import load_pickle
from learnlarge.util.meta import get_xy

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def log(output):
    print(output)
    LOG.write('{}\n'.format(output))
    LOG.flush()


def get_top_n():
    name = os.path.basename(QUERY_LV_PICKLE).split('.')[0]
    print(name)
    sampling = 1

    out_png_1 = os.path.join(OUT_ROOT, '{}_top{}_t{}_path_{}_s{}.pdf'.format(name, N, T, PERPLEXITY, sampling))
    out_png_1c = os.path.join(OUT_ROOT, '{}_top{}_t{}_ct_{}_s{}.pdf'.format(name, N, T, PERPLEXITY, sampling))

    out_pickle = os.path.join(OUT_ROOT, '{}_top{}_t{}_{}_s{}.pickle'.format(name, N, T, PERPLEXITY, sampling))
    if os.path.exists(out_pickle):
        print('{} already exists. Skipping.'.format(out_pickle))
        return

    pca_f = np.array(load_pickle(PCA_LV_PICKLE))

    pca = PCA(whiten=True, n_components=256)
    pca = pca.fit(pca_f)

    query_meta = load_csv(QUERY_CSV)

    query_xy = get_xy(query_meta)[::sampling]

    l_query_f = np.array(load_pickle(QUERY_LV_PICKLE))
    l_query_f = l_query_f[::sampling, :]

    query_f = pca.transform(l_query_f)

    Y = TSNE(n_components=2, perplexity=PERPLEXITY).fit_transform(query_f)

    Y[:, 0] = (Y[:, 0] - min(Y[:, 0])) / (max(Y[:, 0]) - min(Y[:, 0]))
    Y[:, 1] = (Y[:, 1] - min(Y[:, 1])) / (max(Y[:, 1]) - min(Y[:, 1]))

    plt.clf()
    plt.figure(figsize=(3, 3))
    x = [p[0] for p in query_xy]
    y = [p[1] for p in query_xy]

    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)
    x_span = float(x_max - x_min)
    y_span = float(y_max - y_min)

    query_color = [(0, float(p[1] - y_min) / y_span, float(p[0] - x_min) / x_span) for p in query_xy]

    s1 = plt.scatter(x, y, c=query_color, s=2)
    s1.set_rasterized(True)
    plt.savefig(out_png_1, bbox_inches='tight',
                pad_inches=0)

    plt.clf()
    plt.figure(figsize=(3, 3))
    s2 = plt.scatter(Y[:, 0], Y[:, 1], c=query_color, s=2)
    s2.set_rasterized(True)
    plt.savefig(out_png_1c, bbox_inches='tight',
                pad_inches=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Image folder
    parser.add_argument('--img_root', default=os.path.join(fs_root()))
    parser.add_argument('--query_lv_pickle')
    parser.add_argument('--query_csv')

    parser.add_argument('--pca_lv_pickle')
    parser.add_argument('--pca_csv')

    parser.add_argument('--T', default=25, type=float)
    parser.add_argument('--N', default=25, type=int)
    parser.add_argument('--p', default=100, type=int)

    # Output
    parser.add_argument('--out_root', default=os.path.join(fs_root(), 'tsne'))
    parser.add_argument('--log_dir',
                        default=os.path.join(fs_root(), 'logs/tsne'))

    FLAGS = parser.parse_args()

    # Define each FLAG as a variable (generated automatically with util.flags_to_globals(FLAGS))
    flags_to_globals(FLAGS)

    N = FLAGS.N
    T = FLAGS.T
    IMG_ROOT = FLAGS.img_root
    LOG_DIR = FLAGS.log_dir
    OUT_ROOT = FLAGS.out_root
    PERPLEXITY = FLAGS.p
    PCA_CSV = FLAGS.pca_csv
    PCA_LV_PICKLE = FLAGS.pca_lv_pickle
    QUERY_CSV = FLAGS.query_csv
    QUERY_LV_PICKLE = FLAGS.query_lv_pickle

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(OUT_ROOT):
        os.makedirs(OUT_ROOT)

    LOG = open(os.path.join(LOG_DIR, 'tsne_log.txt'), 'a')
    log('Running {} at {}.'.format(__file__, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    log(FLAGS)

    get_top_n()

    LOG.close()
