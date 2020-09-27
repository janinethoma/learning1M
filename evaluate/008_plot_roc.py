#!/path/to/my/bin/python -u

# ROC plot for Figure 3 in the supplementary material. The other roc-plots are produced analogously and therefore not
# included in this folder.

import argparse
import os
from datetime import datetime
from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from learnlarge.util.helper import fs_root, mkdir, flags_to_globals
from learnlarge.util.io import load_pickle

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


def plot_roc(l, d):
    mkdir(OUT_ROOT)
    top_n_root = os.path.join(fs_root(), 'top_n')

    checkpoints = \
        [
            # Trained on cold
            'triplet_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0',

            # Trained on small oxford
            'triplet_5e-6_full-10-25_cu_LRD0.9-5_noPCA_lam0.5_me0',

            # Trained on large oxford
            'ha0_lotriplet_vl64'

            # Treined on Pittsburgh
            'pittsnetvlad',

            # Image-net
            'offtheshelf'

            # III with hard postitives
            'ha6_loevil_triplet_muTrue_vl64'
        ]

    queries = [
        'oxford_night',
        'freiburg_cloudy',
        'oxford_overcast',
        'freiburg_sunny',
        'oxford_snow',
        'pittsburgh_query',
        'oxford_sunny',

    ]

    titles = [
        'Oxford RobotCar, night',
        'Cold Freiburg, cloudy',
        'Oxford RobotCar, overcast',
        'Cold Freiburg, sunny',
        'Oxford RobotCar, snow',
        'Pittsburgh',
        'Oxford RobotCar, sunny',
    ]

    losses = [

        'I Cold Freiburg',
        'II Oxford (small)',
        'III Oxford (large)',
        'IV Pittsburgh',
        'V ImageNet (off-the-shelf)',

        '\\textit{III Oxford (large) + HP}',
    ]

    fill_styles = [
        'none',
        'none',
        'none',
        'full',
        'none',

        'full',

    ]

    markers = [
        '|',
        '.',
        'o',
        '*',
        '',

        'o',
    ]

    lines = [
        '--',

        '-',
        '-',
        '-.',
        ':',

        '-'
    ]

    colors = [
        '#1cad62',

        '#00BFFF',
        '#1E90FF',  # Triplet
        '#8c0054',
        '#000000',

        '#1934e6',  # Triplet HP
    ]

    rows = 2
    cols = 4

    f, axs = plt.subplots(rows, cols, constrained_layout=False)
    if rows == 1:
        axs = np.expand_dims(axs, 0)
    if cols == 1:
        axs = np.expand_dims(axs, 1)
    f.tight_layout()
    f.set_figheight(4.5)  # 8.875in textheight
    f.set_figwidth(8.5)  # 6.875in textwidth

    for i, query in enumerate(queries):
        print(query)

        print_gt = True

        if query.startswith('freiburg'):
            t = 1.5
        else:
            t = 15.0

        setting = 'l{}_dim{}'.format(l, d)

        min_y = 1000
        max_y = 0

        for j, (cp_name, m, line, color) in enumerate(
                zip(checkpoints, cycle(markers), cycle(lines), cycle(colors))):

            t_n_file = os.path.join(top_n_root, setting, '{}_{}.pickle'.format(query, cp_name))
            if not os.path.exists(t_n_file):
                print('Missing: {}'.format(t_n_file))
                continue
            print(t_n_file)

            [top_i, top_g_dists, top_f_dists, gt_i, gt_g_dist, ref_idx] = load_pickle(t_n_file)
            top_g_dists = np.array(top_g_dists)

            if print_gt:
                print_gt = False
                X = np.linspace(0, t, num=50)
                Y = [float(sum(gt_g_dist < x)) / float(len(gt_g_dist)) * 100 for x in X]
                ax = axs[i % rows, i // rows]
                width = 0.75

                ax.plot(X, Y, label='Upper bound', linewidth=width, c='#000000')
                ax.plot([0], [0], linewidth=0, label=' ')
                ax.plot([0], [0], linewidth=0, label='\\textbf{Training datasets:}')
                ax.title.set_text(titles[i])
                ax.set_xlim([0, t])
                ax.grid(True)

            if 'ha6_loevil_triplet_muTrue_vl64' in cp_name:
                ax = axs[i % rows, i // rows]
                ax.plot([0], [0], linewidth=0, label=' ')
                ax.plot([0], [0], linewidth=0, label='\\textbf{With our mining:}')

            t_1_d = np.array([td[0] for td in top_g_dists])
            X = np.linspace(0, t, num=50)

            Y = [float(sum(t_1_d < x)) / float(len(t_1_d)) * 100 for x in X]

            min_y = min(np.min(np.array(Y)), min_y)
            max_y = max(np.max(np.array(Y)), max_y)

            ax = axs[i % rows, i // rows]
            width = 0.75
            ax.plot(X, Y, label=losses[j], linestyle=line, marker=m, linewidth=width, markevery=j % rows + cols,
                    c=color, markersize=3, fillstyle=fill_styles[j])

        ax = axs[i % rows, i // rows]
        ax.set_xlim([0, t])
        ax.set_ylim([min_y, min(max_y + 5, 99)])

        # Major ticks every 20, minor ticks every 5
        major_ticks_x = np.arange(0, t, t / 3)
        minor_ticks_x = np.arange(0, t, t / 3 / 4)

        y_step = 20
        if 'night' in query:
            y_step /= 2

        major_ticks_y = np.arange(min_y, min(max_y + 5, 99), y_step)
        minor_ticks_y = np.arange(min_y, min(max_y + 5, 99), 5)

        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)

        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

    out_name = os.path.join(OUT_ROOT, '{}_training_region_roc.pdf'.format(setting))

    axs[-1, -1].axis('off')

    for i in range(cols):
        axs[-1, i].set_xlabel('Distance threshold $d$ [m]')

    for i in range(rows):
        axs[i, 0].set_ylabel('Correctly localized [\%]')

    left = 0.0  # the left side of the subplots of the figure
    right = 1.0  # the right side of the subplots of the figure
    bottom = 0.0  # the bottom of the subplots of the figure
    top = 1.0  # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for space between subplots,
    # expressed as a fraction of the average axis width
    hspace = 0.25  # the amount of height reserved for space between subplots,
    # expressed as a fraction of the average axis height

    # space = 0.2
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)

    handles, labels = axs[0, 0].get_legend_handles_labels()

    axs[-1, -1].legend(handles, labels, loc='lower left', bbox_to_anchor=(-0.075, 0.0), ncol=1, frameon=True,
                       borderaxespad=0.0)

    plt.savefig(out_name, bbox_inches='tight', pad_inches=0)
    plt.savefig(out_name.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Image folder
    parser.add_argument('--l',
                        default=0.0, type=float)
    parser.add_argument('--d',
                        default='256', type=int)
    parser.add_argument('--checkpoints',
                        default='residual')
    parser.add_argument('--log_dir', default=os.path.join(fs_root(),  'logs', 'roc'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(),  'plots'))

    FLAGS = parser.parse_args()

    # Define each FLAG as a variable (generated automatically with util.flags_to_globals(FLAGS))
    flags_to_globals(FLAGS)

    LOG_DIR = FLAGS.log_dir
    OUT_ROOT = FLAGS.out_root

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(OUT_ROOT):
        os.makedirs(OUT_ROOT)

    LOG = open(os.path.join(LOG_DIR, 'top_n_log.txt'), 'a')
    log('Running {} at {}.'.format(__file__, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    log(FLAGS)

    plot_roc(FLAGS.l, FLAGS.d)

    LOG.close()
