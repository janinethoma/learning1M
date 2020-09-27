#!/path/to/my/bin/python -u
import argparse
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

from learnlarge.util.helper import fs_root, mkdir
from learnlarge.util.io import load_pickle, save_csv


def log(output):
    print(output)
    LOG.write('{}\n'.format(output))
    LOG.flush()


def compile_table(l, d):
    mkdir(OUT_ROOT)
    top_n_root = os.path.join(fs_root(), 'top_n')

    queries = [
        'oxford_night',
        'oxford_overcast',
        'oxford_snow',
        'oxford_sunny',
        'freiburg_cloudy',
        'freiburg_sunny',
        'pittsburgh_query'
    ]

    checkpoints = [
        # Trained on cold
        'triplet_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0',
        'quadruplet_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0',

        'lazy_triplet_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0',
        'lazy_quadruplet_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0',

        'sum_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0',
        'h_sum_5e-6_all_conditions_angle_1-4_cu_LRD0.9-5_noPCA_lam0.5_me0',

        # Trained on small oxford
        'triplet_5e-6_full-10-25_cu_LRD0.9-5_noPCA_lam0.5_me0',
        'quadruplet_5e-6_full-10-25_cu_LRD0.9-5_noPCA_lam0.5_me0',

        'lazy_triplet_5e-6_full-10-25_cu_LRD0.9-5_noPCA_lam0.5_me0',
        'lazy_quadruplet_5e-6_full-10-25_cu_LRD0.9-5_noPCA_lam0.5_me0',

        'h_sum_5e-6_full-10-25_cu_LRD0.9-5_noPCA_lam0.5_me0',

        # Trained on large oxford
        'ha0_lotriplet_vl64'
        'ha0_loquadruplet_vl64'
        'ha0_lolazy_triplet_vl64'
        'ha0_lolazy_quadruplet_vl64'
        'ha0_lodistance_triplet_vl64'
        'ha0_lohuber_distance_triplet_vl64'
        'ha6_loevil_triplet_muTrue_vl64'
        'ha6_loevil_quadruplet_muTrue_vl64'
        'ha6_loresidual_det_muTrue_vl64'
        'ha0_lotriplet_vl0'
        'ha0_loquadruplet_vl0'
        'ha6_loevil_quadruplet_muTrue_vl0'
        'ha6_loresidual_det_muTrue_vl0'
        'ha0_lotriplet_muTrue_vl64'
        'ha0_lotriplet_muFalse_vl64'
        'ha6_lotriplet_muTrue_vl64'
        'ha6_lotriplet_muFalse_vl64'

        # Treined on Pittsburgh
        'pittsnetvlad',

        # Image-net
        'offtheshelf'
    ]

    losses = [
        'GT',

        'Triplet \cite{arandjelovic2016netvlad}',
        'Quadruplet \cite{chen2017beyond}',

        'Lazy triplet \cite{angelina2018pointnetvlad}',
        'Lazy quadruplet \cite{angelina2018pointnetvlad}',

        'Triplet + distance \cite{thoma2020geometrically}',
        'Triplet + Huber dist.~\cite{thoma2020geometrically}',

        'Triplet \cite{arandjelovic2016netvlad}',
        'Quadruplet \cite{chen2017beyond}',

        'Lazy triplet \cite{angelina2018pointnetvlad}',
        'Lazy quadruplet \cite{angelina2018pointnetvlad}',

        'Triplet + Huber dist.~\cite{thoma2020geometrically}',

        'Triplet \cite{arandjelovic2016netvlad}',
        'Quadruplet \cite{chen2017beyond}',

        'Lazy triplet \cite{angelina2018pointnetvlad}',
        'Lazy quadruplet \cite{angelina2018pointnetvlad}',

        'Triplet + distance \cite{thoma2020geometrically}',
        'Triplet + Huber dist.~\cite{thoma2020geometrically}',

        '\\textit{Triplet + HP}',
        '\\textit{Quadruplet + HP}',

        '\\textit{Volume}',
        '$\\mathit{Volume}^*$',

        'Triplet \cite{arandjelovic2016netvlad}',

        'Off-the-shelf \cite{deng2009imagenet}'
    ]

    setting = 'l{}_dim{}'.format(l, d)
    print(setting)

    table = defaultdict(list)

    table['Loss'] = losses

    for_mean = defaultdict(list)
    for i, query in enumerate(queries):
        print(query)

        print_gt = True

        if query.startswith('freiburg'):
            T = [0.5, 1.0, 1.5]
        else:
            T = [5.0, 10.0, 15.0]

        for j, checkpoint in enumerate(checkpoints):

            cp_name = checkpoint

            t_n_file = os.path.join(top_n_root, setting, '{}_{}.pickle'.format(query, cp_name))
            if not os.path.exists(t_n_file):
                print('Missing: {}'.format(t_n_file))
                table[query].append('-')
                for_mean[query].append([-1, -1, -1])
                continue
            print(t_n_file)

            [top_i, top_g_dists, top_f_dists, gt_i, gt_g_dist, ref_idx] = load_pickle(t_n_file)
            top_g_dists = np.array(top_g_dists)

            if print_gt:
                print_gt = False
                Y = [float(sum(gt_g_dist < x)) / float(len(gt_g_dist)) * 100 for x in T]
                table[query].append(['{:.1f}'.format(y) for y in Y])
                for_mean[query].append(Y)

            t_1_d = np.array([td[0] for td in top_g_dists])

            Y = [float(sum(t_1_d < x)) / float(len(t_1_d)) * 100 for x in T]
            table[query].append(['{:.1f}'.format(y) for y in Y])
            for_mean[query].append(Y)

        # Highlight best values:
        b = np.argmax(np.array(for_mean[query])[1:], axis=0)
        b = b + 1
        for ii, ib in enumerate(b):
            table[query][ib][ii] = '\\textbf{' + table[query][ib][ii] + '}'

        for ii in range(len(losses)):
            table[query][ii] = '/'.join(table[query][ii])

    for i in range(len(losses)):
        all = np.array([for_mean[query][i] for query in queries if for_mean[query][i][0] > -1])
        Y = np.mean(all, axis=0)
        table['mean'].append(['{:.1f}'.format(y) for y in Y])
        for_mean['mean'].append(Y)

    # Highlight best values:
    b = np.argmax(np.array(for_mean['mean'])[1:], axis=0)
    b = b + 1
    for ii, ib in enumerate(b):
        table['mean'][ib][ii] = '\\textbf{' + table['mean'][ib][ii] + '}'

    for ii in range(len(losses)):
        table['mean'][ii] = '/'.join(table['mean'][ii])

    out_name = os.path.join(OUT_ROOT, 'accuracy_table.csv')
    save_csv(table, out_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Image folder
    parser.add_argument('--l',
                        default='0.0', type=float)
    parser.add_argument('--d',
                        default='256', type=int)
    parser.add_argument('--log_dir', default=os.path.join(fs_root(),  'logs', 'roc'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(),  'plots'))

    FLAGS = parser.parse_args()

    LOG_DIR = FLAGS.log_dir
    OUT_ROOT = FLAGS.out_root

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(OUT_ROOT):
        os.makedirs(OUT_ROOT)

    LOG = open(os.path.join(LOG_DIR, 'top_n_log.txt'), 'a')
    log('Running {} at {}.'.format(__file__, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    log(FLAGS)

    compile_table(FLAGS.l, FLAGS.d)

    LOG.close()
