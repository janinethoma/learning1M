import argparse
import os

from learnlarge.util.helper import fs_root
from learnlarge.util.sge import run_one_job

parser = argparse.ArgumentParser()
parser.add_argument('--i', default=0, type=int)
args = parser.parse_args()

script = os.path.join(fs_root(), 'code/learnlarge/evaluate/003_any_top-n.py')

queries = [
    'freiburg_cloudy',
    'freiburg_sunny',
    'oxford_night',
    'oxford_overcast',
    'oxford_snow',
    'oxford_sunny',
    'pittsburgh_query'
]

references = {
    'oxford': 'oxford_ref',
    'freiburg': 'freiburg_ref',
    'pittsburgh': 'pittsburgh_ref'

}

pcas = {
    'oxford': 'oxford_pca',
    'freiburg': 'freiburg_pca',
    'pittsburgh': 'pittsburgh_ref'

}

csv_root = os.path.join(fs_root(), 'lists')
log_root = os.path.join(fs_root(), 'logs/top_n')
lv_root = os.path.join(fs_root(), 'lv')
out_root = os.path.join(fs_root(), 'top_n')

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

for cp in checkpoints:
    for query in queries:

        lv = '{}_{}.pickle'.format(query, cp)
        s = lv.split('_')

        # Check if already complete
        L = [0.0, 0.3, 1.0, 5.0]
        D = [64, 128, 256, 512, 1024]

        ref = references[s[0]]
        pca = pcas[s[0]]

        query_lv_pickle = os.path.join(lv_root, lv)
        ref_lv_pickle = os.path.join(lv_root, lv.replace(query, ref))
        pca_lv_pickle = os.path.join(lv_root, lv.replace(query, pca))

        if not os.path.exists(query_lv_pickle):
            print('Missing {}'.format(query_lv_pickle))

        if not os.path.exists(ref_lv_pickle):
            print('Missing {}'.format(ref_lv_pickle))

        if not os.path.exists(pca_lv_pickle):
            print('Missing {}'.format(pca_lv_pickle))

        complete = True
        for l in L:
            for d in D:

                out_folder = os.path.join(out_root, 'l{}_dim{}'.format(l, d))
                name = ''.join(os.path.basename(query_lv_pickle).split('.')[:-1])
                out_pickle = os.path.join(out_folder, '{}.pickle'.format(name))

                if not os.path.exists(out_pickle):
                    complete = False
                    break
            if not complete:
                break

        if complete:
            print('Skipping complete {}'.format(query_lv_pickle))
            continue

        parameters = list()
        parameters.append(('out_root', out_root))
        parameters.append(('query_lv_pickle', query_lv_pickle))
        parameters.append(('ref_lv_pickle', ref_lv_pickle))
        parameters.append(('pca_lv_pickle', pca_lv_pickle))
        parameters.append(('query_csv', os.path.join(csv_root, '{}.csv'.format(query))))
        parameters.append(('ref_csv', os.path.join(csv_root, '{}.csv'.format(ref))))

        name = lv.replace('.pickle', '')

        k = 0
        out_folder = '{}_{:03}'.format(name, k)
        log_dir = os.path.join(log_root, out_folder)
        while os.path.exists(log_dir):
            out_folder = '{}_{:03}'.format(name, k)
            log_dir = os.path.join(log_root, out_folder)
            k = k + 1

        queue = 'middle'
        memory = 90

        if 'pittsburgh' == s[0]:
            queue = 'long'

        if 'freiburg' == s[0]:
            memory = 38

        if 'oxford' == s[0]:
            memory = 49

        parameters.append(('log_dir', log_dir))

        run_one_job(script=script, queue=queue, cpu_only=True, memory=memory,
                    script_parameters=parameters, out_dir=log_dir,
                    name='imp_{}_{:03}'.format(name, k), overwrite=True, hold_off=False,
                    array=False)
