import argparse
import os

from learnlarge.util.helper import fs_root
from learnlarge.util.sge import run_one_job

parser = argparse.ArgumentParser()
parser.add_argument('--i', default=0, type=int)
args = parser.parse_args()

script = os.path.join(fs_root(), 'code/learnlarge/evaluate/011_any_grad_cam.py')


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

setting = 'l0.0_dim256'

csv_root = os.path.join(fs_root(), 'lists')
log_root = os.path.join(fs_root(), 'grad_cam')
top_n_root = os.path.join(fs_root(), 'top_n')
lv_root = os.path.join(fs_root(), 'lv')

for query in queries:
    for cp in checkpoints:

        ref = references[query.split('_')[0]]

        t_n_file = os.path.join(top_n_root, setting, '{}_{}.pickle'.format(query, cp))

        if os.path.exists(os.path.join(fs_root(), 'grad_cam', os.path.splitext(os.path.basename(t_n_file))[0])):
            print('Skipping existing: {}'.format(top_n_root))
            continue

        parameters = list()
        parameters.append(
            ('top_n_pickle', t_n_file))
        parameters.append(('query_csv', os.path.join(csv_root, '{}.csv'.format(query))))
        parameters.append(('ref_csv', os.path.join(csv_root, '{}.csv'.format(ref))))
        parameters.append(('checkpoint', os.path.join(fs_root(), 'checkpoints', cp)))

        if 'vl64' not in cp:
            parameters.append(('vlad_cores', 0))

        k = 0
        out_folder = '{}_{}_{:03}'.format(cp, query, k)
        log_dir = os.path.join(log_root, out_folder)
        while os.path.exists(log_dir):
            k = k + 1
            out_folder = '{}_{}_{:03}'.format(cp, query, k)
            log_dir = os.path.join(log_root, out_folder)

        parameters.append(('log_dir', log_dir))
        run_one_job(script=script, queue='2h', cpu_only=False, memory=30,
                    script_parameters=parameters, out_dir=log_dir,
                    name='grad_{}_{}'.format(cp, query), overwrite=True, hold_off=False, array=False)
