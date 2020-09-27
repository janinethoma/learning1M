import argparse
import os

from learnlarge.util.helper import fs_root
from learnlarge.util.sge import run_one_job

parser = argparse.ArgumentParser()
parser.add_argument('--i', default=0, type=int)
args = parser.parse_args()

script = os.path.join(fs_root(), 'code/learnlarge/evaluate/012_any_tsne.py')

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
log_root = os.path.join(fs_root(), 'logs/tsne')
lv_root = os.path.join(fs_root(), 'lv')
lv_list = os.listdir(lv_root)

pcas = {
    'oxford': 'oxford_pca',
    'freiburg': 'freiburg_pca',
    'pittsburgh': 'pittsburgh_ref'

}

for query in queries:
    for cp in checkpoints:

        pca = pcas[query.split('_')[0]]

        parameters = list()
        parameters.append(('query_lv_pickle', os.path.join(lv_root, '{}_{}.pickle'.format(query, cp))))
        parameters.append(('query_csv', os.path.join(csv_root, '{}.csv'.format(query))))

        parameters.append(('pca_lv_pickle', os.path.join(lv_root, '{}_{}.pickle'.format(pca, cp))))
        parameters.append(('pca_csv', os.path.join(csv_root, '{}.csv'.format(pca))))

        k = 0
        out_folder = '{}_{}_{:03}'.format(cp, query, k)
        log_dir = os.path.join(log_root, out_folder)
        while os.path.exists(log_dir):
            k = k + 1
            out_folder = '{}_{}_{:03}'.format(cp, query, k)
            log_dir = os.path.join(log_root, out_folder)

        parameters.append(('log_dir', log_dir))
        run_one_job(script=script, queue='short', cpu_only=True, memory=30,
                    script_parameters=parameters, out_dir=log_dir,
                    name='tsne_{}_{}'.format(cp, query), overwrite=True, hold_off=False, array=False)
