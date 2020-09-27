import os

from learnlarge.util.helper import fs_root
from learnlarge.util.sge import run_one_job

script = os.path.join(fs_root(), 'code/learnlarge/evaluate/001_any_inference.py')

sets = [
    'oxford_ref',
    'oxford_night',
    'oxford_overcast',
    'oxford_pca',
    'oxford_snow',
    'oxford_sunny',
    'freiburg_pca',
    'freiburg_cloudy',
    'freiburg_ref',
    'freiburg_sunny',
    'pittsburgh_query',
    'pittsburgh_ref',
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

out_root = os.path.join(fs_root(), 'lv')
log_root = os.path.join(fs_root(), 'logs/lv')

common_parameters = [
    ('csv_root', os.path.join(fs_root(), 'lists')),
    ('out_root', out_root),
    ('img_root', fs_root()),
    ('task_id', '0')
]

for s in sets:
    for c_name in checkpoints:

        c_path = os.path.join(fs_root(), 'checkpoints', c_name)

        if os.path.exists(os.path.join(out_root, '{}_{}.pickle'.format(s, c_name))):
            print('{}_{}.pickle already exists. Skipping.'.format(s, c_name))
            continue

        par = common_parameters.copy()
        par.append(('checkpoint', c_path))
        par.append(('set', s))
        par.append(('out_name', c_name))

        if 'vl0' in c_name:
            par.append(('vlad_cores', 0))

        k = 0
        out_folder = '{}_{}_{}_{:03}'.format(s, c_name, k)
        log_dir = os.path.join(log_root, out_folder)
        while os.path.exists(log_dir):
            out_folder = '{}_{}_{}_{:03}'.format(s, c_name, k)
            log_dir = os.path.join(log_root, out_folder)
            k = k + 1

        par.append(('log_dir', log_dir))
        run_one_job(script=script, queue='24h', cpu_only=False, memory=50,
                    script_parameters=par, out_dir=log_dir,
                    name='{}_{}'.format(s, c_name), overwrite=True, hold_off=False, array=False)
