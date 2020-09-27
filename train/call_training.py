import argparse
import os

from learnlarge.util.helper import fs_root
from learnlarge.util.sge import run_one_job

parser = argparse.ArgumentParser()
parser.add_argument('--i', default=0, type=int)
args = parser.parse_args()

SERIES = '1M'

train_script = os.path.join(fs_root(), 'code/learnlarge/train/train.py')
out_root = os.path.join(fs_root(), 'logs/learnlarge')

# Number of jobs per queue
middle = 4
long = 2

settings = list()

# SOTA Baselines
settings.append({'loss': 'triplet', 'vlad_cores': 64, 'hard_positives_per_tuple': 0})  # Triplet
settings.append({'loss': 'quadruplet', 'vlad_cores': 64, 'hard_positives_per_tuple': 0})  # Quadruplet
settings.append({'loss': 'lazy_triplet', 'vlad_cores': 64, 'hard_positives_per_tuple': 0})  # Lazy triplet
settings.append({'loss': 'lazy_quadruplet', 'vlad_cores': 64, 'hard_positives_per_tuple': 0})  # Lazy quadruplet
settings.append({'loss': 'distance_triplet', 'vlad_cores': 64, 'hard_positives_per_tuple': 0})  # Triplet + distance
settings.append(
    {'loss': 'huber_distance_triplet', 'vlad_cores': 64, 'hard_positives_per_tuple': 0})  # Triplet + Huber dist.

# Our losses:
settings.append({'loss': 'evil_triplet', 'vlad_cores': 64, 'hard_positives_per_tuple': 6,
                 'mutually_exclusive_negs': True})  # Triplet + HP
settings.append({'loss': 'evil_quadruplet', 'vlad_cores': 64, 'hard_positives_per_tuple': 6,
                 'mutually_exclusive_negs': True})  # Quadruplet + HP
settings.append({'loss': 'residual_det', 'vlad_cores': 64, 'hard_positives_per_tuple': 6,
                 'mutually_exclusive_negs': True})  # Volume

# Spacial pooling ablation
settings.append({'loss': 'triplet', 'vlad_cores': 0, 'hard_positives_per_tuple': 0})  # Triplet*
settings.append({'loss': 'quadruplet', 'vlad_cores': 0, 'hard_positives_per_tuple': 0})  # Quadruplet*
settings.append({'loss': 'evil_quadruplet', 'vlad_cores': 0, 'hard_positives_per_tuple': 6,
                 'mutually_exclusive_negs': True})  # Quadruplet* + HP
settings.append({'loss': 'residual_det', 'vlad_cores': 0, 'hard_positives_per_tuple': 6,
                 'mutually_exclusive_negs': True})  # Volume*

# Mining ablation
settings.append({'loss': 'triplet', 'vlad_cores': 64, 'hard_positives_per_tuple': 0, 'mutually_exclusive_negs': True})
settings.append({'loss': 'triplet', 'vlad_cores': 64, 'hard_positives_per_tuple': 0, 'mutually_exclusive_negs': False})
settings.append({'loss': 'triplet', 'vlad_cores': 64, 'hard_positives_per_tuple': 6, 'mutually_exclusive_negs': True})
settings.append({'loss': 'triplet', 'vlad_cores': 64, 'hard_positives_per_tuple': 6, 'mutually_exclusive_negs': False})

all_commands = []
for i, setting in enumerate(settings):
    k = 0
    name = '_'.join(x[0] + x[min(1, len(x) - 1)] + str(setting[x]) for x in sorted(setting))

    out_folder = '{}_{}_{:03}'.format(name, SERIES, k)
    out_dir = os.path.join(out_root, out_folder)
    while os.path.exists(out_dir):
        out_folder = '{}_{}_{:03}'.format(name, SERIES, k)
        out_dir = os.path.join(out_root, out_folder)
        k = k + 1

    parameters = [('out_root', os.path.dirname(out_dir)), ('out_folder', os.path.basename(out_dir))]
    for key in setting.keys():
        parameters.append((key, setting[key]))

    os.makedirs(out_dir)
    if i % (middle + long) < long:
        q = '5d'
    else:
        q = '48h'

    run_one_job(script=train_script, queue=q, cpu_only=False, memory=50,
                script_parameters=parameters, out_dir=out_dir,
                name='{}_{}'.format(name, SERIES), overwrite=True, hold_off=False, array=True, num_jobs=1)
