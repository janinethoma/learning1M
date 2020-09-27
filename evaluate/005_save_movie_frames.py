import os

import cv2
import numpy as np

from learnlarge.util.helper import mkdir, fs_root
from learnlarge.util.io import load_txt

base_path = os.path.join(fs_root(), 'grad_cam')
out_path = os.path.join(fs_root(), 'video_frames')

mkdir(out_path)

checkpoints = os.listdir(os.path.join(fs_root(), 'checkpoints'))

cp_keys = list()

for cp in checkpoints:
    c_name = cp.split('/')[-2]
    c_name = ''.join(os.path.basename(c_name).split('.'))  # Removing '.'
    c_name += '_e{}'.format(cp[-1])
    cp_keys.append(c_name)

print(cp_keys)

names = {
    'ha6_loresidual_det_muTrue_vl64': 'III',
    'offtheshelf': 'V',
    'pittsnetvlad': 'IV',
    'quadruplet_5e-6_all_conditions_angle_1-4_cu_LRD09-5_noPCA_lam05_me0_e1': 'I',
    'triplet_5e-6_full-10-25_cu_LRD09-5_noPCA_lam05_me0_e3': 'II'
}

sets = [
    'oxford_night',
    'oxford_overcast',
    'oxford_snow',
    'oxford_sunny',
    'pittsburgh_query',
    'freiburg_cloudy',
    'freiburg_sunny'
]

font = cv2.FONT_HERSHEY_TRIPLEX
font_scale = 0.7
font_thickness = 1


def get_placement(shape, font=font, font_scale=font_scale, font_thickness=font_thickness, text='test'):
    textsize = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    textX = (shape[1] - textsize[0]) // 2
    textY = (shape[0] + textsize[1]) // 2
    return (textX, textY)


text_shape, _ = cv2.getTextSize('Test your font.', font, font_scale, font_thickness)

text_margin = 10
header_height = text_margin + text_shape[1] + text_margin
video_name = os.path.join(out_path, 'grad_cam.avi')
width = 4 * 240 + 2 * header_height
height = 3 * 180 + header_height

img_count = 0
for s in sets:

    image_folder = dict()
    for c_name in cp_keys:
        image_folder[c_name] = os.path.join(base_path, '{}_{}'.format(s, c_name))

    images = sorted([img for img in os.listdir(image_folder[cp_keys]) if img.endswith(".png")])
    errors = sorted([img for img in os.listdir(image_folder[cp_keys]) if img.endswith(".txt")])

    num_frames = len(images)

    for f in range(0, num_frames, 10):
        drop = False
        frame_images = []
        for c_name in cp_keys:
            c_img = cv2.imread(os.path.join(image_folder[c_name], images[f]))
            c_error = load_txt(os.path.join(image_folder[c_name], errors[f]))

            if 'freiburg' in s:
                t = 1.0
            else:
                t = 10.0

            if float(c_error) <= t:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            offset = get_placement((header_height, 180), text=names[c_name])
            header = np.ones([header_height, 180, 3]) * 255
            cv2.putText(header, names[c_name], offset, font, font_scale, color, font_thickness, cv2.LINE_AA)
            header = np.rot90(header, 3, axes=[1, 0])

            offset = get_placement((header_height, 180), text='{}m'.format(c_error))
            error = np.ones([header_height, 180, 3]) * 255
            cv2.putText(error, '{}m'.format(c_error), offset, font, font_scale, color, font_thickness, cv2.LINE_AA)
            error = np.rot90(error, 3, axes=[1, 0])

            frame_images.append(np.concatenate([header, c_img, error], axis=1))

        if drop:
            continue

        header = np.ones([header_height, width, 3]) * 255

        color = (0, 0, 0)
        offset = get_placement((header_height, 480), text='Query')
        cv2.putText(header, 'Query', (offset[0] + header_height, offset[1]), font, font_scale, color, font_thickness,
                    cv2.LINE_AA)
        offset = get_placement((header_height, 480), text='Retrieved')
        cv2.putText(header, 'Retrieved', (offset[0] + header_height + 480, offset[1]), font, font_scale, color,
                    font_thickness, cv2.LINE_AA)
        frame_images.insert(0, header)

        out_file = os.path.join(out_path, '{}_{:05}.png'.format(s, f))
        frame = np.asarray(np.concatenate(frame_images, axis=0), dtype=np.uint8)
        cv2.imwrite(out_file, frame)
