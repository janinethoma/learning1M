import os

import cv2
import numpy as np

from learnlarge.util.helper import fs_root
from learnlarge.util.io import load_txt

base_path = os.path.join(fs_root(), 'grad_cam')
out_path = os.path.join(fs_root(), 'video')

checkpoints = os.listdir(os.path.join(fs_root(), 'checkpoints'))

cp_keys = list()
for cp in checkpoints:
    c_name = cp.split('/')[-2]
    c_name = ''.join(os.path.basename(c_name).split('.'))  # Removing '.'
    c_name += '_e{}'.format(cp[-1])
    cp_keys.append(c_name)

print(cp_keys)

names = {
    'ha6_loresidual_det_muTrue_vl64': 'Ours',
    'offtheshelf': 'Off-the-shelf',
    'pittsnetvlad': 'Pittsburgh',
    'ha0_lotriplet_vl64': 'Triplet'
}

queries = [
    ('oxford_night', 'Oxford RobotCar, Night'),
    ('oxford_overcast', 'Oxford RobotCar, Overcast'),
    ('oxford_snow', 'Oxford RobotCar, Snow'),
    ('oxford_sunny', 'Oxford RobotCar, Sunny'),
    ('pittsburgh_query', 'Pittsburgh'),
    ('freiburg_cloudy', 'Cold Freiburg, Cloudy'),
    ('freiburg_sunny', 'Cold Freiburg, Sunny'),
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

video = cv2.VideoWriter(video_name, 0, 1.5, (width, height))
img_count = 0
for s in queries:
    query = s[0]

    # make title

    frame = np.zeros([height, width, 3])
    cv2.putText(frame, s[1],
                get_placement(frame.shape, text=s[1], font_scale=font_scale * 2, font_thickness=font_thickness * 2),
                font, font_scale * 2, (255, 255, 255), font_thickness * 2, cv2.LINE_AA)
    frame = np.asarray(frame, dtype=np.uint8)
    video.write(frame)
    video.write(frame)
    video.write(frame)
    video.write(frame)
    video.write(frame)

    image_folder = dict()
    for c_name in cp_keys:
        image_folder[c_name] = os.path.join(base_path, '{}_{}'.format(s[0], c_name))

    images = sorted([img for img in os.listdir(image_folder[cp_keys[0]]) if img.endswith(".png")])
    errors = sorted([img for img in os.listdir(image_folder[cp_keys[0]]) if img.endswith(".txt")])

    num_frames = len(images)

    for f in range(0, num_frames, 10):
        frame_images = []
        for c_name in cp_keys:
            c_img = cv2.imread(os.path.join(image_folder[c_name], images[f]))
            c_error = load_txt(os.path.join(image_folder[c_name], errors[f]))

            if 'freiburg' in s[0]:
                t = 1.0
            else:
                t = 10.0

            if float(c_error) <= t:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            offset = get_placement((header_height, 180), text=names[c_name])
            header = np.zeros([header_height, 180, 3])
            cv2.putText(header, names[c_name], offset, font, font_scale, color, font_thickness, cv2.LINE_AA)
            header = np.rot90(header, 3, axes=[1, 0])

            offset = get_placement((header_height, 180), text='{}m'.format(c_error))
            error = np.zeros([header_height, 180, 3])
            cv2.putText(error, '{}m'.format(c_error), offset, font, font_scale, color, font_thickness, cv2.LINE_AA)
            error = np.rot90(error, 3, axes=[1, 0])

            frame_images.append(np.concatenate([header, c_img, error], axis=1))

        header = np.zeros([header_height, width, 3])

        color = (255, 255, 255)
        offset = get_placement((header_height, 480), text='Query')
        cv2.putText(header, 'Query', (offset[0] + header_height, offset[1]), font, font_scale, color, font_thickness,
                    cv2.LINE_AA)
        offset = get_placement((header_height, 480), text='Retrieved')
        cv2.putText(header, 'Retrieved', (offset[0] + header_height + 480, offset[1]), font, font_scale, color,
                    font_thickness, cv2.LINE_AA)
        frame_images.insert(0, header)

        frame = np.asarray(np.concatenate(frame_images, axis=0), dtype=np.uint8)
        video.write(frame)
        img_count = img_count + 1

cv2.destroyAllWindows()
video.release()
print(img_count)
