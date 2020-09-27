#!/path/to/my/bin/python -u

# Call this script using 004_call_grad_cam.py

import argparse
import os
import random
from datetime import datetime
from queue import Queue
from threading import Thread
from time import time

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import pairwise_distances

from learnlarge.model.grad_nets import vgg16Netvlad, vgg16
from learnlarge.util.cv import resize_img, standard_size
from learnlarge.util.helper import flags_to_globals
from learnlarge.util.helper import fs_root, mkdir
from learnlarge.util.io import load_img, load_csv, load_pickle, save_img, save_txt
from learnlarge.util.meta import get_xy

matplotlib.use('Agg')


def log(output):
    print(output)
    LOG.write('{}\n'.format(output))
    LOG.flush()


def cpu_thread():
    global CPU_IN_QUEUE
    global GPU_IN_QUEUE
    while True:
        t = time()
        index, image_info, dist = CPU_IN_QUEUE.get()
        images = load_images(image_info)
        GPU_IN_QUEUE.put((index, images, dist), block=True)
        CPU_IN_QUEUE.task_done()
        print('Loaded images in {}s.'.format(time() - t))


def gpu_thread(sess, ops):
    global GPU_IN_QUEUE
    global GPU_OUT_QUEUE
    while True:
        t = time()

        index, img, dist = GPU_IN_QUEUE.get()
        """
        calculate Grad-CAM // Modiefied from https://github.com/cydonia999/Grad-CAM-in-TensorFlow
        """
        grads = tf.gradients(ops['loss'], ops['grad_in'])[0]  # d loss / d conv
        output, grads_val = sess.run([ops['grad_in'], grads], feed_dict={ops['input']: img})
        weights = np.mean(grads_val, axis=(1, 2))  # average pooling
        weights = np.expand_dims(
            weights, axis=1)
        weights = np.expand_dims(
            weights, axis=1)
        cams = np.sum(weights * output, axis=3)
        GPU_OUT_QUEUE.put((index, img, cams, dist))
        GPU_IN_QUEUE.task_done()
        print('Inferred {} images in {}s.'.format(index, time() - t))


def add_text(img, text):
    return img


def save_thread():
    global GPU_OUT_QUEUE
    while True:
        index, img, cams, dist = GPU_OUT_QUEUE.get()

        query_img = img[0]
        ref_img = img[1]

        query_cam = cams[0]  # the first GRAD-CAM for the first image in  batch
        image = np.uint8(query_img[:, :, ::-1] * 255.0)  # RGB -> BGR
        query_cam = cv2.resize(query_cam, (query_img.shape[1], query_img.shape[0]))  # enlarge heatmap
        query_cam = np.maximum(query_cam, 0)
        heatmap = query_cam / np.max(query_cam)  # normalize
        query_cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)  # balck-and-white to color
        query_cam = np.float32(query_cam) + np.float32(image)  # everlay heatmap onto the image
        query_cam = 255 * query_cam / np.max(query_cam)
        query_cam = np.uint8(query_cam)
        query_cam = cv2.cvtColor(query_cam, cv2.COLOR_BGR2RGB)

        ref_cam = cams[1]  # the first GRAD-CAM for the first image in  batch
        image = np.uint8(ref_img[:, :, ::-1] * 255.0)  # RGB -> BGR
        ref_cam = cv2.resize(ref_cam, (ref_img.shape[1], ref_img.shape[0]))  # enlarge heatmap
        ref_cam = np.maximum(ref_cam, 0)
        heatmap = ref_cam / np.max(ref_cam)  # normalize
        ref_cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)  # balck-and-white to color
        ref_cam = np.float32(ref_cam) + np.float32(image)  # everlay heatmap onto the image
        ref_cam = 255 * ref_cam / np.max(ref_cam)
        ref_cam = np.uint8(ref_cam)
        ref_cam = cv2.cvtColor(ref_cam, cv2.COLOR_BGR2RGB)

        merged = np.concatenate([query_img, query_cam, ref_cam, ref_img], axis=1)

        cam_path = os.path.join(OUT_FOLDER, '{:05}.png'.format(index[0]))
        save_img(merged, cam_path)
        err_path = os.path.join(OUT_FOLDER, '{:05}.txt'.format(index[0]))
        save_txt('{:.2f}'.format(dist[0]), err_path)

        GPU_OUT_QUEUE.task_done()
        print('SAVED_grad_cam')


def load_images(img_path):
    images = [[]] * len(img_path)
    for i in range(len(img_path)):

        if VLAD_CORES > 0:
            if RESCALE:
                images[i] = resize_img(load_img(os.path.join(IMG_ROOT, img_path[i])), LARGE_SIDE)
            else:
                images[i] = load_img(os.path.join(IMG_ROOT, img_path[i]))
        else:
            images[i] = standard_size(load_img(os.path.join(IMG_ROOT, img_path[i])), h=SMALL_SIDE, w=LARGE_SIDE)
    return images


def build_inference_model():
    ops = dict()
    tuple_shape = [1, 1]
    if VLAD_CORES > 0:
        ops['input'] = tf.placeholder(dtype=tf.float32, shape=[TUPLES_PER_BATCH * sum(tuple_shape), None, None, 3])
        ops['output'] = vgg16Netvlad(ops['input'])
    else:
        ops['input'] = tf.placeholder(dtype=tf.float32,
                                      shape=[TUPLES_PER_BATCH * sum(tuple_shape), SMALL_SIDE, LARGE_SIDE, 3])
        ops['output'] = tf.layers.flatten(vgg16(ops['input']))
    ops['outputs'] = tf.split(tf.reshape(ops['output'], [TUPLES_PER_BATCH, sum(tuple_shape), -1]), tuple_shape, 1)
    ops['loss'] = tf.negative(tf.squared_difference(ops['outputs'][0], ops['outputs'][1]))
    return ops, tuple_shape


def restore_weights():
    to_restore = {}
    for var in tf.trainable_variables():
        print(var.name)
        if var.name != 'Variable:0':
            # name_here = var.name
            saved_name = var._shared_name
            to_restore[saved_name] = var
            print(var)
    restoration_saver = tf.train.Saver(to_restore)
    # Create a session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    config.gpu_options.polling_inactive_delay_msecs = 10
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    # Initialize a new model
    init = tf.global_variables_initializer()
    sess.run(init)
    restoration_saver.restore(sess, CHECKPOINT)
    return sess


def save_cam(cams, rank, class_id, class_name, prob, image_batch, input_image_path):
    """
    save Grad-CAM images // Modiefied from https://github.com/cydonia999/Grad-CAM-in-TensorFlow
    """
    cam = cams[0]  # the first GRAD-CAM for the first image in  batch
    image = np.uint8(image_batch[0][:, :, ::-1] * 255.0)  # RGB -> BGR
    cam = cv2.resize(cam, (224, 224))  # enlarge heatmap
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)  # normalize
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)  # balck-and-white to color
    cam = np.float32(cam) + np.float32(image)  # everlay heatmap onto the image
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)

    # create image file names
    base_path, ext = os.path.splitext(input_image_path)
    base_path_class = "{}_{}_{}_{}_{:.3f}".format(base_path, rank, class_id, class_name, prob)
    cam_path = "{}_{}{}".format(base_path_class, "gradcam", ext)
    heatmap_path = "{}_{}{}".format(base_path_class, "heatmap", ext)
    segmentation_path = "{}_{}{}".format(base_path_class, "segmented", ext)

    # write images
    cv2.imwrite(cam_path, cam)
    cv2.imwrite(heatmap_path, (heatmap * 255.0).astype(np.uint8))
    cv2.imwrite(segmentation_path, (heatmap[:, :, None].astype(float) * image).astype(np.uint8))


def get_grad_cam():
    with tf.Graph().as_default() as graph:
        print("In Graph")

        ops, tuple_shape = build_inference_model()
        sess = restore_weights()

        print('\n'.join([n.name for n in tf.all_variables()]))

        # For better gpu utilization, loading processes and gpu inference are done in separate threads.
        # Start CPU threads
        num_loader_threads = 3
        for i in range(num_loader_threads):
            worker = Thread(target=cpu_thread)
            worker.setDaemon(True)
            worker.start()

            worker = Thread(target=save_thread)
            worker.setDaemon(True)
            worker.start()

        # Start GPU threads
        worker = Thread(target=gpu_thread, args=(sess, ops))
        worker.setDaemon(True)
        worker.start()

        ref_meta = load_csv(REF_CSV)
        query_meta = load_csv(QUERY_CSV)
        ref_xy = get_xy(ref_meta)
        query_xy = get_xy(query_meta)

        [top_i, top_g_dists, top_f_dists, gt_i, gt_g_dist, ref_idx] = load_pickle(TOP_N_PICKLE)
        top_n = np.array(top_i)

        num = len(query_meta['path'])
        # Fewer queries for speed
        last_xy = query_xy[0, :]
        selected = [0]
        if QUERY_CSV.startswith('pittsburgh'):
            selected = np.linspace(0, num, 500, dtype=int)
        else:
            if 'freiburg' in QUERY_CSV:
                r = 0.5
            else:
                r = 2
            for i in range(num):
                if sum((query_xy[i, :] - last_xy) ** 2) > r ** 2:
                    last_xy = query_xy[i, :]
                    selected.append(i)

            selected = np.array(selected, dtype=int)

        xy_dists = pairwise_distances(query_xy, ref_xy, metric='euclidean')

        # Clean list
        image_info = [(query_meta['path'][i], ref_meta['path'][top_n[i, 0]]) for i in
                      selected]
        image_dist = [
            (np.linalg.norm(query_xy[i] - ref_xy[top_n[i, 0]])) for i
            in
            selected]

        batched_indices = np.reshape(selected, (-1, TUPLES_PER_BATCH))
        batched_image_info = np.reshape(image_info, (-1, TUPLES_PER_BATCH * 2))
        batched_distances = np.reshape(image_dist, (-1, TUPLES_PER_BATCH))

        for batch_indices, batch_image_info, batched_distance in zip(batched_indices, batched_image_info,
                                                                     batched_distances):
            CPU_IN_QUEUE.put((batch_indices, batch_image_info, batched_distance))

        # Wait for completion & order output
        CPU_IN_QUEUE.join()
        GPU_IN_QUEUE.join()
        GPU_OUT_QUEUE.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Image folder
    parser.add_argument('--img_root', default=os.path.join(fs_root()))
    parser.add_argument('--query_csv')
    parser.add_argument('--ref_csv')
    parser.add_argument('--top_n_pickle')
    parser.add_argument('--checkpoint')

    # Output
    parser.add_argument('--out_root',
                        default=os.path.join(fs_root(), 'grad_cam'))
    parser.add_argument('--log_dir',
                        default=os.path.join(fs_root(), 'logs/grad_cam'))

    # Network
    parser.add_argument('--vlad_cores', default=64, type=int)

    # Image Size
    parser.add_argument('--rescale', default=True)
    parser.add_argument('--small_side', default=180, type=int)
    parser.add_argument('--large_side', default=240, type=int)

    FLAGS = parser.parse_args()

    # Define each FLAG as a variable (generated automatically with util.flags_to_globals(FLAGS))
    flags_to_globals(FLAGS)

    CHECKPOINT = FLAGS.checkpoint
    IMG_ROOT = FLAGS.img_root
    LARGE_SIDE = FLAGS.large_side
    LOG_DIR = FLAGS.log_dir
    OUT_ROOT = FLAGS.out_root
    QUERY_CSV = FLAGS.query_csv
    REF_CSV = FLAGS.ref_csv
    RESCALE = FLAGS.rescale
    SMALL_SIDE = FLAGS.small_side
    TOP_N_PICKLE = FLAGS.top_n_pickle
    VLAD_CORES = FLAGS.vlad_cores

    OUT_FOLDER = os.path.join(OUT_ROOT, os.path.splitext(os.path.basename(TOP_N_PICKLE))[0])
    mkdir(OUT_FOLDER)

    TUPLES_PER_BATCH = 1  # Don't change this, save thread does not handle larger sizes

    CPU_IN_QUEUE = Queue(maxsize=0)
    GPU_IN_QUEUE = Queue(maxsize=10)
    GPU_OUT_QUEUE = Queue(maxsize=0)

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(OUT_ROOT):
        os.makedirs(OUT_ROOT)

    LOG = open(os.path.join(LOG_DIR, 'train_log.txt'), 'a')
    log('Running {} at {}.'.format(__file__, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    log(FLAGS)

    # Make reproducible (and same condition for all loss functions)
    random.seed(42)
    np.random.seed(42)
    get_grad_cam()
    LOG.close()
