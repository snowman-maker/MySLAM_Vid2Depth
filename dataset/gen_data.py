"""Generates data for training/validation and save it to disk."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import multiprocessing
import os
import sys

from absl import app
import logging
import dataset_loader
import numpy as np
import scipy.misc
import tensorflow as tf

from config import configer
import colorprint

color = colorprint.Log()

gfile = tf.compat.v1.gfile

logging.basicConfig(
    stream=sys.stdout,
    format=color.green("[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s ---> %(message)s"),
    level=logging.INFO,
)

# Process data in chunks for reporting progress.
NUM_CHUNKS = 100


def _generate_data(cfg):
    """Extract sequences from dataset_dir and store them in data_dir."""
    dataset_name = cfg['gen_data']['dataset_name']
    dataset_dir = cfg['gen_data']['dataset_dir']
    data_dir = cfg['gen_data']['data_dir']
    seq_length = int(cfg['gen_data']['seq_length'])
    img_height = int(cfg['gen_data']['img_height'])
    img_width = int(cfg['gen_data']['img_width'])
    num_threads = int(cfg['gen_data']['num_threads'])

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    logging.info(color.green(f'Processing: {dataset_name}'))

    global dataloader  # pylint: disable=global-variable-undefined
    if dataset_name == 'bike':
        dataloader = dataset_loader.Bike(dataset_dir,
                                         img_height=img_height,
                                         img_width=img_width,
                                         seq_length=seq_length)
    elif dataset_name == 'kitti_odom':
        dataloader = dataset_loader.KittiOdom(dataset_dir,
                                              img_height=img_height,
                                              img_width=img_width,
                                              seq_length=seq_length)
    elif dataset_name == 'kitti_raw_eigen':
        dataloader = dataset_loader.KittiRaw(dataset_dir,
                                             static_frames_file=cfg['gen_data']['static_frames_file'],
                                             test_scene_file=cfg['gen_data']['test_scene_file'],
                                             img_height=img_height,
                                             img_width=img_width,
                                             seq_length=seq_length, )
    elif dataset_name == 'kitti_raw_stereo':
        dataloader = dataset_loader.KittiRaw(dataset_dir,
                                             static_frames_file=cfg['gen_data']['static_frames_file'],
                                             test_scene_file=cfg['gen_data']['test_scene_file'],
                                             img_height=img_height,
                                             img_width=img_width,
                                             seq_length=seq_length)
    elif dataset_name == 'cityscapes':
        dataloader = dataset_loader.Cityscapes(dataset_dir,
                                               img_height=img_height,
                                               img_width=img_width,
                                               seq_length=seq_length)
    else:
        raise ValueError('Unknown dataset')

    # The default loop below uses multiprocessing, which can make it difficult
    # to locate source of errors in data loader classes.
    # Uncomment this loop for easier debugging:

    all_examples = {}
    for i in range(dataloader.num_train):
        _gen_example(i, all_examples, data_dir=data_dir)
        logging.info('Generated: %d examples', len(all_examples))

    all_frames = range(dataloader.num_train)
    frame_chunks = np.array_split(all_frames, NUM_CHUNKS)

    manager = multiprocessing.Manager()
    all_examples = manager.dict()
    num_cores = multiprocessing.cpu_count()
    num_threads = num_cores if num_threads is None else num_threads
    pool = multiprocessing.Pool(num_threads)

    # Split into training/validation sets. Fixed seed for repeatability.
    np.random.seed(8964)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with open(os.path.join(data_dir, 'train.txt'), 'w') as train_f:
        with open(os.path.join(data_dir, 'val.txt'), 'w') as val_f:
            logging.info('Generating data...')
            for index, frame_chunk in enumerate(frame_chunks):
                all_examples.clear()
                pool.map(_gen_example_star,
                         # itertools.izip(frame_chunk, itertools.repeat(all_examples))
                         zip(frame_chunk, itertools.repeat(all_examples))
                         )
                logging.info('Chunk %d/%d: saving %s entries...', index + 1, NUM_CHUNKS,
                             len(all_examples))
                for _, example in all_examples.items():
                    if example:
                        s = example['folder_name']
                        frame = example['file_name']
                        if np.random.random() < 0.1:
                            val_f.write('%s %s\n' % (s, frame))
                        else:
                            train_f.write('%s %s\n' % (s, frame))
    pool.close()
    pool.join()


def _gen_example(i, all_examples, data_dir):
    """Saves one example to file.  Also adds it to all_examples dict."""
    example = dataloader.get_example_with_index(i)
    if not example:
        return
    image_seq_stack = _stack_image_seq(example['image_seq'])
    example.pop('image_seq', None)  # Free up memory.
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    save_dir = os.path.join(data_dir, example['folder_name'])
    if not gfile.Exists(save_dir):
        gfile.MakeDirs(save_dir)
    img_filepath = os.path.join(save_dir, '%s.jpg' % example['file_name'])
    scipy.misc.imsave(img_filepath, image_seq_stack.astype(np.uint8))
    cam_filepath = os.path.join(save_dir, '%s_cam.txt' % example['file_name'])
    example['cam'] = '%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy)
    with open(cam_filepath, 'w') as cam_f:
        cam_f.write(example['cam'])

    key = example['folder_name'] + '_' + example['file_name']
    all_examples[key] = example


def _gen_example_star(params):
    return _gen_example(*params)


def _stack_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res


def main(cfg):
    _generate_data(cfg)


if __name__ == '__main__':
    # 获取 config
    cfg_file_path = '../config/config.yaml'
    cfg = configer.load_config(cfg_file_path)

    app.run(main(cfg))
