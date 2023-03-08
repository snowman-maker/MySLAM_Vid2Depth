"""Generates depth estimates for an entire KITTI video."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
# from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import model
import numpy as np
import scipy.misc
from PIL import Image
import imageio
import tensorflow as tf
import util
import colorprint
from config import configer

gfile = tf.io.gfile

# 多彩输出
color = colorprint.Log()

CMAP = 'plasma'


def _run_inference(cfg):
    """Runs all images through depth model and saves depth maps."""

    kitti_dir = cfg['inference']['kitti_dir']
    output_dir = cfg['inference']['output_dir']
    kitti_video = cfg['inference']['kitti_video']
    model_ckpt = cfg['inference']['model_ckpt']
    batch_size = int(cfg['inference']['batch_size'])
    img_height = int(cfg['inference']['img_height'])
    img_width = int(cfg['inference']['img_width'])
    seq_length = int(cfg['inference']['seq_length'])

    ckpt_basename = os.path.basename(model_ckpt)
    ckpt_modelname = os.path.basename(os.path.dirname(model_ckpt))
    output_dir = os.path.join(output_dir,
                              kitti_video.replace('/', '_') + '_' +
                              ckpt_modelname + '_' + ckpt_basename)
    if not gfile.exists(output_dir):
        gfile.makedirs(output_dir)
    inference_model = model.Model(is_training=False,
                                  seq_length=seq_length,
                                  batch_size=batch_size,
                                  img_height=img_height,
                                  img_width=img_width)
    vars_to_restore = util.get_vars_to_restore(model_ckpt)
    saver = tf.compat.v1.train.Saver(vars_to_restore)
    sv = tf.compat.v1.train.Supervisor(logdir='/tmp/', saver=None)
    with sv.managed_session() as sess:
        saver.restore(sess, model_ckpt)
        if kitti_video == 'test_files_eigen':
            im_files = util.read_text_lines(
                util.get_resource_path('dataset/kitti/test_files_eigen.txt'))
            im_files = [os.path.join(kitti_dir, f) for f in im_files]
        else:
            video_path = os.path.join(kitti_dir, kitti_video)
            im_files = gfile.glob(os.path.join(video_path, 'image_02/data', '*.png'))
            im_files = [f for f in im_files if 'disp' not in f]
            im_files = sorted(im_files)
        for i in range(0, len(im_files), batch_size):
            if i % 100 == 0:
                logging.info('Generating from %s: %d/%d', ckpt_basename, i,
                             len(im_files))
            inputs = np.zeros(
                (batch_size, img_height, img_width, 3),
                dtype=np.uint8)
            for b in range(batch_size):
                idx = i + b
                if idx >= len(im_files):
                    break
                im = imageio.imread(im_files[idx])
                inputs[b] = np.array(Image.fromarray(im).resize((img_width, img_height)))
            results = inference_model.inference(inputs, sess, mode='depth')
            for b in range(batch_size):
                idx = i + b
                if idx >= len(im_files):
                    break
                if kitti_video == 'test_files_eigen':
                    depth_path = os.path.join(output_dir, '%03d.png' % idx)
                else:
                    depth_path = os.path.join(output_dir, '%04d.png' % idx)

                depth_map = results['depth'][b]
                depth_map = np.squeeze(depth_map)
                colored_map = _normalize_depth_for_display(depth_map, cmap=CMAP)
                imageio.imsave(depth_path, colored_map)

                # depth_map = results['depth'][b]
                # depth_map = np.squeeze(depth_map)
                # colored_map = _normalize_depth_for_display(depth_map, cmap=CMAP)
                # input_float = inputs[b].astype(np.float32) / 255.0
                # vertical_stack = np.concatenate((input_float, colored_map), axis=0)
                # scipy.misc.imsave(depth_path, vertical_stack)


def _gray2rgb(im, cmap=CMAP):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img


def _normalize_depth_for_display(depth,
                                 pc=95,
                                 crop_percent=0,
                                 normalizer=None,
                                 cmap=CMAP):
    """Converts a depth map to an RGB image."""
    # Convert to disparity.
    disp = 1.0 / (depth + 1e-6)
    if normalizer is not None:
        disp /= normalizer
    else:
        disp /= (np.percentile(disp, pc) + 1e-6)
    disp = np.clip(disp, 0, 1)
    disp = _gray2rgb(disp, cmap=cmap)
    keep_h = int(disp.shape[0] * (1 - crop_percent))
    disp = disp[:keep_h]
    return disp


def main(cfg):
    _run_inference(cfg)


if __name__ == '__main__':
    # 获取 config
    cfg_file_path = './config/config.yaml'
    cfg = configer.load_config(cfg_file_path)

    # Run main
    app.run(main(cfg))
