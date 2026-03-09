"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import cv2
import numpy as np
import os
import time
import datetime
from time import gmtime, strftime
from six.moves import xrange
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def expand_path(path):
  return os.path.expanduser(os.path.expandvars(path))

def timestamp(s='%Y%m%d.%H%M%S', ts=None):
  if not ts: ts = time.time()
  st = datetime.datetime.fromtimestamp(ts).strftime(s)
  return st
  
def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def preprocess_image(image, normalize=True, denoise=True, enhance=True):
    """增强的图像预处理函数
    Args:
        image: 输入图像 numpy array
        normalize: 是否进行归一化
        denoise: 是否进行去噪
        enhance: 是否进行图像增强
    Returns:
        处理后的图像
    """
    try:
        # 转换为PIL图像以便处理
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image))
            
        if denoise:
            # 使用双边滤波进行去噪，保持边缘清晰
            image_np = np.array(image)
            if len(image_np.shape) == 3:
                for i in range(3):
                    image_np[:,:,i] = cv2.bilateralFilter(image_np[:,:,i], 5, 75, 75)
            else:
                image_np = cv2.bilateralFilter(image_np, 5, 75, 75)
            image = Image.fromarray(np.uint8(image_np))
            
        if enhance:
            # 自适应直方图均衡化
            if image.mode != 'L':
                # 增强对比度但保持自然度
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
                
                # 适度增强色彩饱和度
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.1)
                
                # 增强清晰度
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.3)
                
                # 调整亮度
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.1)
        
        # 转回numpy数组
        image = np.array(image)
        
        if normalize:
            # 使用更稳定的归一化方法
            image = (image.astype(np.float32) - 127.5) / 127.5
            # 裁剪到有效范围
            image = np.clip(image, -1, 1)
            
        return image
    except Exception as e:
        print(f"Image preprocessing failed: {str(e)}")
        return image

def augment_image(image):
    """图像增强函数，增加数据多样性"""
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image))
            
        # 随机应用以下增强方法
        if np.random.random() < 0.5:
            # 随机小角度旋转
            angle = np.random.uniform(-10, 10)
            image = image.rotate(angle, Image.BILINEAR, expand=False)
            
        if np.random.random() < 0.3:
            # 随机轻微缩放
            scale = np.random.uniform(0.95, 1.05)
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.BILINEAR)
            
        if np.random.random() < 0.3:
            # 随机调整亮度
            enhancer = ImageEnhance.Brightness(image)
            factor = np.random.uniform(0.9, 1.1)
            image = enhancer.enhance(factor)
            
        if np.random.random() < 0.3:
            # 随机调整对比度
            enhancer = ImageEnhance.Contrast(image)
            factor = np.random.uniform(0.9, 1.1)
            image = enhancer.enhance(factor)
            
        return np.array(image)
    except Exception as e:
        print(f"Image augmentation failed: {str(e)}")
        return np.array(image)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    """改进的图像加载函数"""
    image = imread(image_path, grayscale)
    
    # 添加预处理步骤
    image = preprocess_image(image, 
                           normalize=False,  # 稍后会归一化
                           denoise=True, 
                           enhance=True)
    
    # 添加数据增强
    image = augment_image(image)
    
    if crop:
        cropped = center_crop(image, input_height, input_width,
                            resize_height, resize_width)
    else:
        # 使用高质量的resize方法
        im = Image.fromarray(np.uint8(image))
        cropped = np.array(im.resize((resize_width, resize_height), 
                                   Image.BICUBIC))
    
    # 确保图像维度正确
    if len(cropped.shape) == 2:
        cropped = np.expand_dims(cropped, axis=2)
    
    # 转换为float32并归一化到[-1, 1]范围
    return np.array(cropped).astype(np.float32)/127.5 - 1.

def save_images(images, size, image_path):
    """改进的图像保存函数"""
    # 反归一化
    images = inverse_transform(images)
    
    # 后处理以提高图像质量
    processed_images = []
    for img in images:
        # 添加轻微锐化和对比度增强
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = ImageEnhance.Sharpness(img).enhance(1.1)
        img = ImageEnhance.Contrast(img).enhance(1.1)
        processed_images.append(np.array(img))
    
    images = np.array(processed_images)
    
    # 确保保存路径存在
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    
    return imsave(images, size, image_path)

def imread(path, grayscale=False):
    if grayscale:
        img = Image.open(path).convert('L')
        return np.array(img)
    else:
        img = Image.open(path).convert('RGB')
        return np.array(img)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    # 确保图像是3通道的
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    # 使用PIL替代scipy.misc
    try:
        image = Image.fromarray(image)
        image.save(path)
        return True
    except:
        # 如果出错，尝试转换数据类型
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(path)
        return True

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    """改进的中心裁剪函数"""
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    
    # 确保裁剪区域不超出图像边界
    j = max(0, min(j, h - crop_h))
    i = max(0, min(i, w - crop_w))
    
    cropped = x[j:j+crop_h, i:i+crop_w]
    
    # 使用高质量的双三次插值进行resize
    im = Image.fromarray(np.uint8(cropped))
    im = im.resize((resize_w, resize_h), Image.BICUBIC)
    
    return np.array(im)

def transform(image, input_height, input_width, 
             resize_height=64, resize_width=64, crop=True):
    # 确保输入图像是uint8类型
    if image.dtype != np.uint8:
        image = (image).astype(np.uint8)
        
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width, 
            resize_height, resize_width)
    else:
        # 使用PIL替代scipy.misc.imresize
        im = Image.fromarray(image)
        cropped_image = np.array(im.resize((resize_height, resize_width), Image.BILINEAR))
    
    # 归一化到 [-1, 1] 范围
    return cropped_image.astype(np.float32)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option, sample_dir='samples'):
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim], os.path.join(sample_dir, 'test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime() )))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      save_images(samples, [image_frame_dim, image_frame_dim], os.path.join(sample_dir, 'test_arange_%s.png' % (idx)))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, dcgan.z_dim - 1) for _ in xrange(dcgan.z_dim)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      try:
        make_gif(samples, './samples/test_gif_%s.gif' % (idx))
      except:
        save_images(samples, [image_frame_dim, image_frame_dim], os.path.join(sample_dir, 'test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime() )))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, os.path.join(sample_dir, 'test_gif_%s.gif' % (idx)))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], os.path.join(sample_dir, 'test_gif_%s.gif' % (idx)))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w
