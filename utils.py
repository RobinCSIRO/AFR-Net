import scipy.misc
import numpy as np
import os
import cv2
import random
import math
from glob import glob
import matplotlib.image as img
import tensorflow as tf
import tensorflow.contrib.slim as slim

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

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
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = merge(images, size)
    return scipy.misc.imsave(path, image)


def inverse_transform(images):
    return np.divide((images+1.0),2.0)


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

#def img_read(path, resize_W, resize_H, resize):
#    Read_img = scipy.misc.imread(path).astype(np.float)
#    if resize == 1:
#        image = scipy.misc.imresize(Read_img, [resize_W,resize_H])
#        image = np.array(image)/127.5 - 1.
#    else:
#        image = scipy.misc.imresize(Read_img, [resize_W,resize_H])
#        image = np.array(image)/127.5 - 1.
#    return image

def img_read(path, resize_H, resize_W, resize=1):
    Read_img = cv2.imread(path, cv2.IMREAD_COLOR)
    #print(Read_img.shape)
    if resize == 1:
        image = cv2.resize(src=Read_img, dsize=(resize_H,resize_W),interpolation=cv2.INTER_LINEAR)
        #print(image.shape)
        image = np.divide(np.array(image, np.float32),127.5) - 1.0
        #print(image.shape)
    else:
        image = cv2.resize(src=Read_img, dsize=(resize_W,resize_H),interpolation=cv2.INTER_LINEAR)
        image = np.divide(np.array(image, np.float32),127.5) - 1.0
        #image = np.expand_dims(image, 0)
    return image

def minmax_scale(input_arr):
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)
    return output_arr

def my_minmax_scale(input_arr):
    #print(input_arr.shape)
    output_arr = np.clip(input_arr,-1.0,1.0)
    #print(output_arr.shape)
    #output_arr = np.multiply((output_arr+1.0),127.5)
    return output_arr

def post_ouput(in_derainimg):
    for i in range(in_derainimg.shape[2]):
        in_derainimg[:,:,i] = my_minmax_scale(in_derainimg[:,:,i])
    #out_derainimg = np.array(in_derainimg, np.uint8)
    out_derainimg = np.array(in_derainimg)
    return out_derainimg

def att_minmax_scale(input_arr):
    #print(input_arr.shape)
    output_arr = np.clip(input_arr,0.0,1.0)
    return output_arr

def att_post_ouput(att_map):
    for i in range(att_map.shape[2]):
        att_map[:,:,i] = att_minmax_scale(att_map[:,:,i])
    return att_map

def get_patches(rimage, gimage, mimage, num_patches=48, patch_size=80, patch_stride=80):
    """Get `num_patches` from the image"""
    num_FSpatches = 16
    num_RApatches = 32
    rpatches = []
    gpatches = []
    mpatches = []
    #R_imgs = ((rimage+1)*127.5).astype(np.uint8)
    #scipy.misc.imsave('results'+'/' + 'rainy.jpg', R_imgs[0,:,:,:])
    for i in range(int(math.sqrt(num_FSpatches))):
      for j in range(int(math.sqrt(num_FSpatches))):
        point_x = patch_stride*i
        point_y = patch_stride*j
        rpatch = rimage[:,(point_x):(point_x+patch_size), (point_y):(point_y+patch_size),:]
        #print(point_x)
        #print(point_y)
        #print(point_y+patch_size)
        #P_imgs = ((rpatch+1)*127.5).astype(np.uint8)
        #scipy.misc.imsave('results'+'/' + 'patch_%d_%d.jpg'%(i,j), P_imgs[0,:,:,:])
        #print(rpatch.shape)
        rpatches.append(rpatch)
        #print(np.array(rpatches).shape)
        gpatch = gimage[:,(point_x):(point_x+patch_size), (point_y):(point_y+patch_size),:]
        gpatches.append(gpatch)
        mpatch = mimage[:,(point_x):(point_x+patch_size), (point_y):(point_y+patch_size),:]
        mpatches.append(mpatch)

    for k in range(num_RApatches):
      point1 = random.randint(0,240) # 116 comes from the image source size (320) - the patch dimension (80)
      point2 = random.randint(0,240)
      #rpatch = tf.image.crop_to_bounding_box(rimage, point1, point2, patch_size, patch_size)
      rpatch = rimage[:,(point1):(point1+patch_size), (point2):(point2+patch_size),:]
      #P_imgs = ((rpatch+1)*127.5).astype(np.uint8)
      #scipy.misc.imsave('results'+'/' + 'patch_%d.jpg'%i, P_imgs[0,:,:,:])
      #print(rpatch.shape)
      rpatches.append(rpatch)
      #print(np.array(rpatches).shape)
      gpatch = gimage[:,(point1):(point1+patch_size), (point2):(point2+patch_size),:]
      gpatches.append(gpatch)
      mpatch = mimage[:,(point1):(point1+patch_size), (point2):(point2+patch_size),:]
      mpatches.append(mpatch)

    rpatches = np.array(rpatches)
    rpatches = np.squeeze(rpatches)
    #print(rpatches.shape)
    gpatches = np.array(gpatches)
    gpatches = np.squeeze(gpatches)
    mpatches = np.array(mpatches)
    mpatches = np.squeeze(mpatches)
    #assert rpatches.get_shape().dims == [num_patches, patch_size, patch_size, 3]
    assert rpatches.shape == (num_patches, patch_size, patch_size, 3)
    return rpatches, gpatches, mpatches

def create3_channel_masks(masks):
    #print(masks.shape)
    masks_3c = np.zeros((*masks.shape, 1), dtype=np.float32)
    #print(masks_3c.shape)
    for idx in range(masks.shape[0]):
        #print(idx)
        mask = masks[idx]
        #print(mask.shape)
        masks_3c[idx, :, :, :] = np.repeat(mask[:, :, np.newaxis], 1, axis=2)
        #print(masks_3c.shape)
    #assert 1 == 0
    return masks_3c

def read_tridata(rainy_path, gt_path, Height, Width, ID, batch_size):
    Data  = []
    GT = [] 
    Res = []
    RMSK_Grey = []
    RMSK_RGB = []
    BMSK_Grey = []  
    BMSK_RGB = [] 
  
    for i in range(batch_size):
        #print('=====================')
        gt_read_path = gt_path+'%d_clean.png'%(i+ID)
        gt = img_read(gt_read_path, Height, Width)
        rain_read_path = rainy_path+'%d_rain.png'%(i+ID)
        rainy = img_read(rain_read_path, Height, Width)

        Data.append(rainy)
        #print(rainy.shape)
        #print(np.array(Data).shape)
        GT.append(gt)
        #print(np.array(GT).shape)
        res = np.abs(rainy-gt)
        Res.append(res)
        #print(np.array(Res).shape)
        curr_res_img = np.abs(rainy-gt)
        #print(curr_res_img.shape)
        mean = np.mean(curr_res_img)
        #print(mean)
        curr_res_img = curr_res_img - mean
        #print(curr_res_img.shape)
        binary_res_img = -np.sign(curr_res_img)
        #print(binary_res_img.shape)
        #print(binary_res_img)

        rainy1 = np.multiply((rainy+1.0),127.5)
        rainy1 = np.array(rainy1).astype(np.uint8)
        rainy1 = cv2.cvtColor(rainy1, cv2.COLOR_BGR2GRAY)
        rainy1 = np.array(rainy1)/127.5 - 1.0
        gt1 = np.multiply((gt+1.0),127.5)
        gt1 = np.array(gt1).astype(np.uint8)
        gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2GRAY)
        gt1 = np.array(gt1)/127.5 - 1
        curr_res_img1 = np.abs(rainy1-gt1)
        #print(curr_res_img1.shape)
        mean1 = np.mean(curr_res_img1)
        mean_forselect = np.multiply(mean1,255)
        #if mean_forselect < 30:
        #    mean1 = 30/255
        #else:
        #    mean1 = mean1
        #print(mean1*255)
        curr_res_img1 = curr_res_img1 - mean1
        #print(curr_res_img1.shape)
        binary_res_img1 = -np.sign(curr_res_img1)
        #print(binary_res_img1)
        #mask_grey1 = (binary_res_img1+1)/2
        RMSK_grey = np.maximum(binary_res_img1, 0)
        #print(RMSK_grey)
        RMSK_Grey.append(RMSK_grey)
        #print(np.array(RMSK_Grey).shape)
        binary_res_img2 = np.sign(curr_res_img1)
        BMSK_grey = np.maximum(binary_res_img2, 0)
        BMSK_Grey.append(BMSK_grey)
        #print(np.array(BMSK_Grey).shape)
        #print('=====================')
    Data = np.array(Data)
    GT = np.array(GT)
    RMSK_Grey = np.array(RMSK_Grey)
    BMSK_Grey = np.array(BMSK_Grey)
    RMSK_RGB = create3_channel_masks(RMSK_Grey)  #RZero Mask (Raindrop part corresponds 0)
    BMSK_RGB = create3_channel_masks(BMSK_Grey)  #BZero Mask (Clean part corresponds 0)
    #print(RMSK_RGB.shape)
    #print(BMSK_RGB.shape)
    return Data, GT, RMSK_RGB, BMSK_RGB
