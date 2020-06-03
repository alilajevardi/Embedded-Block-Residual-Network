import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import path, walk
from random import sample


class PostProcClass:
    def __init__(self, model, LR_img_size=64, scale_factor=4, n_imshow=3):
        self.model = model
        self.LR_img_size = LR_img_size
        self.scale = scale_factor
        self.n_show = n_imshow

    def psnr(self, im1, im2):
        """
        Calculate quality measurement via PSNR (Peak Signal to Noise Ratio) between two images 
        :param im1: predicted super resolution image 
        :param im2: high resolution image (ground truth)
        :return: psnr index
        """
        index = tf.image.psnr(im1, im2, max_val=255)
        return index

    def ssim(self, im1, im2):
        """
        Calculate quality measurement via SSIM (Structural Similarity Index) between two images
        :param im1: predicted super resolution image 
        :param im2: high resolution image (ground truth)
        :return: ssim index
        """
        return tf.image.ssim(im1, im2, max_val=255)

    def resolve(self, lr_batch):
        """
        Generate super resolution images by using trained model and a batch of test images
        :param lr_batch: a batch of test images
        :return: a batch of generated super resolution images 
        """
        lr_batch = tf.cast(lr_batch, tf.float32)
        sr_batch = self.model(lr_batch)
        sr_batch = tf.clip_by_value(sr_batch, 0, 255)
        sr_batch = tf.round(sr_batch)
        sr_batch = tf.cast(sr_batch, tf.uint8)
        return sr_batch

    def eval_img(self, lr, hr):
        """
        Evaluate quality of two images
        :param lr: low resolution image
        :param hr: high resolution image (ground truth)
        :return: psnr and ssim indexes
        """
        pnsr_values = []
        ssim_values = []
        sr = self.resolve(tf.expand_dims(lr, axis=0))[0]
        pnsr_values.append(self.psnr(sr, hr))
        ssim_values.append(self.ssim(tf.expand_dims(hr, axis=0), sr))  # tf.image.ssim needs a batch of images
        pnsr_v = tf.reduce_mean(pnsr_values)
        ssim_v = tf.reduce_mean(ssim_values)
        pnsr_v = float(pnsr_v.numpy())
        ssim_v = float(ssim_v.numpy())
        return pnsr_v, ssim_v

    def plot_sample(self, lr, sr, hr, psnr_n, ssim_n):
        """
        Plot three images in a row
        :param lr: input low res image
        :param sr: predicted super res image
        :param hr: high res image as ground truth
        :param psnr_n: PSNR number
        :param ssim_n: SSIM number
        :return: 
        """
        plt.figure(figsize=(20, 10))
        images = [lr, sr, hr]
        titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]}) \nPNSR:{round(psnr_n, 2)} \nSSIM:{round(ssim_n, 2)}', 'HR']
        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, 3, i + 1)
            plt.imshow(img)
            plt.title(title)
            plt.xticks([])
            plt.yticks([])

    def img_crop(self, img, w, h):
        """
        Select a patch of a big hr image for test then convert it to low res via bicubic method
        :param img: raw test image that must be bigger than low resolution*scale_factor (64*4 = 256)
        :param w: width of raw test image
        :param h: height of raw test image
        :return: set of low res image and its equivalent high res image
        """
        img = np.array(img)
        hr_size = self.LR_img_size * self.scale

        img_w = np.random.randint(0, w - hr_size + 1)
        img_h = np.random.randint(0, h - hr_size + 1)
        hr_img_cropped = img[img_h:img_h + hr_size, img_w:img_w + hr_size]

        #     hr_img_cropped = img[(h-hr_size)//2:(h+hr_size)//2, (w-hr_size)//2:(w+hr_size)//2]

        hr_img_cropped_1 = Image.fromarray(hr_img_cropped)

        lr_img = hr_img_cropped_1.resize((self.LR_img_size, self.LR_img_size), Image.BICUBIC)

        return np.array(lr_img), hr_img_cropped

    def proper_test_img(self, test_dir):
        """
        Search test folder to find and list suitable raw images for test (with bigger than minimum size)
        :param test_dir: path of test image folder
        :return: list of suitable images for test
        """
        lst_proper_img = []
        for dirpath, _, img_files in walk(test_dir):
            pass
        for img_i in img_files:
            img_f = path.join(dirpath, img_i)
            with Image.open(img_f) as img:
                width, height = img.size
                if width < self.LR_img_size * self.scale or height < self.LR_img_size * self.scale:
                    print('{0} [w:{1}, h:{2}] was dropped from test list. Min size = ({3}px by {3}px)'.
                          format(img_i, width, height, self.LR_img_size * self.scale))
                else:
                    # print('{} sizes are [w:{}, h:{}]'.format(img_i, width, height))
                    lst_proper_img.append(img_i)
        return lst_proper_img

    def eval_test_dir(self, test_dir):
        """
        Evaluate images in a test folder by preparing lr image then generate sr image and compare it with hr image
        :param test_dir: path of test image folder
        :return: average of psnr and ssim indexes and number of used images
        """
        lst = self.proper_test_img(test_dir)
        pnsr_lst = []
        ssim_lst = []

        rn = sample(range(0, len(lst)), self.n_show) # n_show images in test_dir will draw randomly

        for ii, img_i in enumerate(lst):
            img_f = path.join(test_dir, img_i)
            with Image.open(img_f) as img:
                width, height = img.size
                lr_img, hr_img = self.img_crop(img, width, height)
                pnsr_v, ssim_v = self.eval_img(lr_img, hr_img)
                pnsr_lst.append(pnsr_v)
                ssim_lst.append(ssim_v)

                # convert single image to batch with expand_dim
                sr_img = self.resolve(tf.expand_dims(lr_img, axis=0))[0]
                if ii in rn:
                    self.plot_sample(lr_img, sr_img, hr_img, pnsr_v, ssim_v)

        return sum(pnsr_lst) / len(pnsr_lst), sum(ssim_lst) / len(ssim_lst), len(lst)
