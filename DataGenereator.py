import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from os import path, walk, getcwd


class DataGenClass:
    def __init__(self, scale, lr_image_size, div2k_dir):

        _scales = [4]  # 2, 8
        if scale in _scales:
            self.scale = scale
        else:
            raise ValueError(f'scale must be in ${_scales}')

        self.crop_size = scale * lr_image_size

        self.hr_images_dir_train = div2k_dir + '/images/DIV2K_train_HR'
        self.hr_images_dir_val = div2k_dir + '/images/DIV2K_valid_HR'
        self.hr_cache_file_train = div2k_dir + '/caches/DIV2K_train_HR.cache'
        self.hr_cache_index_train = div2k_dir + '/caches/DIV2K_train_HR.cache.index'
        self.hr_cache_file_val = div2k_dir + '/caches/DIV2K_valid_HR.cache'
        self.hr_cache_index_val = div2k_dir + '/caches/DIV2K_valid_HR.cache.index'

        if scale == 8:
            pass
        elif scale == 4:
            self.lr_images_dir_train = div2k_dir + '/images/DIV2K_train_LR_bicubic/X4'
            self.lr_images_dir_val = div2k_dir + '/images/DIV2K_valid_LR_bicubic/X4'
            self.lr_cache_file_train = div2k_dir + '/caches/' + f'DIV2K_train_LR_bicubic_X4.cache'
            self.lr_cache_index_train = div2k_dir + '/caches/' + f'DIV2K_train_LR_bicubic_X4.cache.index'
            self.lr_cache_file_val = div2k_dir + '/caches/' + f'DIV2K_valid_LR_bicubic_X4.cache'
            self.lr_cache_index_val = div2k_dir + '/caches/' + f'DIV2K_valid_LR_bicubic_X4.cache.index'

        elif scale == 2:
            pass

        # train
        img_ids_train = range(1, 801)
        self.hr_image_files_train = [path.join(self.hr_images_dir_train, f'{image_id:04}.png') for image_id in
                                     img_ids_train]
        self.lr_image_files_train = [path.join(self.lr_images_dir_train, f'{image_id:04}x4.png') for image_id in
                                     img_ids_train]
        print(
            'hr (n={}) and lr (n={}) images for training.'.format(len(self.hr_image_files_train),
                                                                  len(self.lr_image_files_train)))

        # val
        img_ids_val = range(801, 901)
        self.hr_image_files_val = [path.join(self.hr_images_dir_val, f'{image_id:04}.png') for image_id in img_ids_val]
        self.lr_image_files_val = [path.join(self.lr_images_dir_val, f'{image_id:04}x4.png') for image_id in
                                   img_ids_val]
        print('hr (n={}) and lr (n={}) images for validation.'.format(len(self.hr_image_files_val),
                                                                      len(self.lr_image_files_val)))

    def images_to_dataset(self, subset):
        """
        Read image files from folder then construct and cache a dataset by
        using TensorFlow 'Dataset.from_tensor_slices' method
        :param subset:  train and validation datasets of HR_for_train, LR_for_train, HR_for_val, LR_for_val
        :return: tf dataset
        """
        if subset == 'HR_for_train':
            image_files = self.hr_image_files_train
            cache_file =  self.hr_cache_file_train
            cache_index = self.hr_cache_index_train
        elif subset=='LR_for_train':
            image_files = self.lr_image_files_train
            cache_file = self.lr_cache_file_train
            cache_index = self.lr_cache_index_train
        elif subset == 'HR_for_val':
            image_files = self.hr_image_files_val
            cache_file = self.hr_cache_file_val
            cache_index = self.hr_cache_index_val
        elif subset == 'LR_for_val':
            image_files = self.lr_image_files_val
            cache_file = self.lr_cache_file_val
            cache_index = self.lr_cache_index_val
        else:
            print('subset is unknown: [HR_for_train, LR_for_train, HR_for_val, LR_for_val]')

        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        ds = ds.cache(cache_file)
        if not path.exists(cache_index):
            print(f'Caching decoded images in {cache_file} ...')
            self.create_cache(ds)
        return ds

    def dataset_to_batch(self, ds_lr, ds_hr, batch_size = 32, repeat_count=None, random_transform=True):
        """
        Use two tf datasets of LR dataset and HR dataset to make batches
        :param ds_lr:
        :param ds_hr:
        :param batch_size:
        :param repeat_count:
        :param random_transform:
        :return: dataset of mutual (lr, hr) batches for training or validation
        """
        ds = tf.data.Dataset.zip((ds_lr, ds_hr))
        if random_transform:
            ds = ds.map(lambda lr, hr: self.random_crop(lr, hr, hr_crop_size=self.crop_size, scale=self.scale),
                        num_parallel_calls=AUTOTUNE)
            ds = ds.map(self.random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(self.random_flip, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    @staticmethod
    def create_cache(ds):
        """
        Spand whole dataset (for making cache)
        :param ds: tf dataset object
        :return: none
        """
        _ = list(ds.as_numpy_iterator())
        # for _ in ds: pass
        print(f'Done!')

    # -----------------------------------------------------------
    #  Image preprocesseing and transformations
    # -----------------------------------------------------------
    @staticmethod
    def random_crop(lr_img, hr_img, hr_crop_size, scale):
        """
        Preprocess input images by selecting a part of lr image randomly then cropping its hr counterpart
        :param lr_img: input low res image
        :param hr_img: the equivalent high res image
        :param hr_crop_size: size of high res image as ground truth. usually lr_size * scale
        :param scale:  magnifying scale factor
        :return: a set of lr and hr counterparts
        """
        lr_crop_size = hr_crop_size // scale
        lr_img_shape = tf.shape(lr_img)[:2]

        lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
        lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

        hr_w = lr_w * scale
        hr_h = lr_h * scale

        lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
        hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

        return lr_img_cropped, hr_img_cropped

    @staticmethod
    def random_flip(lr_img, hr_img):
        """
        Flip images
        :param lr_img: input low res image
        :param hr_img: high res image counterpart
        :return: a set of lr and hr counterparts
        """
        rn = tf.random.uniform(shape=(), maxval=1)
        return tf.cond(rn < 0.5,
                       lambda: (lr_img, hr_img),
                       lambda: (tf.image.flip_left_right(lr_img),
                                tf.image.flip_left_right(hr_img)))

    @staticmethod
    def random_rotate(lr_img, hr_img):
        """
        Rotate images in 0, 90, 180 and 270 degrees randomly
        :param lr_img: input low res image
        :param hr_img: high res image counterpart
        :return: a set of lr and hr counterparts
        """
        rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
        return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)
