# %%
import os
import cv2
import time
import numpy as np
import neuralgym as ng

from PIL import Image
from pathlib import Path
from inpaint_model import InpaintCAModel

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# %%
# Load model

FLAGS = ng.Config(Path(__file__).parent.joinpath('inpaint.yml'))

# sess_config = tf.ConfigProto()
# sess_config.gpu_options.allow_growth = True

checkpoint_dir = Path(__file__).parent.joinpath(
    'model_logs/release_places2_256')

# %%


class InpaintImage(object):
    ''' Inpaint image in the folder with generative inpaint '''

    def __init__(self, image_folder):
        '''Initialize the Image inpaint
        :param: image_folder: The folder of the image of interest.
        '''

        print('\n' + '---- Inpaint image with generative inpaint ----')
        print('Working with image_folder: {}'.format(image_folder))
        t = time.time()

        self.folder = image_folder

        # Check if the inpaint has been done,
        # to prevent repeated inpaints.
        self.ok = self.folder.joinpath('inpaint.ok')

        if not self.ok.is_file():
            self.image = self.load_image()

            self.masks = self.load_masks()

            self.inpaint_masks()

            with open(self.ok, 'w') as f:
                f.writelines(
                    ['Inpaint job is done.', '{}'.format(time.time())])
        else:
            print(open(self.ok).read())

        print(
            '---- Inpaint image with generative inpaint costs {:.2f} seconds ----\n'.format(time.time() - t))
        pass

    def inpaint_masks(self):
        '''
        Inpaint the image with masks
        '''
        inpaint_folder = self.folder.joinpath('inpaint')
        if not inpaint_folder.is_dir():
            os.mkdir(inpaint_folder)

        image = self.image
        image = cv2.resize(image, (256, 256))
        h, w, _ = image.shape
        print(image.shape)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)

        model = InpaintCAModel()
        input_image_ph = tf.placeholder(tf.float32, shape=(1, 256, 256*2, 3))
        output = model.build_server_graph(FLAGS, input_image_ph)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.train.load_variable(
                checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')

        raw_size = self.image.shape[:2][::-1]

        for p_mask in self.masks:
            image = self.image.copy()
            p, mask = p_mask

            image = cv2.resize(image, (256, 256))
            mask = cv2.resize(mask, (256, 256))

            assert image.shape == mask.shape, 'Shapes are not the same'

            print('Inpaint: {}'.format(p.name))
            output_path = inpaint_folder.joinpath(p.name)

            h, w, _ = image.shape
            grid = 4
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :]
            print('Shape of image: {}, mask, {}'.format(image.shape, mask.shape))

            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)

            # load pretrained model
            tf.reset_default_graph()
            result = sess.run(output, feed_dict={input_image_ph: input_image})
            cv2.imwrite(output_path.as_posix(), cv2.resize(
                result[0][:, :, ::-1], raw_size))
            print('Processed: {}'.format(output_path.name))

            # # Inpaint the image with the mask
            # h, w, _ = image.shape
            # grid = 8
            # image = image[:h//grid*grid, :w//grid*grid, :]
            # mask = mask[:h//grid*grid, :w//grid*grid, :]
            # print('Shape of image: {}'.format(image.shape))

            # image = np.expand_dims(image, 0)
            # mask = np.expand_dims(mask, 0)
            # input_image = np.concatenate([image, mask], axis=2)

            # sess_config = tf.ConfigProto()
            # sess_config.gpu_options.allow_growth = True

            # model = InpaintCAModel()
            # with tf.Session(config=sess_config) as sess:
            #     input_image = tf.constant(input_image, dtype=tf.float32)
            #     output = model.build_server_graph(FLAGS, input_image)
            #     output = (output + 1.) * 127.5
            #     output = tf.reverse(output, [-1])
            #     output = tf.saturate_cast(output, tf.uint8)

            #     # load pretrained model
            #     vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            #     assign_ops = []
            #     for var in vars_list:
            #         vname = var.name
            #         from_name = vname
            #         var_value = tf.train.load_variable(checkpoint_dir, from_name)
            #         assign_ops.append(tf.assign(var, var_value))

            #     sess.run(assign_ops)
            #     print('Model loaded.')
            #     result = sess.run(output)
            #     cv2.imwrite(output_path.as_posix(), result[0][:, :, ::-1])

            print('Wrote inpaint image: {}'.format(output_path))
        pass

    def load_image(self):
        '''
        Check if the image folder is valid and contains necessary files.
        And load the 'resize.jpg' image file for further processing.

        :return: The image
        '''

        folder = self.folder

        assert folder.is_dir(), 'Invalid image_folder: {}'.format(folder)
        for name in ['resize.jpg']:
            assert folder.joinpath(name).is_file(
            ), 'Invalid file: {}'.format(name)

        image = cv2.imread(folder.joinpath('resize.jpg').as_posix())

        print('Loaded resize.jpg from {}, {}'.format(image.size, folder.name))

        return image

    def load_masks(self):
        ''' Load the mask images from the folder

        :return: The list of masks, the element is the tuple of (path, PIL Image)
        '''
        folder = self.folder.joinpath('segment')
        if not folder.is_dir():
            print('E: Invalid folder: {}'.format(folder))
            return []

        masks = []
        for p in [e for e in folder.iterdir() if e.name.startswith('mask-')]:
            masks.append((p, cv2.imread(p.as_posix())))

        print('Found masks: {}'.format([e[0].name for e in masks]))

        return masks


# %%
if __name__ == "__main__":
    folder = Path(__file__).parent.parent.joinpath('image')

    for image_folder in [e for e in folder.iterdir() if e.is_dir()]:
        InpaintImage(
            image_folder=image_folder,
        )
