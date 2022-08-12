# %%
import cv2
import time
import json
import torch
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from transformers import pipeline
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

# %%
# Load the model,
# the loading work is time consuming,
# so we need to load them once for all.
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
object_detector = pipeline('object-detection')
segment_worker = pipeline('image-segmentation')

# %%


class TransformersImage(object):
    '''
    Analysis the image folder with the transformers.

    We assume the image folder has been established in the correct way.
    '''

    def __init__(self, image_folder):
        '''
        Initial the Image analysis

        :param: image_folder: The folder contains the image of interest.
        '''

        self.folder = image_folder

        print('\n' + '---- Analysis image with Transformers ----')
        print('Working with image_folder: {}'.format(image_folder))
        t = time.time()

        self.image = self.load_image()

        self.estimate_depth()

        self.detect_object()

        self.segment()

        print(
            '---- Analysis image with Transformers costs {:.2f} seconds ----\n'.format(time.time() - t))

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

        image = Image.open(folder.joinpath('resize.jpg'))

        print('Loaded resize.jpg from {}, {}'.format(image.size, folder.name))

        return image

    def segment(self):
        '''
        Segment the image.

        :output: The json of segment masks, 'segment.json'
        :output: [If the segment is available], the jpg image of segment masks, 'segment.jpg'
        :output: Image in the segment/ folder, the masks are the mask .jpg images (255 refers the inside area of the mask), the covers are the patches within the masks
        '''
        image = self.image

        json_path = self.folder.joinpath('segment.json')
        jpg_path = self.folder.joinpath('segment.jpg')

        if json_path.is_file():
            print('W: File exists, {}'.format(json_path))
        else:
            # Segment the image
            segment = segment_worker(image)

            # Save the segment masks
            mat = None
            for j, e in enumerate([e for e in segment]):
                idx = j + 1
                e['idx'] = idx
                label = e['label']
                score = e['score']
                mask = e['mask']

                im = np.array(image, dtype=np.uint8)
                m = np.array(mask, dtype=np.uint8)

                assert len(im.shape) == 3, 'Image size is incorrect'

                im = np.concatenate([im, m[:, :, np.newaxis]], axis=2)

                img = Image.fromarray(im, mode='RGBA')

                p = self.folder.joinpath(
                    'segment', 'cover-{}-{}-{:.2f}.png'.format(idx, label, score))
                img.save(p)
                print('Saved cover: {}'.format(p))

                p = self.folder.joinpath(
                    'segment', 'mask-{}-{}-{:.2f}.jpg'.format(idx, label, score))
                mask.save(p)
                print('Saved mask: {}'.format(p))

                if mat is None:
                    mat = np.array(mask) * 0

                mat[np.array(mask) > 0] = idx

            if mat is not None:
                plt.imshow(mat, cmap=plt.cm.tab10)
                plt.savefig(jpg_path)
                print('Saved segment image: {}'.format(jpg_path))

            # Save the segment json
            seg = segment.copy()
            [e.pop('mask') for e in seg]
            json.dump(seg, open(json_path, 'w'))
            print('Segment into: {}'.format([e['label'] for e in seg]))

        pass

    def detect_object(self):
        '''
        Object detection on the image.

        :output: The json of object boxes, 'objects.json'
        :output: The image of object boxes, 'objects.jpg'
        '''
        image = self.image

        # Object detection
        path = self.folder.joinpath('objects.json')
        if path.is_file():
            print('W: File exists, {}'.format(path))
            objects = json.load(open(path))
        else:
            objects = object_detector(image)
            json.dump(objects, open(path, 'w'))
        print('Detect objects: {}'.format([e['label'] for e in objects]))

        # Draw detection
        path = self.folder.joinpath('objects.jpg')
        if path.is_file():
            print('W: File exists, {}'.format(path))
        else:
            img = image.copy()

            font = ImageFont.truetype(
                font='/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', size=16)

            colormap = px.colors.qualitative.Dark24

            for j, det in enumerate(objects):
                color = colormap[j % len(colormap)]
                label = det['label']
                score = det['score']
                box = det['box']
                shape = [(box['xmin'], box['ymin']),
                         (box['xmax'], box['ymax'])]

                draw = ImageDraw.Draw(img)
                draw.rectangle(shape, outline=color, width=2)
                draw.text((shape[0][0], shape[0][1]-20),
                          '{} ({:.2f})'.format(label, score), font=font, fill=color)
                print('New image: {}'.format(path))

            img.save(path)

        return

    def estimate_depth(self):
        '''
        Estimates the depth map of the image.

        :output: The depth map is saved as 'depth.jpg'
        '''
        image = self.image

        path = self.folder.joinpath('depth.jpg')
        if path.is_file():
            print('W: File exists, {}'.format(path))
        else:
            # Feature selection
            inputs = feature_extractor(image, return_tensors='pt')
            with torch.no_grad():
                feature = model(**inputs)

            # Estimate depth
            depth = feature.predicted_depth

            # Convert into CPU numpy array, an image
            depth = depth.squeeze().cpu().numpy()

            depth = cv2.resize(depth, image.size)

            a = np.max(depth)
            b = np.min(depth)
            depth = (depth - b) / (a - b) * 255
            depth = np.array(depth, dtype=np.uint8)

            depth_image = Image.fromarray(depth, mode="L")

            depth_image.save(path)
            print('New image: {}'.format(path))

        return


# %%
if __name__ == '__main__':
    folder = Path(__file__).parent.parent.joinpath('image')

    for image_folder in [e for e in folder.iterdir() if e.is_dir()]:
        TransformersImage(
            image_folder=image_folder,
        )

# %%
