# %%
import io
import os
import re
import time
import base64
import numpy as np

from PIL import Image
from pathlib import Path

# %%
root = Path(__file__).parent.parent

# %%


def src2image(src):
    '''
    Decode src into PIL image
    :param: src:
        eg:
            src="data:image/gif;base64,R0lGODlhMwAxAIAAAAAAAP///
                yH5BAAAAAAALAAAAAAzADEAAAK8jI+pBr0PowytzotTtbm/DTqQ6C3hGX
                ElcraA9jIr66ozVpM3nseUvYP1UEHF0FUUHkNJxhLZfEJNvol06tzwrgd
                LbXsFZYmSMPnHLB+zNJFbq15+SOf50+6rG7lKOjwV1ibGdhHYRVYVJ9Wn
                k2HWtLdIWMSH9lfyODZoZTb4xdnpxQSEF9oyOWIqp6gaI9pI1Qo7BijbF
                ZkoaAtEeiiLeKn72xM7vMZofJy8zJys2UxsCT3kO229LH1tXAAAOw=="

    :return: PIL image
    '''

    # Parse src
    result = re.search(
        "data:image/(?P<ext>.*?);base64,(?P<data>.*)", src, re.DOTALL)

    if result:
        ext = result.groupdict().get("ext")
        data = result.groupdict().get("data")
    else:
        raise Exception("Can not parse the src")

    # Decode base64 data
    img = base64.urlsafe_b64decode(data)

    # Convert into image
    buffer = io.BytesIO(img)
    image = Image.open(buffer)
    return image

# %%


class MyImage(object):
    def __init__(self, name, src=None):
        ''' Init the image object,

        :param: name: The name of the image;
        :param: src: The src of the image, default is None.

        '''
        self.name = name
        self.src = src
        self.folders = self.mk_folder()
        self.image = self.mk_image_from_src()
        self.resize_image()

        if self.image:
            print('---------------------------------------')
            print('Loaded image: {}'.format(name))
            print('Contains of {}'.format(self.folders['folder']))
            for e in self.folders['folder'].iterdir():
                print('  {}'.format(e.name))
            print('----')
        else:
            print('---------------------------------------')
            print('Failed on image: {}'.format(name))
            print('----')
        pass

    def mk_folder(self):
        ''' Make the folder for the image '''
        folder = root.joinpath('image', self.name)

        if folder.is_dir():
            print('W: Folder exists, create a new one, {}'.format(self.name))
            folder = root.joinpath(
                'image', '{}-{}'.format(self.name, time.time()))

        if folder.is_dir():
            print('W: Folder will be overwrote, {}'.format(folder.name))

        os.mkdir(folder)
        os.mkdir(folder.joinpath('label'))
        os.mkdir(folder.joinpath('segment'))

        folders = dict(
            folder=folder,
            label=folder.joinpath('label'),
            segment=folder.joinpath('segment')
        )

        return folders

    def mk_image_from_src(self):
        ''' Make the raw.ext and image.jpg for the image '''
        if self.src is None:
            print('E: Can not make image from src is None')
            return

        image = src2image(self.src)
        image.save(self.folders['folder'].joinpath(self.name))
        image.save(self.folders['folder'].joinpath('raw.jpg'))

        return image

    def resize_image(self):
        ''' Resize the image '''
        if self.image is None:
            print('E: Can not resize image for None')
            return

        image = self.image
        # size is in the tuple of (width, height)
        size = image.size
        print('Image raw size is {}'.format(size))

        # Resize to width=400px
        w = 400
        k = size[0] / w
        new_size = (w, int(size[1]/k))
        img = image.resize(new_size)
        img.save(self.folders['folder'].joinpath('resize.jpg'))
        print('Resize the image into {}'.format(new_size))

        # Resize to thumbnail size, width=100px
        w = 100
        k = size[0] / w
        new_size = (w, int(size[1]/k))
        img = image.resize(new_size)
        img.save(self.folders['folder'].joinpath('thumb.jpg'))
        print('Resize the image into {}'.format(new_size))

        return
