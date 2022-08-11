# Image folder document

The image folder contains sub-folders.
The sub-folders are images.

The image folders are named as the image. The one of them contains

---
- Image
    - [name], the raw image, the extent name is as the same as the image;
    - raw.jpg, the raw image with jpeg format;
    - resize.jpg, the well formatted image;
    - thumb.jpg, the small thumb version of the image.

---
- Label
    - image-label.jpg, the image with label boxes;
    - label/, the folder of image patches within the label boxes;
    - label/[%1]-[%2].jpg
        - %1, the label index;
        - %2, the label string.
        - For example, 1-person.jpg

---
- Segment
    - image-segment.jpg, the image segmentation;
    - mask/, the 255 mask of the segments;
    - mask/[%1].jpg
        - %1, the segment index;
        - For example, 1.jpg