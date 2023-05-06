'''
miscellaneous functions
'''
import numpy as np
from PIL import Image

def to_PIL(image, mask):
    '''
    Convert cv2 images to PIL images.
    Args:
    - image, cv2 image
    - mask, single channel, binary mask
    Return:
    - resized image
    - resized mask (RGB channel)
    - original size of image (h, w)
    '''
    img = image.astype(np.uint8)
    image_PIL = Image.fromarray(img)
    size = image_PIL.size # original size of image
    
    msk = np.zeros_like(image, dtype=np.uint8)
    msk[mask > 0] = (255, 255, 255)
    mask_PIL = Image.fromarray(msk)
    
    img[mask > 0] = (
        np.average(img[:,:,0]),
        np.average(img[:,:,1]),
        np.average(img[:,:,2])
    )
    masked_image_PIL = Image.fromarray(img)
    
    resize = (512, 512)
    
    return image_PIL.resize(resize), mask_PIL.resize(resize), masked_image_PIL, size