'''
miscellaneous functions
'''
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def to_PIL_image(image):
    '''
    Convert cv2 images to PIL images.
    Args:
    - image, cv2 image
    Return:
    - resized image (for inpaint_model input)
    - original size of image (h, w)
    '''
    img = image.astype(np.uint8)
    image_PIL = Image.fromarray(img)
    size = image_PIL.size # original size of image
    image_PIL = image_PIL.resize((512, 512))
    
    return image_PIL, size

def to_PIL_mask(image, mask):
    '''
    Convert mask to PIL images.
    Args:
    - image, cv2 image
    - mask, single channel, binary mask
    Return:
    - resized mask
    '''
    d, w = np.shape(mask)
    mask_3channel = np.zeros_like(image, dtype=np.uint8)
    mask_3channel[mask > 0] = (255, 255, 255)
    mask_PIL = Image.fromarray(mask_3channel)
    mask_PIL = mask_PIL.resize((512, 512))
    
    return mask_PIL

def to_masked_image(image, person_boxes, verbose=False):
    '''
    Combine image and boxes to create a masked image.
    Masked image is used for caption generation.
    Args:
    - image, cv2 image
    - person_boxes, bounding boxes contains persons in xyxy format
    Return:
    - masked image
    '''
    masked_image = image
    for box in person_boxes:
        masked_image[box[1]: box[3]+1, box[0]:box[2]+1] = np.average(image[box[1]: box[3]+1, box[0]:box[2]+1])
    masked_image_PIL = Image.fromarray(masked_image)
    
    if verbose:
        plt.figure(figsize=(5, 5))
        plt.imshow(masked_image)
        plt.title("Masked Image")
    
    return masked_image_PIL

