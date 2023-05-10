'''
Pipeline related functions
'''
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from .person_segmentation import detr_detection, quick_mask, fine_mask
from .misc import to_PIL_image, to_PIL_mask, to_masked_image
from transformers import DetrImageProcessor, DetrForObjectDetection, pipeline
from segment_anything import SamPredictor, sam_model_registry
from diffusers import StableDiffusionInpaintPipeline

def prepare_pipeline():
    '''
    Model instantiation.
    Return: a tuple of models.
    - DETR processor
    - DETR model
    - SAM model
    - GPT2 model
    - SD model
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detr_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
    detr_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50') # DETR instantiation
    model_type = "vit_h"
    sam_checkpoint = os.getcwd()+"\model_ckpt\sam_vit_h_4b8939.pth"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_model = SamPredictor(sam) # SAM predictor instantiation
    caption_model = pipeline("image-to-text",model="ydshieh/vit-gpt2-coco-en")
    inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16
    ).to(device) # Stable Diffusion pipeline instantiation
    return (detr_processor, detr_model, sam_model, caption_model, inpaint_model)

def person_removal(image, model, mask="quick", prompt="gen", verbose=False):
    '''
    Remove person from image.
    Args:
    - image, cv2 image, with (h, w, c).
    - model, a tuple of models.
    - mask, select between quick mas and fine mask
    - prompt, select between generative and fixed prompt
    - verbose, whether to show intermediate information or not.
    Return:
    
    '''
    # transform image to tensor
    X = torch.tensor(image)
    X = torch.einsum('hwc->chw', X)
    
    # detection and segmentation
    time_start = time.time() # record process starting time
    person_boxes = detr_detection(model[0], model[1], X)
    if mask=="quick":
        mask = quick_mask(image, person_boxes, verbose)
    elif mask=='fine':
        mask = fine_mask(model[2], image, person_boxes, verbose)
    else:
        raise Exception("Sorry, `mask` option must be `quick` or `fine`.")
    
    # transform image
    image_PIL, original_size = to_PIL_image(image)
    
    # transform mask
    mask_PIL = to_PIL_mask(image, mask)
    
    # create masked image
    masked_image = to_masked_image(image, person_boxes, verbose)
    
    # caption (with masked_image)
    if prompt=="fixed":
        filtered_caption = ['background']
    elif prompt=="gen":
        filtered_caption = ['background']
        caption = model[3](masked_image)[0]['generated_text']
        stop_words = set(nltk.corpus.stopwords.words('english'))
        caption = nltk.tokenize.word_tokenize(caption)
        for word in caption:
            if word not in stop_words:
                filtered_caption.append(word)
    else:
        raise Exception("Sorry, `prompt` option must be `fixed` or `gen`.")
            
    # inpainting (image_PIL, mask_PIL)
    inpainted_image = model[4](
        prompt=filtered_caption,
        image=image_PIL,
        mask_image=mask_PIL
    ).images[0]
    time_duration = time.time() - time_start # count duration
    
    if verbose:
        print("Inpainting finished in {:.2f}s.".format(time_duration))
        print(filtered_caption)
        plt.figure()
        plt.imshow(image_PIL.resize(original_size))
        plt.axis("off")
        plt.figure()
        plt.imshow(inpainted_image.resize(original_size))
        plt.axis("off")
        plt.show()
        
    return inpainted_image