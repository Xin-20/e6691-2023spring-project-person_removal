'''
Encapsulate the works in 'person_segmentation_demo.ipynb' as functions.
'''
import numpy as np
import torch
import matplotlib.pyplot as plt

def yolo_detection(yolov8, image):
    '''
    Use YOLOv8 to detect persons.
    Args:
    - yolov8, yolov8 model instance
    - image, cv2 image
    Return:
    - detection_boxes, bounding boxes contains persons in xyxy format
    '''
    # make prediction for persons (cls and bbx)
    preds = yolov8(image, classes=0)
    
    # unzip cls and bbx
    for pred in preds: 
        detection_boxes = [[int(x[0]), int(x[1]), int(x[2]), int(x[3])] for x in list(pred.boxes.xyxy)]
    
    return detection_boxes

def quick_mask(image, person_boxes, verbose=False):
    '''
    Use bounding boxes to do quick segmentation.
    Args:
    - image, cv2 image
    - person_boxes, bounding boxes contains persons in xyxy format
    - verbose, whether print mask
    Return:
    - merged_masks, single channel, binary mask, same size as the input.
    '''
    # mask generation (from bounding boxes)
    h = np.shape(image)[0] # size of masks
    w = np.shape(image)[1]
    merged_masks = np.zeros((h, w))  
    for box in person_boxes:
        mask = np.zeros((h, w))
        mask[box[1]:box[3]+1, box[0]:box[2]+1] = 1
        merged_masks = np.maximum(merged_masks, mask)
    if verbose:
        plt.figure(figsize=(5, 5))
        plt.imshow(merged_masks)
        plt.title("Quick Mask")
        
    return merged_masks

def fine_mask(predictor, image, person_boxes, verbose=False):
    '''
    Use SAM to do fine segmentation.
    Args:
    - predictor: SAM predictor instance
    - image, cv2 image
    - person_boxes, bounding boxes contains persons in xyxy format
    - verbose, whether print mask
    Return:
    - merged_masks, single channel, binary mask, same size as the input.
    '''
    # bounding box pre-processing
    input_boxes = torch.tensor(person_boxes, device=predictor.device) # to tensor
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2]) # to the input frame
    
    # mask prediction
    predictor.set_image(image)
    masks, scores, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )
    
    # mask merging
    m, n = scores.cpu().numpy().shape
    h = np.shape(image)[0] # size of masks
    w = np.shape(image)[1]
    merged_masks = np.zeros((h, w))
    for i in range(m):
        for j in range(n):
            merged_masks = np.maximum(merged_masks, masks.cpu().numpy()[i,j])
    
    # mask expanding (r=15)
    enlarged_masks = np.zeros((h, w)) 
    r = 15
    for i in range(r, h-r):
        for j in range(r, w-r):
            if merged_masks[i, j] > 0:
                enlarged_masks[i-r:i+r+1, j-r:j+r+1] = 1
    
    if verbose:
        plt.figure(figsize=(5, 5))
        plt.imshow(enlarged_masks)
        plt.title("Fine Mask")

    return enlarged_masks