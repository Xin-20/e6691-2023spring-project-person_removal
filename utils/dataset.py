'''
COCO and PennFudanPed dataset classes
'''
import numpy as np
import torch
import os
from PIL import Image

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            # original annotation use the bottom left corner as the origin
            # here we reset it to start from top left
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class (person)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # normalize the boxes
        width, height = img.size

        target = {}
        target["boxes"] = boxes / torch.tensor([width, height] * 2)
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

class CocoPersonDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_type, transform = None):
        """
        Input:
        data_dir = "" # define dataset path
        data_type = "val2017"  # or "train2017"

        Variable:
        self.ann_file: the annotation path for train/validation
        self.coco: initialized COCO api
        self.person_cat_id: the class ID for person, here is 1
        self.person_img_ids: the Image ID list(Images with person), list

        Output:
        image: numpy array, (height, width, channel)
        segmentation: list of array, eg: [array1, array2]
        """
        self.data_dir = data_dir
        self.data_type = data_type

        # get information for train or validation set
        self.ann_file = os.path.join(data_dir, "annotations/instances_{}.json".format(data_type))

        # initialize COCO API
        self.coco = COCO(self.ann_file)

        # get the class ID for person
        self.person_cat_id = self.coco.getCatIds(catNms=["person"])

        # get the Image ID list for person
        self.person_img_ids = self.coco.getImgIds(catIds=self.person_cat_id)

        self.transfrom = transform

    def __len__(self):
        # the length of the dataset
        return len(self.person_img_ids)

    def __getitem__(self, idx):
        # load the image
        img_info = self.coco.loadImgs(self.person_img_ids[idx])[0]
        img_path = os.path.join(self.data_dir, self.data_type, img_info["file_name"])
        image = Image.open(img_path).convert('RGB')

        # get the annotation ID for all persons in this image, iscrowd = None is to make sure no overlap person
        ann_ids = self.coco.getAnnIds(imgIds=img_info["id"], catIds=self.person_cat_id, iscrowd=None)

        # get all annotation for all person in this image
        annotations = self.coco.loadAnns(ann_ids)

        # get mask list for human objs in the image
        masks = [self.coco.annToMask(ann) for ann in annotations]

        # transform image and mask to the same size
        if self.transfrom:
            image = self.transfrom(image)
            # channel, height, width
            image = image.numpy()
            # height, width, channel
            image = image.transpose(1, 2, 0)
            
            # for each obj mask, we make it PIL format and transform it as we do for image to the same size
            for m in range(len(masks)):
                masks[m] = Image.fromarray(masks[m])
                masks[m] = self.transfrom(masks[m]).numpy()
                masks[m] = masks[m].squeeze()

            # create a mask with all 0 that have the same size as image
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            for m in masks:
            # add the each obj's mask to the mask
                mask = np.maximum(mask, m)

        return image, mask