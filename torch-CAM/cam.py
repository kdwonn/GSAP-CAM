import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import glob
import torchvision.datasets.voc as VOC
from torchvision import datasets, transforms
from tqdm import tqdm

import cv2
from scipy import ndimage
from skimage.measure import label, regionprops
import model as cls_model
import data

import os

def load_tensors(path_regex):
    """
        Load a collection of tensors that match the regular expression path
        The individual tensors are loaded in CPU
        Returns a concatenated tensors of all the loaded tensors.
    """
    tensors = None

    files_list = glob.glob(path_regex)
    sortfunc = lambda x: int(os.path.basename(x)[6:-3])
    files_list.sort(key=sortfunc)
    print(files_list)

    for tensor_filename in tqdm(files_list, ncols=50):
        tensor = torch.load(tensor_filename, map_location='cuda:0')
        if not tensors:
            tensors = tensor
        else:
            for k in tensors:
                tensors[k] = torch.cat((tensors[k], tensor[k]))
        del tensor
    for k in tensors:
        print("Key {} has shape {}".format(k, tensors[k].shape))
    return tensors

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def cam(dataset_val, valid_data):
    model = torch.load('./best_model.pt', map_location='cuda:0')
    weight = model.fc.weight.detach().cpu().numpy()

    features, classes = valid_data['features'], valid_data['targets']

    features = features.cpu().numpy()
    classes = classes.cpu().numpy()

    get_required = lambda x: np.where(x == 1)

    over_dataset_mean_iou = 0.0

    for nth_batch in range(features.shape[0]):
    #for nth_batch in range(10):
        f = features[nth_batch]
        c = classes[nth_batch]
        original_img, anno = dataset_val[nth_batch]
        original_img = np.array(original_img)
        h, w = anno['size']['height'], anno['size']['width']
        required_c = get_required(c)[0]
        
        filename = ''
        sum_cam = np.zeros((w, h))

        obj_list = [{'bndbox':i['bndbox'], 'class':data.classes.index(i['name'])} for i in anno['object']]

        over_cls_mean_iou = 0.0

        for nth_cls in range(len(required_c)):
        #for nth_cls in range(0):
            cls_num = required_c[nth_cls]
            filename += '_' + data.classes[cls_num]
            cls_weight = weight[cls_num] #2048 array
            
            cls_cam = ((f.T * cls_weight).T).sum(axis=0)
            cls_cam /= np.max(cls_cam)
            cls_cam = cv2.resize(cls_cam, (w, h))

            sum_cam += cls_cam
            
            labeled, nr_objects = ndimage.label(cls_cam > 0.2)
            props = regionprops(labeled)
            
            mean_iou = 0.0
            for b in props:
                bbox = b.bbox
                left, right, bottom, top = bbox[1], bbox[3], bbox[0], bbox[2]
                bbox_predict = {'x1':left, 'y1':top, 'x2':right, 'y2':bottom}

                bbox_target_list = [o['bndbox'] for o in obj_list if (o['class'] == cls_num)]
                largest_iou = 0.0
                for bbox_target in bbox_target_list:
                    bbox_target = {'x1':bbox_target['xmin'], 
                    'x2':bbox_target['xmax'], 'y1':bbox_target['ymax'], 'y2':bbox_target['ymin']}
                    iou = get_iou(bbox_predict, bbox_target)
                    largest_iou = iou if iou >= largest_iou else largest_iou

                mean_iou += largest_iou
            
            mean_iou /= len(props)
            over_cls_mean_iou += mean_iou
        
        over_cls_mean_iou /= len(required_c)
        over_dataset_mean_iou += over_cls_mean_iou
 
        sum_cam /= np.max(sum_cam)
        original_img = original_img[:,:,::-1].copy()

        heatmap = cv2.applyColorMap(np.uint8(255*sum_cam), cv2.COLORMAP_JET)
        heatmap[np.where(sum_cam<0.2)] = 0
        img_with_heat = heatmap*0.5 + original_img

        cv2.imwrite('./cams2/' + str(nth_batch)+ '_'+ anno['filename'][:-4] + filename + '.jpg', img_with_heat)
    
    over_dataset_mean_iou /= features.shape[0]
    print('mean IoU for validation dataset : {:.5f}'.format(over_dataset_mean_iou))
        
#         displaying_img=mpimg.imread(str(nth_batch)+'.jpg')
#         plt.imshow(displaying_img)
#         plt.show()

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner, xmin ymax
        the (x2, y2) position is at the bottom right corner xmax ymin
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    torch.manual_seed(22)
    device = torch.device("cuda:0")
    kwargs = {'num_workers':0, 'pin_memory':False}

    def target_transformer(t):
        pre_t = data.preprocess_target(t)
        annotation = pre_t['annotation']
        return {'filename' : annotation['filename'], 'size' : annotation['size'], 'object' : annotation['object']}

    dataset_val = VOC.VOCDetection(root='./voc', image_set='val', target_transform = target_transformer, download=True)
    valid_data = load_tensors('./features_valid/valid_*.pt')
    cam(dataset_val, valid_data)
    
    

if __name__ == '__main__':
    main()