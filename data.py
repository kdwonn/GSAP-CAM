import cv2
import glob
import os
import numpy as np
from keras.utils.np_utils import to_categorical

def load_inria_person(path):
    pos_path = os.path.join(path, "pos")
    neg_path = os.path.join(path, "neg")
    pos_images = [cv2.resize(cv2.imread(x), (224, 224)) for x in glob.glob(pos_path + "/*.png")]
    pos_images = [np.transpose(img, (2, 0, 1)) for img in pos_images]
    neg_images = [cv2.resize(cv2.imread(x), (224, 224)) for x in glob.glob(neg_path + "/*.png")]
    neg_images = [np.transpose(img, (2, 0, 1)) for img in neg_images]

    y = [1] * len(pos_images) + [0] * len(neg_images)
    y = to_categorical(y, 2)
    X = np.float32(pos_images + neg_images)
    
    return X, y

def load_inria_validation(validation_path):
    v_pos_path = os.path.join(validation_path, "pos")
    v_neg_path = os.path.join(validation_path, "neg")
    v_pos_images = [cv2.resize(cv2.imread(x), (224, 224)) for x in glob.glob(v_pos_path + "/*.png")]
    v_pos_images = [np.transpose(img, (2, 0, 1)) for img in v_pos_images]
    v_neg_images = [cv2.resize(cv2.imread(x), (224, 224)) for x in glob.glob(v_neg_path + "/*.png")]
    v_neg_images = [np.transpose(img, (2, 0, 1)) for img in v_neg_images]

    v_y = [1] * len(v_pos_images) + [0] * len(v_neg_images)
    v_y = to_categorical(v_y, 2)
    v_X = np.float32(v_pos_images + v_neg_images)

    return v_X, v_y