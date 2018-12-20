from keras.models import *
from keras.callbacks import *
import keras.backend as K
from model import *
from data import *
import cv2
import glob
import argparse

from scipy import ndimage
from skimage.measure import label, regionprops

def train(dataset_path, validation_path, weight_path):
        model = get_model()
        X, y = load_inria_person(dataset_path)
        v_x, v_y = load_inria_validation(validation_path)

        print("training")
        checkpoint_path = weight_path + "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
        #model.fit(X, y, nb_epoch=40, shuffle=True, batch_size=32, validation_split=0.1, verbose=1, callbacks=[checkpoint])
        model.fit(X, y, nb_epoch=40, batch_size=32, validation_data=(v_x, v_y), shuffle=True, verbose=1, callbacks=[checkpoint])
        score = model.evaluate(v_x, v_y, verbose=1)
        print('Total loss : ', score[0])
        print('Total accuracy: ', score[1])
        

def visualize_class_activation_map(model_path, img_path, output_path):
        model = load_model(model_path)
        
        original_img = cv2.imread(img_path, 1)
        reshaped_img = cv2.resize(original_img, (224, 224))
        width, height, _ = original_img.shape

        #Reshape to the network input shape (3, w, h).
        img = np.array([np.transpose(np.float32(reshaped_img), (2, 0, 1))])
        
        #Get the 512 input weights to the softmax.
        class_weights = model.layers[-1].get_weights()[0]
        final_conv_layer = get_output_layer(model, "conv5_3")
        get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
        [conv_outputs, predictions] = get_output([img])
        conv_outputs = conv_outputs[0, :, :, :]

        #Create the class activation map.
        cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])
        for i, w in enumerate(class_weights[:, 1]):
                cam += w * conv_outputs[i, :, :]
        print ("predictions", predictions)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        img = heatmap*0.5 + original_img

        boxes = generate_bbox(cam)
        for b in boxes:
            bbox = b.bbox
            min_x = bbox[1]
            min_y = bbox[0]
            max_x = bbox[3]
            max_y = bbox[2]
            rect = cv2.rectangle(img, (min_x, min_y), (max_x, max_y), color=(0, 255, 0), thickness=3)
            
        cv2.imwrite(output_path, img)

def visualize_class_activation_map_all(model_path, imgs_path, output_path):
    model = load_model(model_path)

    img_num = 0
    imgs = glob.glob(imgs_path + "/*.png")
    img_max = len(imgs)

    for img_path in imgs:
        img_num = img_num + 1

        original_img = cv2.imread(img_path, 1)
        reshaped_img = cv2.resize(original_img, (224, 224))
        width, height, _ = original_img.shape

        #Reshape to the network input shape (3, w, h).
        img = np.array([np.transpose(np.float32(reshaped_img), (2, 0, 1))])
    
        #Get the 512 input weights to the softmax.
        class_weights = model.layers[-1].get_weights()[0]

        final_conv_layer = get_output_layer(model, "conv5_3")
        get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
        [conv_outputs, predictions] = get_output([img])
        conv_outputs = conv_outputs[0, :, :, :]
        #Create the class activation map.

        cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])
        for i, w in enumerate(class_weights[:, 1]):
                cam += w * conv_outputs[i, :, :]
        
        print ("predictions", predictions)
        
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        img = heatmap*0.5 + original_img

        boxes = generate_bbox(cam)
        for b in boxes:
            bbox = b.bbox
            min_x = bbox[1]
            min_y = bbox[0]
            max_x = bbox[3]
            max_y = bbox[2]
            rect = cv2.rectangle(img, (min_x, min_y), (max_x, max_y), color=(0, 255, 0), thickness=3)

        result = 'negative'
        if predictions[0,1] >= 0.5:
            result = 'positive'
        cv2.imwrite(output_path + "/heatmap_" + result + "_" + os.path.basename(img_path), img)
        print ("{0}/{1} heatmap saved".format(img_num, img_max))

def test_acc(model_path):
    model = load_model(model_path)

    v_x, v_y = load_inria_validation('/home/dl5/team9/INRIAPerson/Train/')
    score = model.evaluate(v_x, v_y, verbose=1)
    print('training set Total loss : ', score[0])
    print('training set Total accuracy: ', score[1])

    v_x, v_y = load_inria_validation('/home/dl5/team9/INRIAPerson/Test/')
    score = model.evaluate(v_x, v_y, verbose=1)
    print('test set Total loss : ', score[0])
    print('test set Total accuracy: ', score[1])

def generate_bbox(cam):
    labeled, nr_objects = ndimage.label(cam > 0.2)
    props = regionprops(labeled)
    return props

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type = bool, default = False, help = 'Train the network or visualize a CAM')
    parser.add_argument("--image_path", type = str, help = "Path of an image to run the network on")
    parser.add_argument("--output_path", type = str, default = "heatmap.jpg", help = "Path of an image to run the network on")
    parser.add_argument("--model_path", type = str, help = "Path of the trained model")
    parser.add_argument("--dataset_path", type = str, help = \
        'Path to image dataset. Should have pos/neg folders, like in the inria person dataset. \
        http://pascal.inrialpes.fr/data/human/')
    parser.add_argument("--validation_path", type = str, help = "path to validation dataset")
    parser.add_argument("--weight_path", type=str)
    parser.add_argument("--visualize", type = bool, default = False)
    parser.add_argument("--test", type = bool, default = False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.train:
        train('/home/dl5/team9/INRIAPerson/Train/', '/home/dl5/team9/INRIAPerson/Test/', args.weight_path)
    elif args.visualize:
        visualize_class_activation_map_all(args.model_path, '/home/dl5/team9/INRIAPerson/Test/pos', args.output_path)
    elif args.test:
        test_acc(args.model_path)
    else:
        visualize_class_activation_map(args.model_path, args.image_path, args.output_path)
