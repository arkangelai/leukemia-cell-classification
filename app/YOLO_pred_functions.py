import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from yolov3 import utils
from yolov3.yolov4 import Create_Yolo
from yolov3.configs import *
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from preprocessing_functions import *
import seaborn as sns
import tensorflow_addons as tfa
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
from sklearn.metrics import classification_report
from tensorflow.keras.applications.imagenet_utils import preprocess_input



def preprocess_image_screening(img_path, size=224, from_path=True, img=[]):
    if from_path:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, (size,size), interpolation=cv2.INTER_NEAREST)
    img_dims = np.expand_dims(img_resize, axis=0)
    return preprocess_input_re(img_dims)
    
def predict_screening_score(img, model):
    img_preprocessed = preprocess_image_screening('', from_path=False, img=process_save_im(img))
    return model.predict(img_preprocessed)[0][1]
    
def predict_boxes(model, model_folder, image_path, classes, 
                  score_th, iou_th, return_boxes=True, from_path=True, image=[]):
    model.load_weights(model_folder)
    return utils.detect_image(model, image_path, '', input_size=416, show=False, 
                              CLASSES=classes, score_threshold=score_th, iou_threshold=iou_th, 
                              rectangle_colors='', return_boxes=True, from_path=from_path, image=image)
                              
def predict_boxes_batch(model, model_folder, image_paths, classes, 
                  score_th, iou_th, return_boxes=True, from_path=True, images=[]):
    model.load_weights(model_folder)
    return utils.detect_image_batched(model, image_paths, '', input_size=416, show=False, 
                              CLASSES=classes, score_threshold=score_th, iou_threshold=iou_th, 
                              rectangle_colors='', return_boxes=True, from_path=from_path, images=images)
                              
def iou(boxA, boxB):
    x1i = max(boxA[0], boxB[0])
    y1i = max(boxA[1], boxB[1])
    x2i = min(boxA[2], boxB[2])
    y2i = min(boxA[3], boxB[3])
    
    area_boxA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_boxB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    intersection_area = max(0, x2i - x1i) * max(0, y2i - y1i)
    union_area = area_boxA + area_boxB - intersection_area
    return intersection_area / union_area
    
def recall_n(truth_boxes, pred_boxes, iou_th, not_string=True):
    tp = 0; fn = 0
    for truth in truth_boxes:
        if not_string:
            tbox =[int(coor) for coor in truth.split(',')[:4]]
        else:
            tbox = truth[:4]
        fn+=1
        for prediction in pred_boxes:
            pbox = [int(coor) for coor in prediction[:4]]
            if iou(tbox, pbox) >= iou_th:
                tp+=1
                fn-=1
                break
    return tp, fn
    
def precision_n(truth_boxes, pred_boxes, iou_th, not_string=True):
    tp = 0; fp = 0
    for prediction in pred_boxes:
        pbox = [int(coor) for coor in prediction[:4]]
        fp+=1
        for truth in truth_boxes:
            if not_string:
                tbox =[int(coor) for coor in truth.split(',')[:4]]
            else:
                tbox = truth[:4]
            if iou(tbox, pbox) >= iou_th:
                tp+=1
                fp-=1
                break
    return tp, fp
    
def get_ground_truth_boxes(dataframe, sample_image):
    sample_boxes = dataframe.query("im_name==@sample_image")
    bboxes = sample_boxes
    #print(len(bboxes))
    boxes = []
    labels = []
    for i in range(len(bboxes)):
        box = bboxes.iloc[i]
        x1,y1,x2,y2 = compute_bbox(box.x1, box.y1, box.w, box.h)
        boxes.append([x1,y1,x2,y2])
        labels.append(box.label)
    return boxes, labels

def get_image_info_boxes(info_file, init_idx, end_idx):
    images_info = info_file[init_idx:end_idx]
    paths = []
    im_boxes = []
    for image_info in images_info:
        image_info = image_info.split()
        image_path = image_info[0]
        truth_boxes = image_info[1:]
        paths.append(image_path)
        im_boxes.append(truth_boxes)
    return paths, im_boxes

def preprocess_input_re(x, data_format=None, mode='tf'):
    return preprocess_input(x, data_format, mode)
    
    
def classify_single_box(box, img, classification_model):
    x1,y1,x2,y2 = [int(b) for b in box[:4]]
    im_crop = img[y1:y2, x1:x2]
    im_prep = preprocess_image_screening("", from_path=False, img=im_crop)
    res = classification_model.predict(im_prep).squeeze()[1]>0.5
    #cv2.imwrite(str(np.random.randint(843,84938))+"-"+str(res)+".jpg",cv2.cvtColor(im_crop, cv2.COLOR_RGB2BGR))
    return res
    
def get_batch_images_boxes(grid_struc, model, model_folder, screening_model, classification_model, screen_th, 
                           train_classes, score_th, iou_th, cords_disp, divs=3, batch_size=12):
    pred_boxes = []
    pred_scores = []
    batches = []
    batch = []
    batch_num = 0
    for i in range(divs**2):
        elem = grid_struc[str(i)]
        screen = predict_screening_score(elem['Image'], screening_model)
        if screen>=screen_th:
            if batch_num>=(batch_size):
                batches.append(batch)
                batch = []
                batch_num = 0
            batch.append(elem)
            batch_num+=1
    if len(batch)!=0:
        batches.append(batch)

    pred_boxes = []
    pred_scores = []
    pred_class = []
    for batch in tqdm(batches):
        images = []
        temps = []
        for elem in batch:
            images.append(elem['Image'])
            temps.append(elem["temp"])
        bboxes = predict_boxes_batch(model, model_folder, temps, train_classes, 
                  score_th, iou_th, return_boxes=True, from_path=True, images=images)
        for elem in batch:
            os.remove(elem["temp"])
        for boxes_im, elem in zip(bboxes, batch):
            if len(boxes_im) > 0:
                for box in boxes_im:
                    box_class = classify_single_box(box, elem["Image"], classification_model)
                    x1,y1,x2,y2 = [int(b) for b in box[:4]]
                    pred_boxes.append(displace_bbox(x1,y1,x2,y2,
                                                    -(elem['Coords']['x1']+cords_disp[0]),
                                                    -(elem['Coords']['y1']+cords_disp[1])))
                    pred_scores.append(box[4])
                    pred_class.append(box_class)
    return pred_boxes, pred_scores, pred_class
    
def get_boxes_divisions(grid_struc, model, model_folder, train_classes, score_th, iou_th, cords_disp, divs=3):
    pred_boxes = []
    pred_scores = []
    for i in tqdm(range(divs**2)):
        elem = grid_struc[str(i)]
        detection = predict_boxes(model, model_folder, '', train_classes, 
                      score_th, iou_th, return_boxes=True, from_path=False, image=elem['Image'])
        if len(detection) > 0:
            for box in detection:
                x1,y1,x2,y2 = [int(b) for b in box[:4]]
                pred_boxes.append(displace_bbox(x1,y1,x2,y2,
                                                -(elem['Coords']['x1']+cords_disp[0]),
                                                -(elem['Coords']['y1']+cords_disp[1])))
                pred_scores.append(box[4])
    return pred_boxes, pred_scores
    
def reprocess_boxes(boxes, scores, del_array):
    idx = np.where(del_array==del_array.min())[0].tolist()
    return np.array(boxes)[idx], np.array(scores)[idx], idx
    
def non_max_supression(boxes, scores, iou_th):
    deleted = np.zeros((len(boxes), 1))
    for i in range(len(boxes)):
        box_i = boxes[i]
        score_i = scores[i]
        for j in range(len(boxes)):
            box_j = boxes[j]
            score_j = scores[j]
            if iou(box_i, box_j) > iou_th:
                if score_j > score_i:
                    deleted[i] = 1
    return reprocess_boxes(boxes, scores, deleted)
    
def compute_prediction(model, model_folder, screening_model, classification_model, screen_th, train_classes,
                       score_th, iou_th, subject, 
                       sample_image, divs=3):
    grid_struc, grid_struc_disp = full_preprocess_pipeline_no_gt("", "",
                                                      sample_image, divs=divs)
    sample_h, sample_w, _ = grid_struc['0']['Image'].shape
    cords_disp = [sample_w//(divs), sample_h//(divs)] 
    pred_boxes, pred_scores, pred_class = get_batch_images_boxes(grid_struc, model, model_folder, 
                                                  screening_model, classification_model, screen_th, 
                                                  train_classes, score_th, iou_th, [0,0])
    pred_boxes_disp, pred_scores_disp, pred_class_disp = get_batch_images_boxes(grid_struc_disp, model, model_folder, 
                                                  screening_model, classification_model, screen_th, 
                                                  train_classes, score_th, iou_th, cords_disp)
    boxes = pred_boxes+pred_boxes_disp
    scores = pred_scores+pred_scores_disp
    classifications = pred_class+pred_class_disp
    
    boxes, scores, idx = non_max_supression(boxes, scores, iou_th)
    
    if len(boxes)!=0:
        return boxes, scores, np.array(classifications)[idx]
    else:
        return [], []
