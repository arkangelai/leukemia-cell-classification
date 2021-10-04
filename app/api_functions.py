from YOLO_pred_functions import *
import requests
import os
import shutil
from yolov3 import utils
from yolov3.yolov4 import Create_Yolo
from yolov3.configs import *
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import sys

model_folder = f"models/leukemia_checkpoints_c/{TRAIN_MODEL_NAME}"
screening_model_path = 'models/leukemia-screening.h5'
classification_model_path = "models/leukemia-classification.h5"
screening_th = 0.4
classification_th = 0.5
with tf.device('/cpu:0'):
    screening_model = load_model(screening_model_path)
    classification_model = load_model(classification_model_path)
yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES="ALL_IDB1.names")    
model_folder = f"models/leukemia_checkpoints_c/{TRAIN_MODEL_NAME}"

def get_image(uri):
    size=224
    uri_old = uri.split('?')[0]
    image_name = uri_old.split('/')[-1]
    image_fmt = image_name.split('.')[-1]
    response = requests.get(uri, stream=True)
    
    print(uri)
    sys.stdout.flush()

    print(image_name)
    sys.stdout.flush()

    with open(os.getcwd()+'/'+image_name, 'wb') as file:
        response.raw.decode_content = True
        shutil.copyfileobj(response.raw, file)
    del response
    
    return image_name

def api_predict(json_data):
    uri = json_data["image"]
    image_file = get_image(uri)
    boxes, scores, clas_p = compute_prediction(model=yolo, model_folder=model_folder, screening_model=screening_model,
                                               classification_model=classification_model, screen_th=0.4,
                                               train_classes="ALL_IDB1.names", score_th=0.3, iou_th=0.3, 
                                               subject="",  sample_image=image_file, 
                                               divs=3)
    return {"boxes": [box.tolist() for box in boxes], "scores": list(scores), "classification": clas_p.tolist()}
                                               
