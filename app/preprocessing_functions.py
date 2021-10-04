import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.patches as patches
import copy

def generate_image_crops(crop_coords, image_crop, divs=6, init_disp=(0,0)):
    x1d,y1d,x2d,y2d = crop_coords
    h,w,_ = image_crop.shape
    cx = w//divs
    cy = h//divs
    image_crop = image_crop[init_disp[1]:, init_disp[0]:]
    grid_dict = dict.fromkeys([str(i) for i in range(divs**2)], None)

    count=0
    for i in range(divs):
        cy_i=(i*cy)
        if i<divs-1:
            cy_i1=((i+1)*cy)
        else:
            cy_i1=h
        for j in range(divs):
            cx_j=(j*cx)
            if j<divs-1:
                cx_j1=((j+1)*cx)
            else:
                cx_j1=w 
            crop_path = create_temporal_im(image_crop[cy_i:cy_i1, cx_j:cx_j1])
            saved_dict = {'Image':image_crop[cy_i:cy_i1, cx_j:cx_j1],
                         'Coords':{'x1':cx_j,'y1':cy_i,'x2':cx_j1,'y2':cy_i1},      
                         'bboxes':[],
                         "temp":crop_path}
            grid_dict[str(count)]=saved_dict
            count+=1
    return grid_dict
    
def displace_bbox(x1,y1,x2,y2,xc,yc):
    return x1-xc, y1-yc, x2-xc, y2-yc
    
def area(x1,y1,x2,y2):
    return (x2-x1)*(y2-y1)
    


def search_bbox_idx_translate(bbox, partition_struc):
    x1,y1,x2,y2=bbox
    dict_keys = [key for key in partition_struc.keys()]
    #print(dict_keys)
    for key in dict_keys:
        x1_c = partition_struc[key]['Coords']['x1']
        y1_c = partition_struc[key]['Coords']['y1']
        x2_c = partition_struc[key]['Coords']['x2']
        y2_c = partition_struc[key]['Coords']['y2']
        #print(x1>x1_c, y1>y1_c, x2<x2_c, y2<y2_c)
        
        #standard case: object inside image
        if (x1>x1_c and y1>y1_c and x2<x2_c and y2<y2_c):
            new_bbox = displace_bbox(x1,y1,x2,y2,x1_c,y1_c)
            partition_struc[key]['bboxes'].append(new_bbox)
            #print('box belongs to: ', key)
            return
            
            
def compute_bbox(x1,y1,w,h):
    return x1,y1,x1+w,y1+h
            
def search_bbox_idx_translate_chop(bbox, partition_struc):
    x1,y1,x2,y2=bbox
    dict_keys = [key for key in partition_struc.keys()]
    #print(dict_keys)
    for key in dict_keys:
        x1_c = partition_struc[key]['Coords']['x1']
        y1_c = partition_struc[key]['Coords']['y1']
        x2_c = partition_struc[key]['Coords']['x2']
        y2_c = partition_struc[key]['Coords']['y2']
        #print(x1>x1_c, y1>y1_c, x2<x2_c, y2<y2_c)
            #define cases for chopped objects
        if (x1<x1_c and y1>y1_c and x2<x2_c and y2<y2_c):
            new_bbox = displace_bbox(x1_c,y1,x2,y2,x1_c,y1_c)
            if area(*new_bbox) >= 0.40*area(*bbox):
                partition_struc[key]['bboxes'].append(new_bbox)
            #print('box belongs to: ', key)           
        
        if (x1>x1_c and y1<y1_c and x2<x2_c and y2<y2_c):
            new_bbox = displace_bbox(x1,y1_c,x2,y2,x1_c,y1_c)
            if area(*new_bbox) >= 0.40*area(*bbox):
                partition_struc[key]['bboxes'].append(new_bbox)
            #print('box belongs to: ', key)           
        
        if (x1>x1_c and y1>y1_c and x2>x2_c and y2<y2_c):
            new_bbox = displace_bbox(x1,y1,x2_c,y2,x1_c,y1_c)
            if area(*new_bbox) >= 0.40*area(*bbox):
                partition_struc[key]['bboxes'].append(new_bbox)
            #print('box belongs to: ', key)         
        
        if (x1>x1_c and y1>y1_c and x2<x2_c and y2>y2_c):
            new_bbox = displace_bbox(x1,y1,x2,y2_c,x1_c,y1_c)
            if area(*new_bbox) >= 0.40*area(*bbox):
                partition_struc[key]['bboxes'].append(new_bbox)
            #print('box belongs to: ', key)
            
def add_displace_boxes(parasites_bbox, crop_struc, crop_disp, init_disp):
    grid_dict = copy.deepcopy(crop_struc)
    for i in range(len(parasites_bbox)):
        parasite = parasites_bbox.iloc[i]
        bbox = compute_bbox(parasite.x1, parasite.y1, parasite.w, parasite.h)
        bbox_disp_a = displace_bbox(bbox[0], bbox[1], bbox[2], bbox[3],
                                 crop_disp[0], crop_disp[1])
        bbox_disp = displace_bbox(bbox_disp_a[0], bbox_disp_a[1], 
                                  bbox_disp_a[2], bbox_disp_a[3],
                                 init_disp[0], init_disp[1])
        search_bbox_idx_translate(bbox_disp, grid_dict)
        search_bbox_idx_translate_chop(bbox_disp, grid_dict)
    return grid_dict
    
def get_box_struc_img(image, boxes_dataframe, init_disp=(0,0), divs=6):
    image_crop = image
    x1c=0; y1c=0; x2c=0; y2c=0
    h,w,_ = image_crop.shape
    crops_struct = generate_image_crops(crop_coords=[x1c,y1c,x2c,y2c], image_crop=image_crop, 
                                        divs=divs, init_disp=init_disp)
    grid_dict = add_displace_boxes(boxes_dataframe, crops_struct, crop_disp=[x1c,y1c], init_disp=init_disp)
    return grid_dict
    
def get_box_struc_img_no_gt(image, init_disp=(0,0), divs=3):
    image_crop = image
    x1c=0; y1c=0; x2c=0; y2c=0
    h,w,_ = image_crop.shape
    crops_struct = generate_image_crops(crop_coords=[x1c,y1c,x2c,y2c], image_crop=image_crop, 
                                        divs=divs, init_disp=init_disp)
    return crops_struct
    
def process_image(dataset_path, subject_folder, image_name):
    print(os.path.join(dataset_path, subject_folder, image_name))
    image = cv2.imread(os.path.join(dataset_path, subject_folder, image_name))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb
    
def process_save_im(im):
    rand = np.random.randint(low=12345, high=123456)
    im_name = str(rand)+".jpg"
    cv2.imwrite(im_name, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    z = cv2.imread(im_name)
    os.remove(im_name)
    return cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
    
def create_temporal_im(im):
    rand = np.random.randint(low=12345, high=123456)
    im_name = str(rand)+".jpg"
    cv2.imwrite(im_name, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    return im_name
    
def full_preprocess_pipeline(dataset_path, subject_folder, image_name,
                 annot_file, classname, divs=3):
    image_rgb = process_image(dataset_path, subject_folder, image_name)
    parasites_anot = annot_file.query("im_name==@image_name").query("label==@classname")
    #if len(parasites_anot)==0:
    #    return
    grid_struc = get_box_struc_img(image_rgb, parasites_anot, init_disp=(0,0), divs=divs)
    sample_h, sample_w, _ = grid_struc['0']['Image'].shape
    grid_struc_disp = get_box_struc_img(image_rgb, parasites_anot, init_disp=(sample_w//(divs), sample_h//(divs)), divs=divs)
    return (grid_struc, grid_struc_disp)
    
def full_preprocess_pipeline_no_gt(dataset_path, subject_folder, image_name, divs=3):
    image_rgb = process_image(dataset_path, subject_folder, image_name)
    grid_struc = get_box_struc_img_no_gt(image_rgb, init_disp=(0,0), divs=divs)
    sample_h, sample_w, _ = grid_struc['0']['Image'].shape
    grid_struc_disp = get_box_struc_img_no_gt(image_rgb, init_disp=(sample_w//(divs), sample_h//(divs)), divs=divs)
    return (grid_struc, grid_struc_disp)                 

