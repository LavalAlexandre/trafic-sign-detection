# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:11:24 2024

@author: MEI Yiguang
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np
from src.model import model
import pandas as pd
import os
from tqdm import tqdm
def pyramid(image, scale=1.1, min_size_ratio=0.1):
    yield image
    min_height = int(image.shape[0] * min_size_ratio)
    min_width = int(image.shape[1] * min_size_ratio)
    while True:
        w = int(image.shape[1] / scale)
        image = cv2.resize(image, (w, int(w * image.shape[0] / image.shape[1])))
        if image.shape[0] < min_height or image.shape[1] < min_width:
            break
        yield image

def sliding_window(image, step_size_ratio,window_width):
    height, width = image.shape[:2]
    
    #print("imagesize:",height,width)
    window_width = window_width 
    window_height = window_width
    step_size = 50
    #print("windowsize:",window_width,window_height)
    for y in range(0, height - window_height, step_size):
        for x in range(0, width - window_width, step_size):
            yield (x, y, (window_width, window_height), image[y:y + window_height, x:x + window_width])
        
        

        

def detect_traffic_signs(image, model, step_size_ratio=0.5, min_window_size_ratio=0.1, max_window_size_ratio=0.25, pyramid_scale=1.1):
    detections = []
    for resized in pyramid(image, scale=pyramid_scale, min_size_ratio=0.05):  
        for (x, y, (win_width, win_height), window) in sliding_window(resized, step_size_ratio,int(image.shape[1] * 0.1)):
            if window.shape[0] != win_height or window.shape[1] != win_width:
                continue

            prediction = model.predict_window(window)
            prediction_probabilities = model.predict_proba_window(window)
            max_probabilities = np.max(prediction_probabilities, axis=1)
            
            if prediction != 'none' and max_probabilities[0] > 0.9:  # Assuming '1' indicates a traffic sign
                x_orig = int(x * (image.shape[1] / resized.shape[1]))
                y_orig = int(y * (image.shape[0] / resized.shape[0]))
                w_orig = int(win_width * (image.shape[1] / resized.shape[1]))
                h_orig = int(win_height * (image.shape[0] / resized.shape[0]))
                prob = max_probabilities[0]
                detections.append((x_orig, y_orig, x_orig + w_orig, y_orig + h_orig, prediction[0], prob))
    
    print(detections)
    return detections

###without piramid
def sliding_window_without_piramid(image, step_size_ratio, min_window_size_ratio, max_window_size_ratio):
    height, width = image.shape[:2]
    print("imagesize:",width,height)
    min_window_width = int(width * min_window_size_ratio)
    max_window_width = int(width * max_window_size_ratio)
    size_ratio=max_window_size_ratio-0.1
    #scale_factor = 0.9
    
    window_width = int( max_window_width*size_ratio)
    window_height = window_width
    
    
    while window_width >= 150:  
        print("windowsize:",window_width,window_height)
        #step_size = 50
        step_size = int(window_width * step_size_ratio)
        print("step_size:",step_size)
        for y in range(0, height - window_height, step_size):
            for x in range(0, width - window_width, step_size):
                yield (x, y, (window_width, window_height), image[y:y + window_height, x:x + window_width])
        
        size_ratio=size_ratio-0.1
        window_width = int( max_window_width*size_ratio)
        window_height = window_width
        
def sliding_window_without_piramid_feu(image, step_size_ratio, min_window_size_ratio, max_window_size_ratio):
    height, width = image.shape[:2]
    print("imagesize:",width,height)
    min_window_height = int(height * min_window_size_ratio)
    max_window_height = int(height * max_window_size_ratio)
    size_ratio=max_window_size_ratio-0.1
    #scale_factor = 0.9
    
    window_height = int(max_window_height*size_ratio)
    window_width = int(window_height/2.2)
    
    
    while window_height >= 200:  
        print("windowsize:",window_width,window_height)
        #step_size = 50
        step_size = int(window_height * step_size_ratio)
        print("step_size:",step_size,int(step_size/2.2))
        for y in range(0, height - window_height, step_size):
            for x in range(0, width - window_width, int(step_size/2)):
                yield (x, y, (window_width, window_height), image[y:y + window_height, x:x + window_width])
        
        size_ratio=size_ratio-0.1
        window_height =  int(max_window_height*size_ratio)
        window_width = int(window_height/2.2)

def detect_traffic_signs_without_piramid(image, model, step_size_ratio=0.2, min_window_size_ratio=0.1, max_window_size_ratio=1.0):
    detections = []
   
    for (x, y, (win_width, win_height), window) in sliding_window_without_piramid(image, step_size_ratio,min_window_size_ratio,max_window_size_ratio):
        if window.shape[0] != win_height or window.shape[1] != win_width:
            continue

        prediction = model.predict_window(window)
        prediction_probabilities = model.predict_proba_window(window)
        max_probabilities = np.max(prediction_probabilities, axis=1)
        
        if prediction != 'none' and prediction != 'frouge' and prediction != 'forange' and prediction != 'fvert' and max_probabilities[0] > 0.9:  # Assuming '1' indicates a traffic sign
            x_orig = x
            y_orig = y
            w_orig = win_width
            h_orig = win_height
            prob = max_probabilities[0]
            detections.append((x_orig, y_orig, x_orig + w_orig, y_orig + h_orig, prediction[0], prob))
            
    for (x, y, (win_width, win_height), window) in sliding_window_without_piramid_feu(image, step_size_ratio,min_window_size_ratio,max_window_size_ratio):
        if window.shape[0] != win_height or window.shape[1] != win_width:
            continue

        prediction = model.predict_window(window)
        prediction_probabilities = model.predict_proba_window(window)
        max_probabilities = np.max(prediction_probabilities, axis=1)
        
        if (prediction == 'frouge' or prediction == 'forange' or prediction == 'fvert') and max_probabilities[0] > 0.9:  # Assuming '1' indicates a traffic sign
            x_orig = x
            y_orig = y
            w_orig = win_width
            h_orig = win_height
            prob = max_probabilities[0]
            detections.append((x_orig, y_orig, x_orig + w_orig, y_orig + h_orig, prediction[0], prob))
    
    
    print(detections)
    return detections


def non_max_suppression(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    
    x1 = boxes[:, 0].astype(int)
    y1 = boxes[:, 1].astype(int)
    x2 = boxes[:, 2].astype(int)
    y2 = boxes[:, 3].astype(int)
    probs = boxes[:, 5].astype(float)
    
    idxs = np.argsort(probs)[::-1]
    
    picked_boxes = []
    
    while len(idxs) > 0:
        i = idxs[0]
        picked_boxes.append(boxes[i])
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / ((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1))
        
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
    
    return np.array(picked_boxes)





#akdshj
#%% detection all
def detection_image(image_path, model,csv_path):
    image = cv2.imread(image_path)
    detections = detect_traffic_signs_without_piramid(image, model)
    boxes = np.array(detections)
    picked_boxes = non_max_suppression(boxes, 0.1)  
    
    results = []
    for (x1, y1, x2, y2, label, max_probabilities) in picked_boxes:
        x1, y1, x2, y2 ,max_probabilities= int(x1), int(y1), int(x2), int(y2),float(max_probabilities)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
  
        text = f"{label}: {max_probabilities:.2f}"
        # 显示文本
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)        
        results.append([image_path, x1, y1, x2, y2, max_probabilities, label])
    
    df = pd.DataFrame(results, columns=['Num img', 'Coin h-g x', 'Coin h-g y', 'Coin b-d x', 'Coin b-d y', 'Score', 'Classe'])
    df.to_csv(csv_path, index=False)

def detection_images_in_folder(folder_path, model, csv_path):
    results = []
    output_folder = os.path.join(os.getcwd(), 'output_images')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in tqdm(os.listdir(folder_path), desc="Detecting traffic signs"):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            print(filename)
            image = cv2.imread(image_path)
            detections = detect_traffic_signs_without_piramid(image, model)
            boxes = np.array(detections)
            picked_boxes = non_max_suppression(boxes, 0.1)  

            for (x1, y1, x2, y2, label, max_probabilities) in picked_boxes:
                x1, y1, x2, y2, max_probabilities = int(x1), int(y1), int(x2), int(y2), float(max_probabilities)
                results.append([filename, x1, y1, x2, y2, max_probabilities, label])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label}: {max_probabilities:.2f}"
                # 显示文本
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)   
          
            output_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_image_path, image)
         
    df = pd.DataFrame(results, columns=['Num img', 'Coin h-g x', 'Coin h-g y', 'Coin b-d x', 'Coin b-d y', 'Score', 'Classe'])
    df.to_csv(csv_path, index=False)


def select_region_and_predict(image_path, model):

    global ref_point, cropping, image, clone
    cropping = False
    ref_point = []

    def click_and_crop(event, x, y, flags, param):
        global ref_point, cropping

        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point = [(x, y)]
            cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            ref_point.append((x, y))
            cropping = False

            cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
            cv2.imshow("image", image)

    image = cv2.imread(image_path)
    clone = image.copy()

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            image = clone.copy()

        elif key == ord('c') and len(ref_point) == 2:
            roi = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            #prediction = model.predict_window(roi)
            prediction_probabilities = model.predict_proba_window(roi)

            top_indices = prediction_probabilities.argsort()[0][-3:][::-1]
            top_classes = model.classifier.classes_[top_indices]
            top_probabilities = prediction_probabilities[0][top_indices]

            cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
            text = f"Size: {ref_point[1][0] - ref_point[0][0]}x{ref_point[1][1] - ref_point[0][1]}"
            cv2.putText(image, text, (ref_point[0][0], ref_point[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            y_offset = ref_point[1][1] + 30
            for i, (cls, prob) in enumerate(zip(top_classes, top_probabilities)):
                text = f"{i+1}: {cls} ({prob:.2f})"
                cv2.putText(image, text, (ref_point[0][0], y_offset + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("image", image)
            ref_point = []


        elif key == ord('0'):
            break

    cv2.destroyAllWindows()










