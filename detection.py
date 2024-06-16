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

#%% detection function


"""
def pyramid(image, scale=1.5, min_size=(50, 50)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = cv2.resize(image, (w, int(w * image.shape[0] / image.shape[1])))
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image

def sliding_window(image, step_size, window_sizes):
    for window_size in window_sizes:
        for y in range(0, image.shape[0] - window_size[1], step_size):
            for x in range(0, image.shape[1] - window_size[0], step_size):
                yield (x, y, window_size, image[y:y + window_size[1], x:x + window_size[0]])

def detect_traffic_signs(image, model, win_sizes=[(50,50),(200,200),(400,400)], step_size=25, pyramid_scale=1.5):
    detections = []
    for resized in pyramid(image, scale=pyramid_scale, min_size=(50, 50)):
        for (x, y, win_size,window) in sliding_window(resized, step_size, win_sizes):
            if window.shape[0] != win_size[1] or window.shape[1] != win_size[0]:
                continue
                
           # img_window=img(window=window) 
            #print(img_window.data)
            prediction = model.predict_window(window)
            
            #print(prediction)
            
            prediction_probabilities=model.predict_proba_window(window)
            #print(model.classifier.classes_)
            #print(prediction_probabilities)
            max_probabilities = np.max(prediction_probabilities, axis=1)            
            #print(prediction)          
            if prediction!='none' and max_probabilities > 0.9 :  # Assuming '1' indicates a traffic sign
            
                
                x_orig = int(x * (image.shape[1] / resized.shape[1]))
                y_orig = int(y * (image.shape[0] / resized.shape[0]))
                w_orig = int(win_size[0] * (image.shape[1] / resized.shape[1]))
                h_orig = int(win_size[1] * (image.shape[0] / resized.shape[0]))
                print(model.classifier.classes_)
                #print(prediction_probabilities)
                print(prediction)
                prob =max_probabilities[0]
                detections.append((x_orig, y_orig, x_orig+w_orig, y_orig+h_orig,prediction[0],prob))
    
    return detections
"""
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
    
    print("imagesize:",height,width)       
    window_width = window_width 
    window_height = window_width
    step_size = 50
    print("windowsize:",window_width,window_height)
    for y in range(0, height - window_height, step_size):
        for x in range(0, width - window_width, step_size):
            yield (x, y, (window_width, window_height), image[y:y + window_height, x:x + window_width])
        
        

        

def detect_traffic_signs(image, model, step_size_ratio=0.5, min_window_size_ratio=0.1, max_window_size_ratio=0.25, pyramid_scale=1.1):
    detections = []
    for resized in pyramid(image, scale=pyramid_scale, min_size_ratio=0.05):  # 设置最小尺寸比例为1/4
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

    #scale_factor = 0.9
    
    window_width = max_window_width 
    window_height = window_width
    
    
    while window_width > min_window_width:  
        print("windowsize:",window_width,window_height)
        step_size = 50
        #step_size = int(window_width * step_size_ratio)
        print("step_size:",step_size)
        for y in range(0, height - window_height, step_size):
            for x in range(0, width - window_width, step_size):
                yield (x, y, (window_width, window_height), image[y:y + window_height, x:x + window_width])
        
        window_width =  window_width-50
        window_height = window_width
        
def sliding_window_without_piramid_feu(image, step_size_ratio, min_window_size_ratio, max_window_size_ratio):
    height, width = image.shape[:2]
    print("imagesize:",width,height)
    min_window_height = int(height * min_window_size_ratio)
    max_window_height = int(height * max_window_size_ratio)

    #scale_factor = 0.9
    
    window_height = max_window_height
    window_width = int(window_height/2.2)
    
    
    while window_height > min_window_height:  
        print("windowsize:",window_width,window_height)
        step_size = 50
        #step_size = int(window_width * step_size_ratio)
        print("step_size:",step_size)
        for y in range(0, height - window_height, step_size):
            for x in range(0, width - window_width, int(step_size/2)):
                yield (x, y, (window_width, window_height), image[y:y + window_height, x:x + window_width])
        
        window_height =  window_height-50
        window_width = int(window_height/2.2)

def detect_traffic_signs_without_piramid(image, model, step_size_ratio=0.1, min_window_size_ratio=0.1, max_window_size_ratio=0.8):
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
        
        if (prediction == 'frouge' or prediction == 'forange' or prediction == 'fvert') and max_probabilities[0] > 0.7:  # Assuming '1' indicates a traffic sign
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






import joblib
#use model.predict_window(self, window) to predict a window
seed = 42
my_model = model(seed)
my_model.classifier = joblib.load("src/models/bangerv3.pkl")
print(my_model.classifier)

#%% detection all
def detection_image(image_path, model,csv_path):
    image = cv2.imread(image_path)
    detections = detect_traffic_signs_without_piramid(image, model)
    boxes = np.array(detections)
    picked_boxes = non_max_suppression(boxes, 0.2)  
    
    results = []
    for (x1, y1, x2, y2, label, max_probabilities) in picked_boxes:        
        results.append([image_path, x1, y1, x2, y2, max_probabilities, label])
    
    df = pd.DataFrame(results, columns=['Num img', 'Coin h-g x', 'Coin h-g y', 'Coin b-d x', 'Coin b-d y', 'Score', 'Classe'])
    df.to_csv(csv_path, index=False)

def detection_images_in_folder(folder_path, model, csv_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            detections = detect_traffic_signs_without_piramid(image, model)
            boxes = np.array(detections)
            picked_boxes = non_max_suppression(boxes, 0.2)  

            for (x1, y1, x2, y2, label, max_probabilities) in picked_boxes:
                x1, y1, x2, y2, max_probabilities = int(x1), int(y1), int(x2), int(y2), float(max_probabilities)
                results.append([filename, x1, y1, x2, y2, max_probabilities, label])
          
    
    df = pd.DataFrame(results, columns=['Num img', 'Coin h-g x', 'Coin h-g y', 'Coin b-d x', 'Coin b-d y', 'Score', 'Classe'])
    df.to_csv(csv_path, index=False)

detection_images_in_folder('C:/Users/23158/Desktop/SY32PROJET/dataset2/val/images/', my_model,'detections.csv')
    
  
#%%get results

detections_path = 'detections.csv'
labels_path = 'C:/Users/23158/Desktop/SY32PROJET/dataset2/val/labels/'

detections_df = pd.read_csv(detections_path, header=None)
detections_df.columns = ['image', 'x1', 'y1', 'x2', 'y2', 'score','label']

print(detections_df)

images_path = 'C:/Users/23158/Desktop/SY32PROJET/dataset2/val/images/'
output_path = 'C:/Users/23158/Desktop/SY32PROJET/dataset2/val/output/'
os.makedirs(output_path, exist_ok=True)

for image_name in detections_df['image'].unique():
    image_path = os.path.join(images_path, image_name)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"no: {image_path}")
        continue
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1)
    ax.imshow(image_rgb)
    
   
    detections = detections_df[detections_df['image'] == image_name]
    for _, row in detections.iterrows():
        rect = plt.Rectangle((row['x1'], row['y1']), row['x2'] - row['x1'], row['y2'] - row['y1'], edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        label_x = row['x1']
        label_y = row['y2'] + 5  
        ax.text(label_x, label_y, row['label'], color='r', fontsize=12)
    

    label_file = os.path.join(labels_path, image_name.replace('.jpg', '.csv'))
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                if len(parts) < 5:
                    print(f"no: {label_file} line: {line}")
                    continue
                label = parts[4]
                if label != 'ff':
                    try:
                        x1, y1, x2, y2 = map(float, parts[0:4])
                        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='g', facecolor='none')
                        ax.add_patch(rect)
                    except ValueError as e:
                        print(f"error: {e} in {label_file} line: {line}")
    
    plt.axis('off')
    output_image_path = os.path.join(output_path, image_name)
    plt.savefig(output_image_path)
    plt.close()
    print(f"save: {output_image_path}")
#%%validation.csv
import os
import pandas as pd


val_path = 'C:/Users/23158/Desktop/SY32PROJET/dataset2/val/labels'


csv_files = [file for file in os.listdir(val_path) if file.endswith('.csv')]


dfs = []


for csv_file in csv_files:
    file_path = os.path.join(val_path, csv_file)
    try:
       
        df = pd.read_csv(file_path, header=None)
        if df.empty:
          
            continue
        
 
        image_name = os.path.splitext(csv_file)[0]
        
        df.insert(0, 'num_image', image_name + '.jpg')
       
        dfs.append(df)
    except pd.errors.EmptyDataError:
     
        print(f"'{csv_file}'empty")

if not dfs:
    print("nocsv")

else:
   
    merged_df = pd.concat(dfs, ignore_index=True)

    output_file = 'validations.csv'
    merged_df.to_csv(output_file, index=False, header=False)

    print(f"save{output_file}")
#%% analays
import pandas as pd



# Load the data
validations = pd.read_csv('validations.csv',header=None)
detections = pd.read_csv('detections.csv',header=None)

# Filter out the 'ff' labels
validations = validations[validations.iloc[:, 5] != 'ff']
detections = detections[detections.iloc[:, 6] != 'ff']



# Initialize lists to store y (ground truth) and y_pred (predictions)
y = []
y_pred = []
matching_validation_row=[]
# Define IoU function
def iou(box1, box2):
    # Unpack the coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Calculate the intersection area
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate the areas of each box
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate the union area
    union_area = box1_area + box2_area - inter_area

    # Calculate the IoU
    iou = inter_area / union_area

    return iou

# Match detections with ground truth
for _, detection_row in detections.iterrows():
    image_id_d, x_min_d, y_min_d, x_max_d, y_max_d, score_d, label_d = detection_row
    max_iou = 0
    matching_validation = 'none'

    for _, validation_row in validations.iterrows():
        image_id_v, x_min_v, y_min_v, x_max_v, y_max_v, label_v = validation_row
        
        if image_id_v == image_id_d and label_v == label_d:
            iou_value = iou((x_min_v, y_min_v, x_max_v, y_max_v), (x_min_d, y_min_d, x_max_d, y_max_d))
            
            if iou_value > max_iou:
                max_iou = iou_value
                matching_validation = label_v
                
                matching_validation_row.append(_)
    
    # Add ground truth and prediction based on maximum IoU
    #if matching_validation:
    y_pred.append(label_d)
    y.append(matching_validation)

for _, validation_row in validations.iterrows():
    image_id_v, x_min_v, y_min_v, x_max_v, y_max_v, label_v = validation_row
    if _  not in matching_validation_row:
        y.append(label_v)
        y_pred.append('none')
     

print(len(y_pred))
print(len(y))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# plot confusion matrix
cm = confusion_matrix(y, y_pred)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
# Get unique labels
unique_labels = list(set(y))

plt.figure(figsize=(10, 7))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2%",
    xticklabels=unique_labels,
    yticklabels=unique_labels,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

precision = {}
recall = {}

for i in range(len(unique_labels)):
    cls = unique_labels[i]
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp
    
    precision[cls] = tp / (tp + fp) if tp + fp > 0 else 0
    recall[cls] = tp / (tp + fn) if tp + fn > 0 else 0

print("Precision:\n", precision)
print("\nRecall:\n", recall)

ap_per_class = []
for cls in unique_labels:
    if precision[cls] + recall[cls] > 0:
        ap_per_class.append(precision[cls] * recall[cls] / (precision[cls] + recall[cls]))

mAP = np.mean(ap_per_class) if ap_per_class else 0

print("\nmAP:\n", mAP)


#%% select_region_and_predict
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
            prediction = model.predict_window(roi)
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

# exemple
image_path = 'C:/Users/23158/Desktop/SY32PROJET/dataset2/val/images/0196.jpg'
result = select_region_and_predict(image_path, my_model)
print("Prediction Result:", result)

#%% stop
image = cv2.imread('C:/Users/23158/Desktop/SY32PROJET/dataset2/val/images/0196.jpg')
detections = detect_traffic_signs_without_piramid(image, my_model)
boxes = np.array(detections)
picked_boxes = non_max_suppression(boxes, 0.2)
#picked_boxes=boxes
print(picked_boxes)
for (x1, y1, x2, y2, label,max_probabilities) in picked_boxes:  
    x1, y1, x2, y2 ,max_probabilities= int(x1), int(y1), int(x2), int(y2),float(max_probabilities)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    text = f"{label}: {max_probabilities:.2f}"
    # 显示文本
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  
plt.title('Detections')  
plt.axis('off')  
plt.show()












