# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:11:24 2024

@author: 23158
"""



import matplotlib.pyplot as plt
import cv2
import numpy as np
from src.model import model
import pandas as pd
import os
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



def sliding_window1(image, window_size):
    for y in range(0, image.shape[0] - window_size[1] + 1, window_size[1]//5):
        for x in range(0, image.shape[1] - window_size[0] + 1, window_size[0]//5):
            yield (x, y, window_size, image[y:y + window_size[1], x:x + window_size[0]])

def detect_traffic_signs1(image, model):
    detections = []
    window_size = (500, 500)
    scale_factor = 0.8
    
    while window_size[0] >= 50 and window_size[1] >= 50:
        for (x, y, win_size, window) in sliding_window1(image, window_size):
            if window.shape[0] != win_size[1] or window.shape[1] != win_size[0]:
                continue

            prediction = model.predict_window(window)
            prediction_probabilities = model.predict_proba_window(window)
            max_probabilities = np.max(prediction_probabilities, axis=1)

            if prediction != 'none' and max_probabilities > 0.9:  # Assuming '1' indicates a traffic sign
                x_orig = int(x)
                y_orig = int(y)
                w_orig = int(win_size[0])
                h_orig = int(win_size[1])
                prob = max_probabilities[0]
                detections.append((x_orig, y_orig, x_orig + w_orig, y_orig + h_orig, prediction[0], prob))

        # 缩小窗口尺寸
        window_size = (int(window_size[0] * scale_factor), int(window_size[1] * scale_factor))

    return detections








# Example usage:
# Assuming you have an image `image` and a model `model` defined somewhere
# detections = detect_traffic_signs(image, model, win_sizes=[(450, 450)], step_size=25, min_prob=0.5)





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


#%%
def process_image(image_path, model, win_sizes, step_size, pyramid_scale, csv_path):
    image = cv2.imread(image_path)
    detections = detect_traffic_signs(image, model, win_sizes, step_size, pyramid_scale)
    boxes = np.array(detections)
    picked_boxes = non_max_suppression(boxes, 0.3)  # 阈值0.3可根据需要调整
    
    results = []
    for (x1, y1, x2, y2, label, max_probabilities) in picked_boxes:
        #x1, y1, x2, y2, max_probabilities = int(x1), int(y1), int(x2), int(y2), float(max_probabilities)
        #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #text = f"{label}: {max_probabilities:.2f}"
        #cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        results.append([image_path, x1, y1, x2, y2, max_probabilities, label])
    
    df = pd.DataFrame(results, columns=['Num img', 'Coin h-g x', 'Coin h-g y', 'Coin b-d x', 'Coin b-d y', 'Score', 'Classe'])
    df.to_csv(csv_path, index=False)
    
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.title('Detections')
    #plt.axis('off')
    #plt.show()

def process_images_in_folder(folder_path, model, win_sizes, step_size, pyramid_scale, csv_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            detections = detect_traffic_signs(image, model, win_sizes, step_size, pyramid_scale)
            boxes = np.array(detections)
            picked_boxes = non_max_suppression(boxes, 0.3)  # 阈值0.3可根据需要调整

            for (x1, y1, x2, y2, label, max_probabilities) in picked_boxes:
                x1, y1, x2, y2, max_probabilities = int(x1), int(y1), int(x2), int(y2), float(max_probabilities)
                #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #text = f"{label}: {max_probabilities:.2f}"
                #cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                results.append([filename, x1, y1, x2, y2, max_probabilities, label])
            
            # 显示图像（可选）
            #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            #plt.title(f'Detections for {filename}')
            #plt.axis('off')
            #plt.show()
    
    df = pd.DataFrame(results, columns=['Num img', 'Coin h-g x', 'Coin h-g y', 'Coin b-d x', 'Coin b-d y', 'Score', 'Classe'])
    df.to_csv(csv_path, index=False)

process_images_in_folder('dataset2/val/images/', my_model, [(50,50),(100,100),(200,200),(400,400),(50,125),(100,250),(200,500)], 25, 1.5, 'detections.csv')

#%% select_region_and_predict

def select_region_and_predict(image_path, model):
    """
    读取图像并使用鼠标选择一块区域，然后使用模型预测该区域的结果。
    
    参数:
    - image_path: 图像文件路径
    - model: 模型对象，包含 predict_window 和 predict_proba_window 方法
    
    返回:
    - 预测结果
    """
    # 初始化全局变量
    global ref_point, cropping, image, clone
    cropping = False
    ref_point = []

    # 定义鼠标回调函数
    def click_and_crop(event, x, y, flags, param):
        global ref_point, cropping

        # 如果左键按下，记录起始点
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point = [(x, y)]
            cropping = True

        # 如果左键松开，记录结束点
        elif event == cv2.EVENT_LBUTTONUP:
            ref_point.append((x, y))
            cropping = False

            # 画出矩形选区
            cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
            cv2.imshow("image", image)

    # 读取图像
    image = cv2.imread(image_path)
    clone = image.copy()

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    # 显示图像并等待用户进行选择
    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # 按 'r' 键重置选择
        if key == ord('r'):
            image = clone.copy()

        # 按 'c' 键确认选择并进行预测
        elif key == ord('c') and len(ref_point) == 2:
            roi = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            prediction = model.predict_window(roi)
            prediction_probabilities = model.predict_proba_window(roi)

            # 获取前3个预测结果
            top_indices = prediction_probabilities.argsort()[0][-3:][::-1]
            top_classes = model.classifier.classes_[top_indices]
            top_probabilities = prediction_probabilities[0][top_indices]

            # 在图像上显示选择区域、尺寸和预测结果
            cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
            text = f"Size: {ref_point[1][0] - ref_point[0][0]}x{ref_point[1][1] - ref_point[0][1]}"
            cv2.putText(image, text, (ref_point[0][0], ref_point[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            y_offset = ref_point[1][1] + 30
            for i, (cls, prob) in enumerate(zip(top_classes, top_probabilities)):
                text = f"{i+1}: {cls} ({prob:.2f})"
                cv2.putText(image, text, (ref_point[0][0], y_offset + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("image", image)
            ref_point = []

        # 按 '0' 键退出程序
        elif key == ord('0'):
            break

    cv2.destroyAllWindows()

# 使用示例
image_path = 'dataset2/val/images/0004.jpg'
result = select_region_and_predict(image_path, my_model)
print("Prediction Result:", result)



#%%obligation

image = cv2.imread('dataset2/val/images/0325.jpg')
detections = detect_traffic_signs(image, my_model,win_sizes=[(50,50),(100,100),(200,200),(400,400)], step_size=25, pyramid_scale=1.5)
#detections=detect_traffic_signs1(image,my_model)
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
#%%danger and interdition
image = cv2.imread('dataset2/val/images/0757.jpg')
detections = detect_traffic_signs(image, my_model,win_sizes=[(50,50),(100,100),(200,200),(400,400)], step_size=25, pyramid_scale=1.5)
#detections=detect_traffic_signs1(image,my_model)
boxes = np.array(detections)
picked_boxes = non_max_suppression(boxes, 0.1)
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
#%% stop
image = cv2.imread('dataset2/val/images/0088.jpg')
detections = detect_traffic_signs(image, my_model,win_sizes=[(50,50),(100,100),(200,200),(400,400)], step_size=25, pyramid_scale=1.5)
#detections=detect_traffic_signs1(image,my_model)
boxes = np.array(detections)
picked_boxes = non_max_suppression(boxes, 0)
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


#%% stop
image = cv2.imread('dataset2/val/images/0568.jpg')
detections = detect_traffic_signs(image, my_model,win_sizes=[(50,50),(100,100),(200,200),(400,400)], step_size=25, pyramid_scale=1.5)
#detections=detect_traffic_signs1(image,my_model)
boxes = np.array(detections)
picked_boxes = non_max_suppression(boxes, 0)
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

#%% stop
image = cv2.imread('dataset2/val/images/0139.jpg')
detections = detect_traffic_signs(image, my_model,win_sizes=[(50,50),(100,100),(200,200),(400,400)], step_size=25, pyramid_scale=1.5)
#detections=detect_traffic_signs1(image,my_model)
boxes = np.array(detections)
picked_boxes = non_max_suppression(boxes, 0)
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
#%% stop
image = cv2.imread('dataset2/val/images/0004.jpg')
detections = detect_traffic_signs(image, my_model,win_sizes=[(50,50),(100,100),(200,200),(400,400),(50,125),(100,250),(200,500)], step_size=25, pyramid_scale=1.5)
#detections=detect_traffic_signs1(image,my_model)
boxes = np.array(detections)
picked_boxes = non_max_suppression(boxes, 0)
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

