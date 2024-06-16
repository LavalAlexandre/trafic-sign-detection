import cv2
import numpy as np


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

def detect_traffic_signs_without_piramid(image, model, step_size_ratio=0.1, min_window_size_ratio=0.1, max_window_size_ratio=0.8):
    detections = []
   
    for (x, y, (win_width, win_height), window) in sliding_window_without_piramid(image, step_size_ratio,min_window_size_ratio,max_window_size_ratio):
        if window.shape[0] != win_height or window.shape[1] != win_width:
            continue

        prediction = model.predict_window(window)
        prediction_probabilities = model.predict_proba_window(window)
        max_probabilities = np.max(prediction_probabilities, axis=1)
        
        if prediction != 'none' and max_probabilities[0] > 0.9:  # Assuming '1' indicates a traffic sign
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