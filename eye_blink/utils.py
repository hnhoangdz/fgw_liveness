from email.policy import strict
from scipy.spatial import distance as dist
from torchvision import transforms
import torch
import numpy as np
import cv2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eye_landmarks_to_bbox(eyes, padding=6):
    p0, p1, p2, p3, p4, p5 = [x for x in eyes]
    xmin = p0[0]
    ymin = min(p1[1], p2[1])
    xmax = p3[0]
    ymax = max(p4[1], p5[1])
    return xmin - padding-2, ymin - padding, xmax + padding+2, ymax + padding

def eye_aspect_ratio(eyes):
    A = dist.euclidean(eyes[1],eyes[5])
    B = dist.euclidean(eyes[2],eyes[4])
    C = dist.euclidean(eyes[0],eyes[3])
    ear = (A + B) / (2.0 * C)
    return ear

def data_transform(mean=(0.5,), std=(0.5,), input_size =(24, 24)):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform

def load_model(model_path, model):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    return model

def predict(img, model_path, model, 
            class_dict = ['close','open']):
        
    model = load_model(model_path, model)
    transform = data_transform()
    img_transformed = transform(img)
    img_transformed = img_transformed.unsqueeze_(0) # (1, c, h, w)
    with torch.no_grad():
        outputs = model(img_transformed)
    predict_id = np.argmax(outputs.detach().numpy())
    if predict_id == 0:
        if torch.nn.Softmax(-1)(outputs)[0][0] < 0.95:
            predict_id = 1
    predict_label = class_dict[predict_id]
    
    return predict_label

def get_visualize(frame, landmarks):
    for (x,y) in landmarks:
        cv2.circle(frame, (x,y), 1, (255,255,0), 1, cv2.LINE_4)