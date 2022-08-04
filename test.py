from eye_blink.model import Model
import torch
from cfg import *
import cv2
from PIL import Image
from torchvision import transforms
# from eye_blink.utils import data_transform
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(2)

def load_model(model_path, model, device):
    load_weights = torch.load(model_path, map_location=device)
    model.load_state_dict(load_weights, strict=False)
    return model

model_path = eye_blink_model_zju_v2
model = Model(2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, model, device)
model = model.to(device)
model.eval()

resize = (24, 24)
mean = (0.5,)
std = (0.5, )

class_names = ['close', 'open']

def predict_img(img_path):
    
    data_transform = transforms.Compose([
        transforms.Resize(size=resize),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    img = cv2.imread(img_path, 0)
    img = Image.fromarray(img)  
    img = data_transform(img).to(device).unsqueeze_(0)
    outs = model(img)
    print('outs: ', outs)
    probs = torch.softmax(outs, dim=1)
    print('prob: ', probs)
    pred_label = class_names[np.argmax(outs.detach().cpu().numpy())]
    
    return pred_label

parent_paths = 'test_images'
paths = ['leftEye.jpg', 'rightEye.jpg', 'close.png', 'ct0001.jpg', 'ct0004.jpg',\
         'closed_eye.jpg', 'close_right.png', 'open_left.png', 'open_right.png','open_left_g.png', 'open_right_g.png']

for img_path in paths:
    print('img_path: ', os.path.join(parent_paths,img_path)) 
    label = predict_img(os.path.join(parent_paths,img_path))
    print('label: ', label)
    print('==================')

