from eye_blink.model import Model
import torch
from cfg import *
import cv2
from PIL import Image
from torchvision import transforms
from eye_blink.utils import data_transform
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(2)

def load_model(model_path, model):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

resize = (24,24)
mean = (0.5,)
std = (0.5, )

data_transform = transforms.Compose([
                transforms.Resize(size=resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
model_path = eye_blink_model_zju

model = load_model(model_path, model)
model.eval()

left_eye = cv2.imread('ct0001.jpg', 0)
print(left_eye)
left_eye = Image.fromarray(left_eye)  
left_eye = data_transform(left_eye).unsqueeze_(0)

right_eye = cv2.imread('ct0004.jpg', 0)
print(right_eye.shape)
right_eye = Image.fromarray(right_eye)  
right_eye = data_transform(right_eye).unsqueeze_(0)

class_names = ['close', 'open']

left_eye_pred = model(left_eye)
print('left eye:', torch.nn.Softmax(-1)(left_eye_pred))
print('left eye name: ', class_names[np.argmax(left_eye_pred.detach().numpy())])

right_eye_pred = model(right_eye)
print('right eye:', torch.nn.Softmax(-1)(right_eye_pred))
print('right eye name: ', class_names[np.argmax(right_eye_pred.detach().numpy())])

