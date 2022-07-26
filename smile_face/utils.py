from torchvision import transforms
import torch
import numpy as np
import cv2
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ',device)

def pre_process(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def data_transform(input_size=(64, 64)):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform

def load_model(model_path, model):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def predict(img, model_path, model, 
            class_dict = ['non-smile','smile']):
        
    model = load_model(model_path, model)
    
    transform = data_transform()
    img = Image.fromarray(img)
    img_transformed = transform(img)
    img_transformed = img_transformed.unsqueeze_(0)

    outputs = model(img_transformed)
    probs = torch.softmax(outputs, dim=1)
    happy_prob = probs[0][1]
    predict_id = np.argmax(outputs.detach().numpy())
    print('probablity happy: ', happy_prob)
    predict_label = class_dict[predict_id]
    # if happy_prob < 0.8 and predict_id == 1:
    #     print('not sure')
    #     predict_id = 0

    # predict_label = class_dict[predict_id]
    # if predict_label == 'happy':
        
    #     print(happy_prob)
    return predict_label

def get_visualize(frame, label):
    if label == 'smile':
        color = (0,255,0)
    else:
        color = (0,0,255)
        
    cv2.putText(frame, f'emoji: {label}', (7, 70), 1, 1, color, 1)    