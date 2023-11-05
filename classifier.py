import os
import torch
import numpy as np
from network import CNN_model
import cv2
import shutil
torch.backends.cudnn.benchmark = True
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import keyboard


NUM_CLASS = 14
LABEL_FILE  = 'Chinese_chess_labels.txt'
MODEL_FILE  = 'run\\run_6\\models\\CNN-Chinese_chess_epoch-59.pth.tar'



class Chinese_chess_classifier:
    def __init__(self, num_class=NUM_CLASS, label_file=LABEL_FILE, model_file=MODEL_FILE):
        self.num_class = num_class      # number of classes
        self.label_file = label_file    # label file path
        self.model_file = model_file    # model file path

        # size of image used for training
        self.resize_height = 50         
        self.resize_width = 50

        # init device
        print('MODEL_FILE is',self.model_file)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device being used:", self.device)
        
        # init model
        self.model = CNN_model.CNNModel(num_classes=self.num_class)
        self.checkpoint = torch.load(self.model_file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # init label files
        with open(self.label_file, 'r') as f:
            self.class_names = f.readlines()
            f.close()



    # Method to classify a single chess piece
    def single_classification(self, img_uin8):
        
        # convert uint8 to float32
        image_float32 = img_uin8.astype(np.float32) / 255.0
        image_float32 = image_float32 - np.array([[90.0, 98.0, 102.0]])/255.0

        # prepare the input of the model
        inputs = image_float32
        inputs = np.expand_dims(inputs, axis=0)
        inputs = np.transpose(inputs,(0, 3, 1, 2 ))
        inputs = torch.from_numpy(inputs)
        inputs = torch.autograd.Variable(inputs, requires_grad=False).to(self.device)

        # input into model
        with torch.no_grad():
            outputs = self.model.forward(inputs)

        probs = torch.nn.Softmax(dim=1)(outputs)
        label_order = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
        predict_label = self.class_names[label_order].split(' ')[-1].strip()
        return predict_label


    # Method to classify a list of chess pieces
    def overall_classification(self, img_uin8_list):
        '''
        param: img_uin8_list is a list of uint8 numpy arrays
        '''
        classification_results = []
        for img in img_uin8_list:

            w, h = img.shape[:2]
            if (w != self.resize_height) or (h != self.resize_width):
                img = cv2.resize(img, (self.resize_width, self.resize_height))

            result = self.single_classification(img)
            classification_results.append(result)
        return classification_results



if __name__ == '__main__':
    pass