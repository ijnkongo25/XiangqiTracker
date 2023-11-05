import os
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
import keyboard

frame_num = 40

class VideoDataset(Dataset):


    def __init__(self, dataset='Chinese_chess', split='train', preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 50
        self.resize_width = 50
        self.resize_channel = 3


        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if  preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)


        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}

        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        
        if dataset == 'Chinese_chess_database':
            if not os.path.exists('Chinese_chess_labels.txt'):
                with open('Chinese_chess__labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

    
    def __len__(self):
        return len(self.fnames)


    def __getitem__(self, index):
        # Loading and preprocessing.

        buffer = self.load_frames(self.fnames[index])
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)

        buffer = buffer - np.array([[90.0, 98.0, 102.0]])/255
        buffer = self.to_tensor(buffer)

        return torch.from_numpy(buffer), torch.from_numpy(labels)



    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True


    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            img_files = [name for name in os.listdir(file_path)]

            train_and_valid, test = train_test_split(img_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for chess_img in train:
                self.process_imgs(chess_img, file, train_dir)

            for chess_img in val:
                self.process_imgs(chess_img, file, val_dir)

            for chess_img in test:
                self.process_imgs(chess_img, file, test_dir)

        print('Preprocessing finished.')


    def process_imgs(self, chess_img, chess_name, save_dir):

        image_path = os.path.join(self.root_dir,chess_name, chess_img)

        img = Image.open(image_path)
        img_array = np.asarray(img, dtype=np.uint8)


        i = 0
        brightness_values=(-30, 0, 30)
        rotate_angles = (-10, -5, 0, 5, 10)

        for value in brightness_values:
            
            adjusted_image = cv2.convertScaleAbs(img_array, alpha=1, beta=value)

            w, h = adjusted_image.shape[:2]
            if (w != self.resize_height) or (h != self.resize_width):
                adjusted_image = cv2.resize(adjusted_image, (self.resize_width, self.resize_height))
            
            for angle in rotate_angles:
                w2, h2 = adjusted_image.shape[:2]
                rotation_mx = cv2.getRotationMatrix2D((w2 / 2, h2 / 2), angle, 1)
                rotated_img_array = cv2.warpAffine(adjusted_image, rotation_mx, (w2, w2))

                w3, h3 = rotated_img_array.shape[:2]
                if (w3 != self.resize_height) or (h3 != self.resize_width):
                    rotated_img_array = cv2.resize(rotated_img_array, (self.resize_width, self.resize_height))

                while os.path.exists(os.path.join(save_dir, '0000{}.jpg'.format(str(i)))):
                    i += 1
                img_save_path = os.path.join(save_dir, '0000{}.jpg'.format(str(i)))

                bgr_image = cv2.cvtColor(rotated_img_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename=img_save_path, img=bgr_image)




    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)
        return buffer


    def normalize(self, buffer):
        buffer = buffer - np.array([[90.0, 98.0, 102.0]])/255.0
        return buffer


    def to_tensor(self, buffer):
        return buffer.transpose((2, 0, 1 ))

    def load_frames(self, file_dir):
        img_uin8 = np.array(cv2.imread(file_dir)).astype(np.uint8)
        image_float32 = img_uin8.astype(np.float32) / 255.0
        buffer = image_float32
        return buffer


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='Auslan', split='train', clip_len=8, preprocess=True)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

