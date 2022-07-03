import torch
from torch.utils.data import Dataset
from enum import Enum
import glob
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import math

class DatasetType(Enum):
    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"

class NSRVideoDataset(Dataset):
    def __init__(self, dataset_path : str, split : tuple, dataset_type : str, transform = None) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.transform = transform
        self.cache = {}

        label_paths = glob.glob(dataset_path + "\*")
        illegal_paths = glob.glob(label_paths[0] + "\*")
        labels = [[1]] * len(illegal_paths)

        legal_paths = glob.glob(label_paths[1] + "\*")
        labels += [[0]] * len(legal_paths)
        data_paths = illegal_paths + legal_paths

        x_train, x_valid, y_train, y_valid = train_test_split(data_paths, labels, test_size=split[0], shuffle=True, stratify=labels, random_state=34)
        x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=split[1], shuffle=True, stratify=y_valid, random_state=34)

        if dataset_type == DatasetType.TRAIN.value:
            self.data_paths, self.labels = x_train, y_train
        elif dataset_type == DatasetType.VALIDATION.value:
            self.data_paths, self.labels = x_valid, y_valid
        else:
            self.data_paths, self.labels = x_test, y_test
            
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        image = self.extract_diagnoal_matrix(self.data_paths[index])
        label = torch.FloatTensor(self.labels[index])
        # label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)
        
        self.cache[index] = image, label

        return image, label
        

    def extract_diagnoal_matrix(self, data_path : str):
        videocap = cv2.VideoCapture(data_path)
        video_total_frames_num = videocap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_frame_per_s = int(videocap.get(cv2.CAP_PROP_FPS))

        sections, retstep = np.linspace(1, video_total_frames_num, 256, retstep=True)
        sections = list(map(math.floor, sections))
        frame_diagonals = []

        while(videocap.isOpened()):
            ret, frame = videocap.read()
            
            if not ret:
                break
            
            if int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) in sections:
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                frame = cv2.resize(frame, (256, 256))

                frame_r, frame_g, frame_b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
                frame_r, frame_g, frame_b = np.diag(frame_r), np.diag(frame_g), np.diag(frame_b)
                
                frame_diagonal = np.stack([frame_r, frame_g, frame_b], -1)
                frame_diagonal = np.expand_dims(frame_diagonal, 1)

                frame_diagonals.append(frame_diagonal)
        
        videocap.release()
        video_diagonal = np.concatenate(frame_diagonals, axis=1)
    
        return video_diagonal