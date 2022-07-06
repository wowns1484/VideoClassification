import torch
from torch.utils.data import Dataset
from enum import Enum
import glob
import cv2
import numpy as np
import math
import random
from PIL import Image

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

        self.label_paths = glob.glob(dataset_path + "\*")
        legal_paths = glob.glob(self.label_paths[0] + "\*")
        legal_paths = [(legal_path, [1]) for legal_path in legal_paths]
        
        illegal_paths = glob.glob(self.label_paths[1] + "\*")
        illegal_paths = [(illegal_path, [0]) for illegal_path in illegal_paths]

        train_split_index_legal, train_split_index_illegal = int(len(legal_paths)*split[0]), int(len(illegal_paths)*split[0])
        validation_split_index_legal, validation_split_index_illegal = train_split_index_legal + int(len(legal_paths)*split[1]), train_split_index_illegal + int(len(illegal_paths)*split[1])

        if dataset_type == DatasetType.TRAIN.value:
            self.data_paths = legal_paths[:train_split_index_legal] + illegal_paths[:train_split_index_illegal]
        elif dataset_type == DatasetType.VALIDATION.value:
            self.data_paths = legal_paths[train_split_index_legal:validation_split_index_legal] + illegal_paths[train_split_index_illegal:validation_split_index_illegal]
        else:
            self.data_paths = legal_paths[validation_split_index_legal:] + illegal_paths[validation_split_index_illegal:]
        
        random.shuffle(self.data_paths)
            
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        video_diagonal = self.extract_diagnoal_matrix(self.data_paths[index][0])
        # video_diagonal = video_diagonal.permute(2, 0, 1)
        label = torch.FloatTensor(self.data_paths[index][1])

        if self.transform is not None:
            video_diagonal = self.transform(video_diagonal)
        
        self.cache[index] = (video_diagonal, label)

        return video_diagonal, label
        

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
                # frame_r = [[np.mean(row) / 255] for row in frame_r]
                # frame_g = [[np.mean(row) / 255] for row in frame_g]
                # frame_b = [[np.mean(row) / 255] for row in frame_b]
                
                frame_diagonal = np.stack([frame_r, frame_g, frame_b], -1)
                frame_diagonal = np.expand_dims(frame_diagonal, 1)

                frame_diagonals.append(frame_diagonal)
        
        videocap.release()
        video_diagonal = np.concatenate(frame_diagonals, axis=1)
        video_diagonal = Image.fromarray(video_diagonal)
    
        return video_diagonal