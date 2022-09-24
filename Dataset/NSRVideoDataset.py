import torch
from torch.utils.data import Dataset
from enum import Enum
import glob
import cv2
import numpy as np
import math
import random
from PIL import Image
from torchvision.transforms import transforms

class DatasetType(Enum):
    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"

class NSRVideoDataset(Dataset):
    def __init__(self, dataset_path : str, split : tuple, dataset_type : str, feature : str, use_frame_df : bool, is_transform : bool) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.is_transform = is_transform
        self.feature = feature
        self.use_frame_df = use_frame_df

        self.label_paths = glob.glob(dataset_path + "\*")
        agree_paths = glob.glob(self.label_paths[0] + "\*.mp4")
        agree_paths = [(agree_path, [1]) for agree_path in agree_paths]
        
        non_agree_paths = glob.glob(self.label_paths[1] + "\*.mp4")
        non_agree_paths = [(non_agree_path, [0]) for non_agree_path in non_agree_paths]

        train_split_index_agree, train_split_index_non_agree = int(len(agree_paths)*split[0]), int(len(non_agree_paths)*split[0])
        validation_split_index_agree, validation_split_index_non_agree = train_split_index_agree + int(len(agree_paths)*split[1]), train_split_index_non_agree + int(len(non_agree_paths)*split[1])

        random.shuffle(agree_paths)
        random.shuffle(non_agree_paths)

        if dataset_type == DatasetType.TRAIN.value:
            self.data_paths = agree_paths[:train_split_index_agree] + non_agree_paths[:train_split_index_non_agree]
        elif dataset_type == DatasetType.VALIDATION.value:
            self.data_paths = agree_paths[train_split_index_agree:validation_split_index_agree] + non_agree_paths[train_split_index_non_agree:validation_split_index_non_agree]
        else:
            self.data_paths = agree_paths[validation_split_index_agree:] + non_agree_paths[validation_split_index_non_agree:]
        
        random.shuffle(self.data_paths)
            
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        video_diagonal = self.extract_feature(self.data_paths[index][0])
        label = torch.FloatTensor(self.data_paths[index][1])

        if self.is_transform:
            if self.feature == "hist":
                video_diagonal = self.transform_features(video_diagonal)
            elif self.feature == "diag":
                video_diagonal_tr = self.transform_features(video_diagonal.copy())
                mean, std = video_diagonal_tr.mean([1,2]), video_diagonal_tr.std([1,2])
                video_diagonal = self.transform_features(video_diagonal, mean, std, is_normalize=True)

        return video_diagonal, label
        

    def extract_feature(self, data_path : str):
        videocap = cv2.VideoCapture(data_path)
        video_total_frames_num = videocap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_frame_per_s = int(videocap.get(cv2.CAP_PROP_FPS))

        # Use frame difference
        if self.use_frame_df:
            section_split = 257
        # Use each frame 
        else:
            section_split = 256
        
        sections, retstep = np.linspace(1, video_total_frames_num, section_split, retstep=True)
        sections = list(map(math.floor, sections))
        frame_diagonals = []
        video_name = data_path.split("\\")[-1]
        
        while(videocap.isOpened()):
            ret, frame = videocap.read()
            
            if not ret:
                break
            
            if int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) in sections and self.feature == "diag":
                frame = cv2.resize(frame, (256, 256))

                if self.use_frame_df:
                    if int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) == 1:
                        pre_frame = frame
                        continue
                
                    frame_df = cv2.absdiff(pre_frame, frame)
                    frame_r, frame_g, frame_b = frame_df[:,:,0], frame_df[:,:,1], frame_df[:,:,2]
                    pre_frame = frame
                
                else:
                    frame_r, frame_g, frame_b = frame[:,:,0], frame[:,:,1], frame[:,:,2]

                frame_r, frame_g, frame_b = np.diag(frame_r), np.diag(frame_g), np.diag(frame_b)
                frame_diagonal = np.stack([frame_r, frame_g, frame_b], -1)
                frame_diagonal = np.expand_dims(frame_diagonal, 1)
                frame_diagonals.append(frame_diagonal)
            
            elif int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) in sections and self.feature == "hist":
                frame_r, frame_g, frame_b = frame[:,:,0], frame[:,:,1], frame[:,:,2]

                frame_r_ht = cv2.calcHist(frame_r, [0], None, [256], [0, 256])
                frame_g_ht = cv2.calcHist(frame_g, [0], None, [256], [0, 256])
                frame_b_ht = cv2.calcHist(frame_b, [0], None, [256], [0, 256])

                if self.use_frame_df:
                    if int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) == 1:
                        pre_frame_r_ht = frame_r_ht
                        pre_frame_g_ht = frame_g_ht
                        pre_frame_b_ht = frame_b_ht
                        continue

                    frame_r_ht_df = abs(pre_frame_r_ht - frame_r_ht)
                    frame_g_ht_df = abs(pre_frame_g_ht - frame_g_ht)
                    frame_b_ht_df = abs(pre_frame_b_ht - frame_b_ht)

                    frame_diagonal = np.stack([frame_r_ht_df, frame_g_ht_df, frame_b_ht_df], -1)
                    pre_frame_r_ht = frame_r_ht
                    pre_frame_g_ht = frame_g_ht
                    pre_frame_b_ht = frame_b_ht

                else:
                    frame_diagonal = np.stack([frame_r_ht, frame_g_ht, frame_b_ht], -1)
                
                frame_diagonals.append(frame_diagonal)

        videocap.release()

        # use histogram feature with commercial log
        if self.feature == "hist":
            video_diagonal = np.concatenate(frame_diagonals, axis=1)
            video_diagonal = np.log(video_diagonal + 1)
            # video_diagonal = video_diagonal.astype(np.uint8)
            # video_diagonal = Image.fromarray(video_diagonal)
        elif self.feature == "diag":
            video_diagonal = np.concatenate(frame_diagonals, axis=1)

        return video_diagonal

    def transform_features(self, image, mean = 0, std = 0, is_normalize = False):
        if is_normalize:
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
            ])

            return transform(image)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(), 
            ])

            return transform(image)