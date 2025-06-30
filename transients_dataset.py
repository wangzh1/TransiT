import os
import glob
import numpy as np
import scipy.io
import pdb
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.io import savemat, loadmat
import time 
from scipy.ndimage import zoom
import random

import decord
from decord import VideoReader, cpu

def read_video_decord(video_path):
    decord.bridge.set_bridge('torch')
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(range(len(vr)))
    # print('here', frames.shape, frames.dtype) # torch.Size([1000, 256, 256, 3]) torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # 调整维度
    frames = torch.mean(frames.float(), dim=1)  # 转换为灰度
    return frames

class TransientsTransform:
    def __init__(self, crop_hist, crop_hist_start, crop_hist_end, downsample_hist, downsample_spatial):
        self.crop_hist = crop_hist
        self.crop_hist_start = crop_hist_start
        self.crop_hist_end = crop_hist_end
        self.downsample_hist = downsample_hist
        self.downsample_spatial = downsample_spatial

    def __call__(self, data):

        if self.crop_hist:
            # crop data
            data = data[self.crop_hist_start:self.crop_hist_end, :, :]

        if self.downsample_hist > 0:
            from scipy.ndimage import zoom
            zoom_factors = [1/self.downsample_hist, 1., 1.]
            data = zoom(data, zoom_factors, order=1) 

        # if self.downsample_spatial > 0:
        #     data = data[:, ::self.downsample_spatial, ::self.downsample_spatial]
            
        # flatten data
        # l, _, _ = data.shape
        # data = data.reshape(l, -1)

        # normalize data
        if data.max() > 0: 
            data = data / data.max()

        return data

class ReaslMatDataDataset(Dataset):
    def __init__(self, directory):
        """
        初始化 MatDataDataset
        :param directory: 存放 .mat 文件的目录路径
        """
        self.directory = directory
        self.files = [f for f in os.listdir(directory) if f.endswith('.mat')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 获取文件路径
        file_path = os.path.join(self.directory, self.files[idx])
        
        # 读取 .mat 文件中的 "data" 变量
        mat_data = loadmat(file_path)
        
        if 'data' not in mat_data:
            raise KeyError(f"File {self.files[idx]} does not contain a 'data' variable.")
        
        # 提取 "data" 变量并将其转换为张量
        data = mat_data['data']
        data_tensor = torch.tensor(data, dtype=torch.float32)
        data_tensor = data_tensor[:,:,400:1000]
        data_tensor = data_tensor[:,:,::3] + data_tensor[:,:,1::3] + data_tensor[:,:,2::3]
        data_tensor = data_tensor.permute([2,0,1])
        data_tensor = data_tensor / data_tensor.max()
        return data_tensor

class RealVideoMatDataDataset(Dataset):
    def __init__(self, directory, n=1):
        """
        初始化 MatDataDataset
        :param directory: 存放 .mat 文件的目录路径
        :param n: 每次读取的连续数据数量
        """
        self.directory = directory
        self.files = sorted([f for f in os.listdir(directory) if f.endswith('.mat')], key=lambda x: int(x.split('_')[-1].replace('frame', '').replace('.mat', '')))
        self.n = n

    def __len__(self):
        # 数据集的长度需要考虑 n 的大小，防止越界
        return len(self.files) - self.n + 1

    def __getitem__(self, idx):
        tensors = []
        # 读取连续的 n 个文件
        for i in range(self.n):
            file_path = os.path.join(self.directory, self.files[idx + i])
            mat_data = loadmat(file_path)

            if 'data' not in mat_data:
                raise KeyError(f"File {self.files[idx + i]} does not contain a 'data' variable.")

            # 提取 "data" 变量并将其转换为张量
            data = mat_data['data']
            data_tensor = torch.tensor(data, dtype=torch.float32)
            data_tensor = data_tensor[:, :, 400:1000]
            data_tensor = data_tensor[:, :, ::3] + data_tensor[:, :, 1::3] + data_tensor[:, :, 2::3]
            data_tensor = data_tensor.permute([2, 0, 1])
            data_tensor = data_tensor / data_tensor.max()
            tensors.append(data_tensor)

        # 合并连续的 n 个数据
        # combined_tensor = torch.cat(tensors, dim=0)
        combined_tensor = torch.stack(tensors) 
        return combined_tensor

class NLOSVideoDatasetTest(Dataset):
    def __init__(self, root_dir, sequence_length=3, group_size=50, transform_hdr=None, transform_gray=None):
        """
        初始化数据集

        Args:
            root_dir (str): 数据集的根目录
            sequence_length (int): 序列长度，默认为3（连续三帧）
            group_size (int): 每组帧的数量，默认为50
            transform_hdr (callable, optional): HDR 图像的预处理函数
            transform_gray (callable, optional): 灰度图像的预处理函数
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.group_size = group_size
        self.transform_hdr = transform_hdr
        self.transform_gray = transform_gray
        self.data_sequences = []
        
        self.shift_num = torch.tensor(loadmat('shift_num.mat')['shift_num'].astype(np.float64))
        self.d1_process = torch.tensor(loadmat('D1_process.mat')['D1_process'][..., None])
        time_bin = 10
        self.shift_num /= (time_bin / 4.0)
        self.shift_num = self.shift_num.to(torch.int16)
        
        # # 获取所有物体的路径
        object_dirs = [os.path.join(root_dir, obj) for obj in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, obj))]
        # 仅获取名称包含 "windmill" 的文件夹路径
        # object_dirs = [os.path.join(root_dir, obj) for obj in os.listdir(root_dir)
        #                if os.path.isdir(os.path.join(root_dir, obj)) and ("R5" in obj)]
        for obj_dir in object_dirs:
            # 获取该物体下的所有帧目录，按照修改时间排序
            frame_dirs = [os.path.join(obj_dir, frame) for frame in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, frame))]
            # frame_dirs.sort(key=os.path.getmtime)
            frame_dirs.sort()

            # 将帧目录分组，每组 group_size 个帧
            num_frames = len(frame_dirs)
            num_groups = num_frames // self.group_size
            for g in range(num_groups):
                group_frames = frame_dirs[g*self.group_size : (g+1)*self.group_size]
                # 在组内生成连续三帧的序列
                self._generate_sequences(group_frames)

            # 处理剩余不足 group_size 的帧（如果有且数量足够）
            remaining_frames = frame_dirs[num_groups*self.group_size:]
            if len(remaining_frames) >= self.sequence_length:
                self._generate_sequences(remaining_frames)

    def _generate_sequences(self, frame_list):
        """
        在给定的帧列表中生成连续序列

        Args:
            frame_list (list): 帧目录列表
        """
        for i in range(len(frame_list) - self.sequence_length + 1):
            hdr_sequence = []
            gray_sequence = []
            for j in range(self.sequence_length):
                frame_dir = frame_list[i + j]
                # 构建 light-*.hdr 文件路径（匹配以 'light-' 开头的 HDR 文件）
                hdr_pattern = os.path.join(frame_dir, 'light-*.hdr')
                hdr_files = sorted(glob.glob(hdr_pattern))
                if not hdr_files:
                    print(f"Warning: No light-*.hdr found in {frame_dir}")
                    continue
                hdr_path = hdr_files[0]  # 如果有多个文件，取第一个

                # 构建 confocal-0-*.png 文件路径
                gray_pattern = os.path.join(frame_dir, 'confocal-0-*.png')
                gray_files = sorted(glob.glob(gray_pattern))
                if not gray_files:
                    print(f"Warning: No confocal-0-*.png found in {frame_dir}")
                    continue
                gray_path = gray_files[0]  # 如果有多个文件，取第一个

                if os.path.exists(hdr_path) and os.path.exists(gray_path):
                    hdr_sequence.append(hdr_path)
                    gray_sequence.append(gray_path)
                else:
                    print(f"Warning: {hdr_path} or {gray_path} does not exist.")
            if len(hdr_sequence) == self.sequence_length and len(gray_sequence) == self.sequence_length:
                self.data_sequences.append((hdr_sequence, gray_sequence))

    def __len__(self):
        return len(self.data_sequences)

    def __getitem__(self, idx):
        """
        获取一个样本

        Args:
            idx (int): 索引

        Returns:
            tuple: (hdr_sequence, gray_sequence)，分别为 HDR 图像序列和灰度图像序列
        """
        hdr_paths, gray_paths = self.data_sequences[idx]
        hdr_images = []
        gray_images = []
        for hdr_path, gray_path in zip(hdr_paths, gray_paths):
            # 读取 HDR 图像
            hdr_img = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if hdr_img is None:
                raise ValueError(f"Failed to read HDR image {hdr_path}")
            hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_RGB2GRAY)
            hdr_img = hdr_img.astype('float32')
            
            # 读取灰度图像
            gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                raise ValueError(f"Failed to read grayscale image {gray_path}")
            gray_img = gray_img.astype('float32')

            # 图像预处理
            if self.transform_hdr:
                hdr_img = self.transform_hdr(hdr_img)
            else:
                # 默认将图像从 HWC 转换为 CHW
                hdr_img = torch.from_numpy(hdr_img.reshape(1000,64,64))
                # hdr_img = hdr_img[400:,:,:]
                hdr_img = hdr_img.permute(1, 2, 0)
                shift_num = self.shift_num
                d1_process = self.d1_process

                # Simulate real acquisition
                output = torch.zeros((16, 16, hdr_img.shape[-1]))
                ind_col_odd = torch.tensor([0, 5, 9, 13, 17, 21, 26, 30, 34, 38, 42, 47, 51, 55, 59], dtype=torch.long)
                ind_row_odd = torch.tensor([4, 13, 21, 30, 38, 46, 55, 63], dtype=torch.long)
                ind_col_even = torch.tensor([5, 9, 13, 17, 21, 26, 30, 34, 38, 42, 47, 51, 55, 59, 63], dtype=torch.long)
                ind_row_even = torch.tensor([0, 9, 17, 26, 34, 42, 51, 59], dtype=torch.long)

                row_grid_even, col_grid_even = torch.meshgrid(ind_row_even, ind_col_even, indexing='ij')
                row_grid_odd, col_grid_odd = torch.meshgrid(ind_row_odd, ind_col_odd, indexing='ij')

                # Vectorized shift operation
                mask = shift_num != 0
                indices = torch.nonzero(mask, as_tuple=True)
                hdr_img[indices] = torch.stack([torch.roll(hdr_img[i, j], shifts=shift_num[i, j].item()) for i, j in zip(*indices)])

                hdr_img *= d1_process

                # Vectorized accumulation operations
                for i in range(4):
                    output[::2, 1:] += hdr_img[row_grid_even, col_grid_even - i] / 4
                    output[1::2, :-1] += hdr_img[row_grid_odd, col_grid_odd + i] / 4

                    for j in range(2, 15, 2):
                        output[j, 0] += hdr_img[ind_row_even[j // 2] - i, 0] / 4
                    for j in range(1, 16, 2):
                        output[j, -1] += hdr_img[ind_row_odd[(j + 1) // 2 - 1] - i, -1] / 4
                        
                hdr_img = hdr_img.permute(2, 0, 1)
                hdr_img = hdr_img[400:,:,:]
                hdr_img = hdr_img[::3,:,:] + hdr_img[1::3,:,:] + hdr_img[2::3,:,:]
                hdr_img = hdr_img / hdr_img.max()
                
            hdr_images.append(hdr_img)

            if self.transform_gray:
                gray_img = self.transform_gray(gray_img)
            else:
                # 将灰度图像增加一个通道维度，形状为 (1, H, W)
                # gray_img = torch.from_numpy(gray_img).unsqueeze(0)
                gray_img = torch.from_numpy(gray_img)
                gray_img = gray_img / gray_img.max()
                
            gray_images.append(gray_img)

        # 将图像序列堆叠为张量
        hdr_sequence = torch.stack(hdr_images)    # 形状为 (sequence_length, C, H, W)
        gray_sequence = torch.stack(gray_images)  # 形状为 (sequence_length, 1, H, W)

        return hdr_sequence.squeeze(), gray_sequence


class NLOSVideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length=3, group_size=50, transform_hdr=None, transform_gray=None):
        """
        初始化数据集

        Args:
            root_dir (str): 数据集的根目录
            sequence_length (int): 序列长度，默认为3（连续三帧）
            group_size (int): 每组帧的数量，默认为50
            transform_hdr (callable, optional): HDR 图像的预处理函数
            transform_gray (callable, optional): 灰度图像的预处理函数
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.group_size = group_size
        self.transform_hdr = transform_hdr
        self.transform_gray = transform_gray
        self.data_sequences = []
        
        self.shift_num = torch.tensor(loadmat('shift_num.mat')['shift_num'].astype(np.float64))
        self.d1_process = torch.tensor(loadmat('D1_process.mat')['D1_process'][..., None])
        time_bin = 10
        self.shift_num /= (time_bin / 4.0)
        self.shift_num = self.shift_num.to(torch.int16)
        
        # 获取所有物体的路径
        object_dirs = [os.path.join(root_dir, obj) for obj in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, obj))]
        for obj_dir in object_dirs:
            # 获取该物体下的所有帧目录，按照修改时间排序
            frame_dirs = [os.path.join(obj_dir, frame) for frame in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, frame))]
            frame_dirs.sort(key=os.path.getmtime)

            # 将帧目录分组，每组 group_size 个帧
            num_frames = len(frame_dirs)
            num_groups = num_frames // self.group_size
            for g in range(num_groups):
                group_frames = frame_dirs[g*self.group_size : (g+1)*self.group_size]
                # 在组内生成连续三帧的序列
                self._generate_sequences(group_frames)

            # 处理剩余不足 group_size 的帧（如果有且数量足够）
            remaining_frames = frame_dirs[num_groups*self.group_size:]
            if len(remaining_frames) >= self.sequence_length:
                self._generate_sequences(remaining_frames)

    def _generate_sequences(self, frame_list):
        """
        在给定的帧列表中生成连续序列

        Args:
            frame_list (list): 帧目录列表
        """
        for i in range(len(frame_list) - self.sequence_length + 1):
            hdr_sequence = []
            gray_sequence = []
            for j in range(self.sequence_length):
                frame_dir = frame_list[i + j]
                # 构建 light-*.hdr 文件路径（匹配以 'light-' 开头的 HDR 文件）
                hdr_pattern = os.path.join(frame_dir, 'light-*.hdr')
                hdr_files = sorted(glob.glob(hdr_pattern))
                if not hdr_files:
                    print(f"Warning: No light-*.hdr found in {frame_dir}")
                    continue
                hdr_path = hdr_files[0]  # 如果有多个文件，取第一个

                # 构建 confocal-0-*.png 文件路径
                gray_pattern = os.path.join(frame_dir, 'confocal-0-*.png')
                gray_files = sorted(glob.glob(gray_pattern))
                if not gray_files:
                    print(f"Warning: No confocal-0-*.png found in {frame_dir}")
                    continue
                gray_path = gray_files[0]  # 如果有多个文件，取第一个

                if os.path.exists(hdr_path) and os.path.exists(gray_path):
                    hdr_sequence.append(hdr_path)
                    gray_sequence.append(gray_path)
                else:
                    print(f"Warning: {hdr_path} or {gray_path} does not exist.")
            if len(hdr_sequence) == self.sequence_length and len(gray_sequence) == self.sequence_length:
                self.data_sequences.append((hdr_sequence, gray_sequence))

    def __len__(self):
        return len(self.data_sequences)

    def __getitem__(self, idx):
        """
        获取一个样本

        Args:
            idx (int): 索引

        Returns:
            tuple: (hdr_sequence, gray_sequence)，分别为 HDR 图像序列和灰度图像序列
        """
        hdr_paths, gray_paths = self.data_sequences[idx]
        hdr_images = []
        gray_images = []
        for hdr_path, gray_path in zip(hdr_paths, gray_paths):
            # 读取 HDR 图像
            hdr_img = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if hdr_img is None:
                raise ValueError(f"Failed to read HDR image {hdr_path}")
            hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_RGB2GRAY)
            hdr_img = hdr_img.astype('float32')
            
            # 读取灰度图像
            gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                raise ValueError(f"Failed to read grayscale image {gray_path}")
            gray_img = gray_img.astype('float32')

            # 图像预处理
            if self.transform_hdr:
                hdr_img = self.transform_hdr(hdr_img)
            else:
                # 默认将图像从 HWC 转换为 CHW
                hdr_img = torch.from_numpy(hdr_img.reshape(1000,64,64))
                # hdr_img = hdr_img[400:,:,:]
                hdr_img = hdr_img.permute(1, 2, 0)
                shift_num = self.shift_num
                d1_process = self.d1_process

                # Simulate real acquisition
                output = torch.zeros((16, 16, hdr_img.shape[-1]))
                ind_col_odd = torch.tensor([0, 5, 9, 13, 17, 21, 26, 30, 34, 38, 42, 47, 51, 55, 59], dtype=torch.long)
                ind_row_odd = torch.tensor([4, 13, 21, 30, 38, 46, 55, 63], dtype=torch.long)
                ind_col_even = torch.tensor([5, 9, 13, 17, 21, 26, 30, 34, 38, 42, 47, 51, 55, 59, 63], dtype=torch.long)
                ind_row_even = torch.tensor([0, 9, 17, 26, 34, 42, 51, 59], dtype=torch.long)

                row_grid_even, col_grid_even = torch.meshgrid(ind_row_even, ind_col_even, indexing='ij')
                row_grid_odd, col_grid_odd = torch.meshgrid(ind_row_odd, ind_col_odd, indexing='ij')

                # Vectorized shift operation
                mask = shift_num != 0
                indices = torch.nonzero(mask, as_tuple=True)
                hdr_img[indices] = torch.stack([torch.roll(hdr_img[i, j], shifts=shift_num[i, j].item()) for i, j in zip(*indices)])

                hdr_img *= d1_process

                # Vectorized accumulation operations
                for i in range(4):
                    output[::2, 1:] += hdr_img[row_grid_even, col_grid_even - i] / 4
                    output[1::2, :-1] += hdr_img[row_grid_odd, col_grid_odd + i] / 4

                    for j in range(2, 15, 2):
                        output[j, 0] += hdr_img[ind_row_even[j // 2] - i, 0] / 4
                    for j in range(1, 16, 2):
                        output[j, -1] += hdr_img[ind_row_odd[(j + 1) // 2 - 1] - i, -1] / 4
                        
                hdr_img = hdr_img.permute(2, 0, 1)
                hdr_img = hdr_img[400:,:,:]
                hdr_img = hdr_img[::3,:,:] + hdr_img[1::3,:,:] + hdr_img[2::3,:,:]
                hdr_img = hdr_img / hdr_img.max()
                
            hdr_images.append(hdr_img)

            if self.transform_gray:
                gray_img = self.transform_gray(gray_img)
            else:
                # 将灰度图像增加一个通道维度，形状为 (1, H, W)
                # gray_img = torch.from_numpy(gray_img).unsqueeze(0)
                gray_img = torch.from_numpy(gray_img)
                gray_img = gray_img / gray_img.max()
                
            gray_images.append(gray_img)

        # 将图像序列堆叠为张量
        hdr_sequence = torch.stack(hdr_images)    # 形状为 (sequence_length, C, H, W)
        gray_sequence = torch.stack(gray_images)  # 形状为 (sequence_length, 1, H, W)

        return hdr_sequence.squeeze(), gray_sequence

class NLOSTransientsDataset(Dataset):
    def __init__(self, root_dir, sequence_length=3, group_size=50, transform_hdr=None, transform_gray=None):
        """
        初始化数据集

        Args:
            root_dir (str): 数据集的根目录
            sequence_length (int): 序列长度，默认为3（连续三帧）
            group_size (int): 每组帧的数量，默认为50
            transform_hdr (callable, optional): HDR 图像的预处理函数
            transform_gray (callable, optional): 灰度图像的预处理函数
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.group_size = group_size
        self.transform_hdr = transform_hdr
        self.transform_gray = transform_gray
        self.data_sequences = []

        # 获取所有物体的路径
        object_dirs = [os.path.join(root_dir, obj) for obj in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, obj))]
        for obj_dir in object_dirs:
            # 获取该物体下的所有帧目录，按照修改时间排序
            frame_dirs = [os.path.join(obj_dir, frame) for frame in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, frame))]
            frame_dirs.sort(key=os.path.getmtime)

            # 将帧目录分组，每组 group_size 个帧
            num_frames = len(frame_dirs)
            num_groups = num_frames // self.group_size
            for g in range(num_groups):
                group_frames = frame_dirs[g*self.group_size : (g+1)*self.group_size]
                # 在组内生成连续三帧的序列
                self._generate_sequences(group_frames)

            # 处理剩余不足 group_size 的帧（如果有且数量足够）
            remaining_frames = frame_dirs[num_groups*self.group_size:]
            if len(remaining_frames) >= self.sequence_length:
                self._generate_sequences(remaining_frames)

    def _generate_sequences(self, frame_list):
        """
        在给定的帧列表中生成连续序列

        Args:
            frame_list (list): 帧目录列表
        """
        for i in range(len(frame_list) - self.sequence_length + 1):
            hdr_sequence = []
            gray_sequence = []
            for j in range(self.sequence_length):
                frame_dir = frame_list[i + j]
                # 构建 light-*.hdr 文件路径（匹配以 'light-' 开头的 HDR 文件）
                hdr_pattern = os.path.join(frame_dir, 'light-*.hdr')
                hdr_files = sorted(glob.glob(hdr_pattern))
                if not hdr_files:
                    print(f"Warning: No light-*.hdr found in {frame_dir}")
                    continue
                hdr_path = hdr_files[0]  # 如果有多个文件，取第一个

                # 构建 confocal-0-*.png 文件路径
                gray_pattern = os.path.join(frame_dir, 'confocal-0-*.png')
                gray_files = sorted(glob.glob(gray_pattern))
                if not gray_files:
                    print(f"Warning: No confocal-0-*.png found in {frame_dir}")
                    continue
                gray_path = gray_files[0]  # 如果有多个文件，取第一个

                if os.path.exists(hdr_path) and os.path.exists(gray_path):
                    hdr_sequence.append(hdr_path)
                    gray_sequence.append(gray_path)
                else:
                    print(f"Warning: {hdr_path} or {gray_path} does not exist.")
            if len(hdr_sequence) == self.sequence_length and len(gray_sequence) == self.sequence_length:
                self.data_sequences.append((hdr_sequence, gray_sequence))

    def __len__(self):
        return len(self.data_sequences)

    def __getitem__(self, idx):
        """
        获取一个样本

        Args:
            idx (int): 索引

        Returns:
            tuple: (hdr_sequence, gray_sequence)，分别为 HDR 图像序列和灰度图像序列
        """
        hdr_paths, gray_paths = self.data_sequences[idx]
        hdr_images = []
        gray_images = []
        for hdr_path, gray_path in zip(hdr_paths, gray_paths):
            # 读取 HDR 图像
            hdr_img = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if hdr_img is None:
                raise ValueError(f"Failed to read HDR image {hdr_path}")
            hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_RGB2GRAY)
            hdr_img = hdr_img.astype('float32')
            
            # 读取灰度图像
            gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                raise ValueError(f"Failed to read grayscale image {gray_path}")
            gray_img = gray_img.astype('float32')

            # 图像预处理
            if self.transform_hdr:
                hdr_img = self.transform_hdr(hdr_img)
            else:
                # 默认将图像从 HWC 转换为 CHW
                hdr_img = torch.from_numpy(hdr_img.reshape(1000,64,64))
                hdr_img = hdr_img[400:,:,:]
                hdr_img = hdr_img / hdr_img.max()
                
            hdr_images.append(hdr_img)

            if self.transform_gray:
                gray_img = self.transform_gray(gray_img)
            else:
                # 将灰度图像增加一个通道维度，形状为 (1, H, W)
                # gray_img = torch.from_numpy(gray_img).unsqueeze(0)
                gray_img = torch.from_numpy(gray_img)
                gray_img = gray_img / gray_img.max()
                
            gray_images.append(gray_img)

        # 将图像序列堆叠为张量
        hdr_sequence = torch.stack(hdr_images)    # 形状为 (sequence_length, C, H, W)
        gray_sequence = torch.stack(gray_images)  # 形状为 (sequence_length, 1, H, W)

        return hdr_sequence.squeeze(), gray_sequence

class NLOSTransientsDatasetTest(Dataset):
    def __init__(self, root_dir, sequence_length=3, group_size=50, transform_hdr=None, transform_gray=None):
        """
        初始化数据集

        Args:
            root_dir (str): 数据集的根目录
            sequence_length (int): 序列长度，默认为3（连续三帧）
            group_size (int): 每组帧的数量，默认为50
            transform_hdr (callable, optional): HDR 图像的预处理函数
            transform_gray (callable, optional): 灰度图像的预处理函数
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.group_size = group_size
        self.transform_hdr = transform_hdr
        self.transform_gray = transform_gray
        self.data_sequences = []

        # 仅获取名称包含 "windmill" 的文件夹路径
        object_dirs = [os.path.join(root_dir, obj) for obj in os.listdir(root_dir)
                       if os.path.isdir(os.path.join(root_dir, obj)) and "K_1_diffuse" in obj]
        
        for obj_dir in object_dirs:
            # 获取该物体下的所有帧目录，按照修改时间排序
            frame_dirs = [os.path.join(obj_dir, frame) for frame in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, frame))]
            frame_dirs.sort(key=os.path.getmtime)

            # 将帧目录分组，每组 group_size 个帧
            num_frames = len(frame_dirs)
            num_groups = num_frames // self.group_size
            for g in range(num_groups):
                group_frames = frame_dirs[g*self.group_size : (g+1)*self.group_size]
                # 在组内生成连续三帧的序列
                self._generate_sequences(group_frames)

            # 处理剩余不足 group_size 的帧（如果有且数量足够）
            remaining_frames = frame_dirs[num_groups*self.group_size:]
            if len(remaining_frames) >= self.sequence_length:
                self._generate_sequences(remaining_frames)

    def _generate_sequences(self, frame_list):
        """
        在给定的帧列表中生成连续序列

        Args:
            frame_list (list): 帧目录列表
        """
        for i in range(len(frame_list) - self.sequence_length + 1):
            hdr_sequence = []
            gray_sequence = []
            for j in range(self.sequence_length):
                frame_dir = frame_list[i + j]
                # 构建 light-*.hdr 文件路径（匹配以 'light-' 开头的 HDR 文件）
                hdr_pattern = os.path.join(frame_dir, 'light-*.hdr')
                hdr_files = sorted(glob.glob(hdr_pattern))
                if not hdr_files:
                    print(f"Warning: No light-*.hdr found in {frame_dir}")
                    continue
                hdr_path = hdr_files[0]  # 如果有多个文件，取第一个

                # 构建 confocal-0-*.png 文件路径
                gray_pattern = os.path.join(frame_dir, 'confocal-0-*.png')
                gray_files = sorted(glob.glob(gray_pattern))
                if not gray_files:
                    print(f"Warning: No confocal-0-*.png found in {frame_dir}")
                    continue
                gray_path = gray_files[0]  # 如果有多个文件，取第一个

                if os.path.exists(hdr_path) and os.path.exists(gray_path):
                    hdr_sequence.append(hdr_path)
                    gray_sequence.append(gray_path)
                else:
                    print(f"Warning: {hdr_path} or {gray_path} does not exist.")
            if len(hdr_sequence) == self.sequence_length and len(gray_sequence) == self.sequence_length:
                self.data_sequences.append((hdr_sequence, gray_sequence))

    def __len__(self):
        return len(self.data_sequences)

    def __getitem__(self, idx):
        """
        获取一个样本

        Args:
            idx (int): 索引

        Returns:
            tuple: (hdr_sequence, gray_sequence)，分别为 HDR 图像序列和灰度图像序列
        """
        hdr_paths, gray_paths = self.data_sequences[idx]
        hdr_images = []
        gray_images = []
        for hdr_path, gray_path in zip(hdr_paths, gray_paths):
            # 读取 HDR 图像
            hdr_img = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if hdr_img is None:
                raise ValueError(f"Failed to read HDR image {hdr_path}")
            hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_RGB2GRAY)
            hdr_img = hdr_img.astype('float32')
            
            # 读取灰度图像
            gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                raise ValueError(f"Failed to read grayscale image {gray_path}")
            gray_img = gray_img.astype('float32')

            # 图像预处理
            if self.transform_hdr:
                hdr_img = self.transform_hdr(hdr_img)
            else:
                # 默认将图像从 HWC 转换为 CHW
                hdr_img = torch.from_numpy(hdr_img.reshape(1000,64,64))
                hdr_img = hdr_img[400:,:,:]
                hdr_img = hdr_img / hdr_img.max()
                
            hdr_images.append(hdr_img)

            if self.transform_gray:
                gray_img = self.transform_gray(gray_img)
            else:
                # 将灰度图像增加一个通道维度，形状为 (1, H, W)
                # gray_img = torch.from_numpy(gray_img).unsqueeze(0)
                gray_img = torch.from_numpy(gray_img)
                gray_img = gray_img / gray_img.max()
                
            gray_images.append(gray_img)

        # 将图像序列堆叠为张量
        hdr_sequence = torch.stack(hdr_images)    # 形状为 (sequence_length, C, H, W)
        gray_sequence = torch.stack(gray_images)  # 形状为 (sequence_length, 1, H, W)

        return hdr_sequence.squeeze(), gray_sequence

if __name__ == '__main__':
 
    import glob

    root_directory = "/data2/NLOSVideo"  # 替换为您的数据集根目录
    batch_size = 4

    # 创建数据集和数据加载器
    dataset = NLOSTransientsDataset(root_dir=root_directory, sequence_length=1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 遍历数据
    for idx, (hdr_sequence, gray_sequence) in enumerate(dataloader):
        # hdr_sequence 的形状为 (batch_size, sequence_length, C, H, W)
        # gray_sequence 的形状为 (batch_size, sequence_length, 1, H, W)
        print(f"Batch {idx}")
        print(f"  HDR sequence shape: {hdr_sequence.shape}")
        print(f"  Grayscale sequence shape: {gray_sequence.shape}")
        # exit()
        # 将 HDR 图像保存为 .mat 文件
        mat_path = '/home/user/NLOS-Video/test.mat'
        data = hdr_sequence[0,:,:,:].squeeze().numpy() 
        data = data / data.max() * 20

        # # 1. 对仿真直方图进行系统时间响应函数（IRF）卷积
        # # 假设系统的时间响应函数为一个高斯函数
        # jitter_std = 6  # 时间抖动的标准差，单位为纳秒
        # # irf_bins = np.arange(-100, 101, 1)  # IRF 的时间范围
        # irf_bins = np.arange(-50, 51, 1)  # IRF 的时间范围
        # irf = np.exp(-0.5 * (irf_bins / jitter_std) ** 2)
        # irf = irf / irf.sum()  # 归一化

        import scipy
        irf = scipy.io.loadmat('./kernel_10ps.mat')
        irf = torch.tensor(irf['kernel_10ps']).squeeze(0)
        irf = irf.numpy()

        # 对整个数据进行卷积（使用 FFT 加速）
        # 注意，fftconvolve 的 axes 参数指定沿哪个轴进行卷积，这里是时间轴（轴 0）
        convolved_histogram = scipy.signal.fftconvolve(data, irf[:, np.newaxis, np.newaxis], mode='same', axes=0)

        # 2. 添加暗计数和背景光噪声
        total_background_rate = 0.3

        # 生成背景噪声，对于每个时间 bin 和像素
        background_counts = total_background_rate * np.ones_like(convolved_histogram)

        # 3. 对总信号应用泊松噪声
        # 将背景噪声添加到卷积后的直方图
        total_signal = convolved_histogram + background_counts

        # 应用泊松噪声
        noisy_histogram = np.random.poisson(total_signal)


        data_fft = torch.fft.fftn(torch.from_numpy(noisy_histogram))
        
        # data_fft[0:5,:,:] = 0
        # data_fft[595:,:,:] = 0
   
        data_fft[200:400,:,:] = 0   
        # 进行逆傅里叶变换
        data_ifft = torch.fft.ifftn(data_fft)

        # 获取实部作为恢复后的数据
        restored_data = torch.real(data_ifft)

        scipy.io.savemat(mat_path, {'hdr_image': noisy_histogram, 'clean':data, 'fftdata':restored_data.numpy()})
        print(f"HDR image has been saved to {mat_path}")

        exit()  # 仅测试一批次
    mat_path = '/home/user/NLOS-Video/test.mat'
    import scipy
    data = scipy.io.loadmat("/home/user/NLOS-Video/nlos_alpaca.mat")
    data = torch.tensor(data['data']).squeeze()
    data = data.numpy()
    data = data.reshape([1024, 64, 64])
    # print(data.shape)
    # exit()
    irf = scipy.io.loadmat('./kernel_10ps.mat')
    irf = torch.tensor(irf['kernel_10ps']).squeeze(0)
    irf = irf.numpy()

    convolved_histogram = scipy.signal.fftconvolve(data, irf[:, np.newaxis, np.newaxis], mode='same', axes=0)

    # 2. 添加暗计数和背景光噪声
    total_background_rate = 0.3

    # 生成背景噪声，对于每个时间 bin 和像素
    background_counts = total_background_rate * np.ones_like(convolved_histogram)

    # 3. 对总信号应用泊松噪声
    # 将背景噪声添加到卷积后的直方图
    total_signal = convolved_histogram + background_counts

    # 应用泊松噪声
    noisy_histogram = np.random.poisson(total_signal)


    data_fft = torch.fft.fftn(torch.from_numpy(noisy_histogram))
    
    # data_fft[0:5,:,:] = 0
    # data_fft[595:,:,:] = 0

    data_fft[200:400,:,:] = 0   
    # 进行逆傅里叶变换
    data_ifft = torch.fft.ifftn(data_fft)

    # 获取实部作为恢复后的数据
    restored_data = torch.real(data_ifft)

    scipy.io.savemat(mat_path, {'hdr_image': noisy_histogram, 'clean':data, 'fftdata':restored_data.numpy()})
    print(f"HDR image has been saved to {mat_path}")