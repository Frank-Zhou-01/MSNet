import os

import cv2
import torch.nn.functional
from torch.utils.data import Dataset
from utils.image_util import ImageUtil


class HeatTrainDataset(Dataset):
    def __init__(self, data_option: dict) -> None:

        self.__turb_path = data_option['turb_path']
        self.__gt_path = data_option['gt_path']
        self.__turb_list = []
        self.__gt_list = []
        self.__patch_size = data_option['patch_size']

        self.__dirs = os.listdir(self.__turb_path)
        for i in range(len(self.__dirs)):
            turb_images = os.listdir(os.path.join(self.__turb_path, self.__dirs[i], 'turb'))
            turb_images.sort(key=lambda x: int(x[5:-4]))
            turb_images = [os.path.join(self.__turb_path, self.__dirs[i], 'turb', j) for j in turb_images]

            gt_image = os.path.join(self.__gt_path, self.__dirs[i], 'gt.png')
            gt_images = [gt_image for _ in range(len(turb_images))]

            self.__turb_list.extend(turb_images)
            self.__gt_list.extend(gt_images)

        self.__dirs = os.listdir('/home/frank/MTRNet/datasets/Heat_Turb/test')
        for i in range(len(self.__dirs)):
            turb_images = os.listdir(os.path.join(self.__turb_path[:-5], 'test', self.__dirs[i], 'turb'))
            turb_images.sort(key=lambda x: int(x[5:-4]))
            turb_images = [os.path.join(self.__turb_path[:-5], 'test', self.__dirs[i], 'turb', j) for j in turb_images]

            gt_image = os.path.join(self.__turb_path[:-5], 'test', self.__dirs[i], 'gt.png')
            gt_images = [gt_image for _ in range(len(turb_images))]

            self.__turb_list.extend(turb_images)
            self.__gt_list.extend(gt_images)

    def __getitem__(self, index: int) -> tuple:
        """
        根据索引获取训练集中对应的模糊图像和清晰图像
        :param index: 图像的索引
        :return: 成对的模糊图像和清晰图像
        """
        turb_path = self.__turb_list[index]
        gt_path = self.__gt_list[index]

        turb_image = cv2.imread(turb_path)
        gt_image = cv2.imread(gt_path)

        # 对图像进行Resize和翻转处理
        turb_image, gt_image = ImageUtil.image_random_crop(turb_image, gt_image, patch_size=self.__patch_size)
        turb_image, gt_image = ImageUtil.image_flip(turb_image, gt_image, h_flip=True, rotation=True)

        # 将图像转换成Tensor张量
        turb_image = ImageUtil.numpy_to_tensor(turb_image)
        gt_image = ImageUtil.numpy_to_tensor(gt_image)

        # turb_image = torch.unsqueeze(turb_image, dim=0)
        # gt_image = torch.unsqueeze(gt_image, dim=0)

        # turb_image = torch.nn.functional.interpolate(turb_image, size=(256, 256), mode='bilinear')
        # gt_image = torch.nn.functional.interpolate(gt_image, size=(256, 256), mode='bilinear')

        # turb_image = torch.squeeze(turb_image, dim=0)
        # gt_image = torch.squeeze(gt_image, dim=0)

        return turb_image, gt_image

    def __len__(self) -> int:
        """
        获取训练数据集中的图像对数
        """
        return len(self.__turb_list)
