import os

import cv2
from torch.utils.data import Dataset
from utils.image_util import ImageUtil


class TurbTrainDataset(Dataset):
    def __init__(self, data_option: dict, patch_size: int) -> None:

        self.__turb_path = data_option['turb_path']
        self.__gt_path = data_option['gt_path']
        print(self.__turb_path)
        self.__turb_list = []
        self.__gt_list = []
        self.__patch_size = patch_size

        self.__dirs = os.listdir(self.__turb_path)
        for i in range(len(self.__dirs)):
            turb_images = os.listdir(os.path.join(self.__turb_path, self.__dirs[i]))
            turb_images.sort(key=lambda x: int(x[:-4]))
            turb_images = [os.path.join(self.__turb_path, self.__dirs[i], j) for j in turb_images]

            gt_images = os.listdir(os.path.join(self.__gt_path, self.__dirs[i]))
            gt_images.sort(key=lambda x: int(x[:-4]))
            gt_images = [os.path.join(self.__gt_path, self.__dirs[i], j) for j in gt_images]

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

        # 对图像进行裁剪和翻转处理
        turb_image, gt_image = ImageUtil.image_random_crop(turb_image, gt_image, self.__patch_size)
        turb_image, gt_image = ImageUtil.image_flip(turb_image, gt_image, h_flip=True, rotation=True)

        # 将图像转换成Tensor张量
        turb_image = ImageUtil.numpy_to_tensor(turb_image)
        gt_image = ImageUtil.numpy_to_tensor(gt_image)

        return turb_image, gt_image

    def __len__(self) -> int:
        """
        获取训练数据集中的图像对数
        """
        return len(self.__turb_list)


class TurbTestData(Dataset):
    def __init__(self, data_option: dict) -> None:

        self.__turb_path = data_option['turb_path']
        self.__gt_path = data_option['gt_path']
        self.__turb_list = []
        self.__gt_list = []

        self.__dirs = os.listdir(self.__turb_path)
        for i in range(len(self.__dirs)):
            turb_images = os.listdir(os.path.join(self.__turb_path, self.__dirs[i]))
            turb_images.sort(key=lambda x: int(x[:-4]))
            turb_images = [os.path.join(self.__turb_path, self.__dirs[i], j) for j in turb_images]

            gt_images = os.listdir(os.path.join(self.__gt_path, self.__dirs[i]))
            gt_images.sort(key=lambda x: int(x[:-4]))
            gt_images = [os.path.join(self.__gt_path, self.__dirs[i], j) for j in gt_images]

            self.__turb_list.append(turb_images)
            self.__gt_list.append(gt_images)

    def __getitem__(self, index: int) -> tuple:
        """
        根据索引获取测试集中对应的模糊图像和清晰图像
        :param index: 图像的索引
        :return: 成对的模糊图像和清晰图像
        """
        turb_path = self.__turb_list[index]
        gt_path = self.__gt_list[index]

        turb_image = cv2.imread(turb_path)
        gt_image = cv2.imread(gt_path)

        # 将图像转换成Tensor张量
        turb_image = ImageUtil.numpy_to_tensor(turb_image)
        gt_image = ImageUtil.numpy_to_tensor(gt_image)

        return turb_image, gt_image

    def __len__(self) -> int:
        """
        获取测试数据集中的图像对数
        """
        return len(self.__turb_list)
