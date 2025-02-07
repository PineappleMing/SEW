import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
# 定义一个包含多个转换步骤的 transform
wsi_transforms = transforms.Compose([
    # 随机改变图像的亮度、对比度和饱和度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # 将PIL图像转换为Tensor
    transforms.ToTensor(),
    # 归一化图像
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
pixel_2_label = {59: 1, 70: 2, 73: 3, 82: 4, 109: 5, 117: 6, 120: 7, 160: 8, 161: 9, 170: 10, 217: 11,
                 255: 12}
pixel_2_label_array = np.vectorize(pixel_2_label.get)(np.arange(256))
class GdphBatchDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx])
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            remapped_mask = pixel_2_label_array[mask]
            plt.imshow(mask)
            plt.show()
            remapped_mask = np.array(remapped_mask, dtype=np.uint8)
            plt.imshow(remapped_mask)
            plt.show()
            image = wsi_transforms(image)
            mask = torch.from_numpy(remapped_mask)
            return image, mask
        except (IOError, SyntaxError, OSError):
            os.remove(self.image_paths[idx])
            print(f"已删除不完整的 JPEG 图片: {self.image_paths[idx]}")
            return None, None
