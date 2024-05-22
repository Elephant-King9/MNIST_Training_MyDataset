import os

import torch
from PIL import Image
from torchvision import transforms


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, train, transform=None):
        self.root_path = root_path
        # 判断变化规则
        self.transform = transform

        # 判断是否是训练集
        if train:
            self.data_path = os.path.join(self.root_path, 'training')
        else:
            self.data_path = os.path.join(self.root_path, 'testing')

        self.img_paths = []
        self.labels = []

        # 遍历每个子文件夹（标签）
        for label_dir in os.listdir(self.data_path):
            label_path = os.path.join(self.data_path, label_dir)
            if os.path.isdir(label_path):  # 只处理目录
                # 遍历子文件夹中的所有图像文件
                for img_name in os.listdir(label_path):
                    single_img_path = os.path.join(label_path, img_name)
                    # 将单个图片路径添加到img_paths中
                    self.img_paths.append(single_img_path)
                    # 将图片对应的标签添加到labels中
                    self.labels.append(int(label_dir))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_PIL = Image.open(img_path).convert('L')
        # 如果变化规则不为空
        if self.transform is not None:
            img_tensor = self.transform(img_PIL)
        else:
            img_tensor = img_PIL
        # 确定对应下标的标签
        label = self.labels[index]
        return img_tensor, label


if __name__ == '__main__':
    root_dir = '../datasets/mnist_png'
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    my_dataset = MyDataset(root_dir, train=True, transform=transform)
