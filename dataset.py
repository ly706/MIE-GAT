from __future__ import print_function

import json
import logging
import random
import os
import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
import transforms
import torchnet as tnt
from config.data_config import config
import pandas as pd
import inflect

"""# LIDC with attribute
class GATData(data.Dataset):  # 继承Dataset
    def __init__(self, X_test, y_test, char_test, log, partition='train', fold=1, split=1):
        self.cf = config()
        self.partition = partition
        self.log = log
        self.fold = fold
        self.split = split
        self.translate = inflect.engine()
        self.data_size = [self.cf.crop_size, self.cf.crop_size, self.cf.crop_size]
        self.train_files, self.y, self.character = X_test, y_test, char_test
        self.class_num = 2
        # set transformer
        if self.partition == 'train':
            angle = [90, 180, 270]
            padding = [4, 6, 8]
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomZFlip(),
                                            transforms.RandomYFlip(), transforms.TransverseSection(),
                                            transforms.SagittalSection(),
                                            transforms.RandomCrop(self.cf.crop_size, padding=np.random.choice(padding, 1)[0]),
                                            transforms.RandomRotation(np.random.choice(angle, 1)[0])])

        else:  # 'val' or 'test' ,
            self.transform = None

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, index): 
        actual_index = index
        img = self.train_files[actual_index]
        struct_fea = np.array(self.character[actual_index])
        if self.transform:  
            img = self.transform(img)
            img = img.astype(float)
        id = actual_index
        label = self.y[actual_index]  # 获取该图片的label
        sample = {'img': img, 'struct_fea': struct_fea, 'label': label, 'id': id,
                  "desc_feature": struct_fea}  # 根据图片和标签创建字典
        return sample  """



"""# LIDC 
class GATData(data.Dataset):  # 继承Dataset
    def __init__(self, log, partition='train', fold=1, split=1):
        self.cf = config()
        self.partition = partition
        self.log = log
        self.fold = fold
        self.split = split
        self.translate = inflect.engine()
        self.data_size = [self.cf.crop_size, self.cf.crop_size, self.cf.crop_size]
        self.id_labels_features = pd.read_excel(self.cf.path_label, sheet_name=0, header=0,
                                                usecols=[0, 5, 7],
                                                names=["id", "length", "malignant"])

        self.class_num = 2
        # set transformer
        if self.partition == 'train':
            angle = [90, 180, 270]
            padding = [4, 6, 8]
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomZFlip(),
                                            transforms.RandomYFlip(), transforms.TransverseSection(),
                                            transforms.SagittalSection(),
                                            transforms.RandomCrop(self.cf.crop_size, padding=np.random.choice(padding, 1)[0]),
                                            transforms.RandomRotation(np.random.choice(angle, 1)[0])])

        else:  # 'val' or 'test' ,
            self.transform = None

        # load data
        self.log.info('Loading LIDC dataset -phase {}'.format(self.partition))

        class_names, images_path, ids = self.get_image_paths(self.partition)
        self.labels = class_names
        self.data = images_path
        self.struct_feas = None
        self.ids = ids

    def get_image_paths(self, partition):
        print("------------------------get_image_paths-----------------------------------")
        with open(self.cf.path_split[self.split - 1], 'r') as f:
            split = json.load(f)

        datatest = split['test']
        datatra = []
        for i in range(1, self.cf.cross_validation_num + 1):
            if i == self.fold:
                # 列表中的每一个元素都是一个三维向量的id，表示一个CT样本(3D图像)的id
                dataval = split['fold' + str(i)]
                # dataval = [i for i in val if 'aug' not in i]
            else:
                datatra = datatra + (split['fold' + str(i)])

        images_path, class_names, ids = [], [], []

        if partition == 'train':
            sum = int(len(datatra) * self.cf.train_ratio)
            files = datatra[:sum]
        elif partition == 'val':
            files = dataval
        elif partition == 'test':
            files = datatest

        class_0 = 0
        class_1 = 1
        for file in files:
            path = os.path.join(self.cf.path_nodule, file + ".npy")
            id_label = self.id_labels_features[self.id_labels_features["id"] == file]  
            class_ = int(id_label.iloc[0, 2])  
            if class_ == 0:
                class_0 += 1
            else:
                class_1 += 1
            images_path.append(path)
            class_names.append(class_)
            ids.append(file)
        self.log.info("The total number: " + str(len(class_names)))
        self.log.info("The number of images of class 0: {}".format(class_0))
        self.log.info("The number of images of class 1: {}".format(class_1))
        return class_names, images_path, ids

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)

    def __getitem__(self, index): 
        image_path = self.data[index]  # 根据索引index获取该图片
        img = np.load(image_path)  # 读取该图片
        
        crop_size = self.cf.crop_size
        bgx = int(img.shape[0] / 2 - crop_size / 2)
        bgy = int(img.shape[1] / 2 - crop_size / 2)
        bgz = int(img.shape[2] / 2 - crop_size / 2)
        img = np.array(img[bgx:bgx + crop_size, bgy:bgy + crop_size, bgz:bgz + crop_size], dtype='float32')

        if self.transform:  
            img = self.transform(img)
            img = img.astype(float)
        id = self.ids[index]

        label = self.labels[index]  

        sample = {'img': img, 'struct_fea': img, 'label': label, 'id': id}  

        return sample """


# LIDP with Attribute
class GATData(data.Dataset):  # 继承Dataset
    def __init__(self, log, partition='train', fold=1, split=1):
        self.cf = config()
        self.partition = partition
        self.log = log
        self.fold = fold
        self.split = split
        self.translate = inflect.engine()
        self.data_size = [self.cf.crop_size, self.cf.crop_size, self.cf.crop_size]
        self.id_labels_features = pd.read_excel(self.cf.path_label, sheet_name=0, header=0,
                                                usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12],
                                                names=["id", "position", "burr", "lobed", "PleuralPull", "length",
                                                       "type", "malignant", "calcification", "gender", "age"])

        self.class_num = 2
        if self.partition == 'train':
            angle = [90, 180, 270]
            padding = [4, 6, 8]
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomZFlip(),
                                            transforms.RandomYFlip(), transforms.TransverseSection(),
                                            transforms.SagittalSection(),
                                            transforms.RandomCrop(self.cf.crop_size, padding=np.random.choice(padding, 1)[0]),
                                            transforms.RandomRotation(np.random.choice(angle, 1)[0])])

        else:  # 'val' or 'test' ,
            self.transform = None

        # load data
        self.log.info('Loading LIDP dataset -phase {}'.format(self.partition))

        if self.partition == 'all':
            class_names, images_path, struct_feas, ids = self.get_all_image_paths()
        else:
            class_names, images_path, struct_feas, ids = self.get_image_paths(self.partition)

        self.labels = class_names
        self.data = images_path
        self.struct_feas = struct_feas
        self.ids = ids

    def get_image_paths(self, partition):
        print("------------------------get_image_paths-----------------------------------")
        with open(self.cf.path_split[self.split - 1], 'r') as f:
            split = json.load(f)

        datatest = split['test']
        datatra = []
        for i in range(1, self.cf.cross_validation_num + 1):
            if i == self.fold:
                dataval = split['fold' + str(i)]
            else:
                datatra = datatra + (split['fold' + str(i)])

        images_path, class_names, struct_feas, ids = [], [], [], []

        if partition == 'train':
            sum = int(len(datatra) * self.cf.train_ratio)
            files = datatra[:sum]
        elif partition == 'val':
            files = dataval
        elif partition == 'test':
            files = datatest

        class_0 = 0
        class_1 = 1
        for file in files:
            path = os.path.join(self.cf.path_nodule, file + ".npy")
            id_label = self.id_labels_features[self.id_labels_features["id"] == file]  # 获取该图片的label
            class_ = int(id_label.iloc[0, 7])  # 获取良恶性标签
            if class_ == 0:
                class_0 += 1
            else:
                class_1 += 1
            struct_fea = id_label.iloc[0, [1, 2, 3, 4, 5, 6, 8, 9, 10]].values.tolist()
            struct_fea = struct_fea[:self.cf.struct_num]
            images_path.append(path)
            class_names.append(class_)
            struct_feas.append(struct_fea)
            ids.append(file)

        self.log.info("The total number: " + str(len(class_names)))
        self.log.info("The number of images of class 0: {}".format(class_0))
        self.log.info("The number of images of class 1: {}".format(class_1))


        return class_names, images_path, struct_feas, ids


    def get_all_image_paths(self):
        print("------------------------get_image_paths-----------------------------------")
        # 加载训练集和验证集数据
        with open(self.cf.path_split[self.split - 1], 'r') as f:
            split = json.load(f)

        datatest = split['test']
        for i in range(1, self.cf.cross_validation_num + 1):
                datatest = datatest + (split['fold' + str(i)])

        images_path, class_names, struct_feas, ids = [], [], [], []

        files = datatest

        class_0 = 0
        class_1 = 1
        for file in files:
            path = os.path.join(self.cf.path_nodule, file + ".npy")
            id_label = self.id_labels_features[self.id_labels_features["id"] == file]
            class_ = int(id_label.iloc[0, 7])  # 获取良恶性标签
            if class_ == 0:
                class_0 += 1
            else:
                class_1 += 1
            # print('class:', class_)
            struct_fea = id_label.iloc[0, [1, 2, 3, 4, 5, 6, 8, 9, 10]].values.tolist()
            struct_fea = struct_fea[:self.cf.struct_num]
            images_path.append(path)
            class_names.append(class_)
            struct_feas.append(struct_fea)
            ids.append(file)

        self.log.info("The total number: " + str(len(class_names)))
        self.log.info("The number of images of class 0: {}".format(class_0))
        self.log.info("The number of images of class 1: {}".format(class_1))

        return class_names, images_path, struct_feas, ids

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_path = self.data[index]  # 根据索引index获取该图片
        img = np.load(image_path)  # 读取该图片

        crop_size = self.cf.crop_size
        bgx = int(img.shape[0] / 2 - crop_size / 2)
        bgy = int(img.shape[1] / 2 - crop_size / 2)
        bgz = int(img.shape[2] / 2 - crop_size / 2)
        img = np.array(img[bgx:bgx + crop_size, bgy:bgy + crop_size, bgz:bgz + crop_size], dtype='float32')

        struct_fea = np.array(self.struct_feas[index])

        if self.transform:
            img = self.transform(img)
            img = img.astype(float)

        id = self.ids[index]

        label = self.labels[index]

        sample = {'img': img, 'struct_fea': struct_fea, 'label': label, 'id': id}

        return sample