import torch
import os
import cPickle
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class DatasetProcessingCIFAR_10(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        fp_label = open(label_filepath, 'r')
        labels = [int(x.strip()) for x in fp_label]
        fp_label.close()
        self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([self.label[index]])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)

class Crossmodal(Dataset):
    def __init__(self, img_filename, label_filename):
        with open(img_filename, 'rb') as f:
            self.feat = cPickle.load(f)
            self.feat = [torch.from_numpy(v) for v in self.feat]
            self.feat = torch.stack(self.feat, 0).type(torch.LongTensor)
            print(str(type(self.feat)) + str(self.feat.shape))
        with open(label_filename, 'rb') as f:
            self.label = cPickle.load(f)
            self.label = torch.LongTensor(self.label).unsqueeze(1)



    def __getitem__(self, index):
        feat = self.feat[index, :]
        label = self.label[index, :]
        return feat, label, index

    def __len__(self):
        return len(self.feat)

class DatasetProcessingNUS_WIDE(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)

class DatasetProcessingFlickr(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)
class DatasetProcessingMS_COCO(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)

