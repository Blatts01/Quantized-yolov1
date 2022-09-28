import os
import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from xml.etree import ElementTree

from .util import parse_cfg, xywhc2label


class YOLODataset(Dataset):
    def __init__(self, img_path, label_path, S, B, num_classes, transforms=None):
        self.img_path = img_path  # images folder path
        self.label_path = label_path  # labels folder path
        self.transforms = transforms
        self.filenames = os.listdir(img_path)
        self.filenames.sort()
        self.S = S
        self.B = B
        self.num_classes = num_classes

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # read image
        img = cv2.imread(os.path.join(self.img_path, self.filenames[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ori_width, ori_height = img.shape[1], img.shape[0]  # image's original width and height

        img = Image.fromarray(img).convert('RGB')
        img = self.transforms(img)  # resize and to tensor

        # read each image's corresponding label(.txt)
        xywhc = []
        with open(os.path.join(self.label_path, self.filenames[idx].split('.')[0] + '.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    continue
                line = line.strip().split(' ')

                # convert xywh str to float, class str to int
                x, y, w, h, c = float(line[0]), float(line[1]), float(line[2]), float(line[3]), int(line[4])

                xywhc.append((x, y, w, h, c))

        label = xywhc2label(xywhc, self.S, self.B, self.num_classes)  # convert xywhc list to label
        label = torch.Tensor(label)
        return img, label


class PascalVOC(Dataset):
    def __init__(self, data_path, S, B, class_names, transforms=None):
        self.img_path = data_path  # images folder path
        self.label_path = data_path  # labels folder path
        self.transforms = transforms
        for file in os.listdir(data_path):
            if file.endswith(".jpg"):
                self.filenames.append(file)
        self.filenames.sort()
        self.S = S
        self.B = B
        self.num_classes = len(class_names)
        self.class_names = class_names
        self.class_id = dict()
        for count, class_name in enumerate(class_names):
            self.class_id[str(class_name)] = str(count)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # read image
        img = cv2.imread(os.path.join(self.img_path, self.filenames[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img).convert('RGB')
        img = self.transforms(img)  # resize and to tensor

        # read the correpsondenting *.xml file and convert it to yolo
        xywhc = voc2yolo(os.path.join(self.label_path, os.path.splitext(self.filenames[idx])[0] + '.xml'))

        label = xywhc2label(xywhc, self.S, self.B, self.num_classes)  # convert xywhc list to label
        label = torch.Tensor(label)
        return img, label

    def voc2yolo(self, xml_file):
        in_file = open(xml_file)
        tree = ElementTree.parse(in_file)
        size = tree.getroot().find('size')
        height = int(size.find('height').text)
        width = int(size.find('width').text)
        class_exists = False

        for obj in tree.findall('object'):
            name = obj.find('name').text
            if name in self.class_names:
                class_exists = True

        #check for all objects 
        xywhc = []
        if class_exists:
            for obj in tree.findall('object'):
                # ignore difficult objects 
                difficult = obj.find('difficult').text
                if int(difficult) == 1:
                    continue
                xml_box = obj.find('bndbox')
                x_min = float(xml_box.find('xmin').text)
                y_min = float(xml_box.find('ymin').text)
                x_max = float(xml_box.find('xmax').text)
                y_max = float(xml_box.find('ymax').text)

                box_x_center = (x_min + x_max) / 2.0 - 1 # according to darknet annotation
                box_y_center = (y_min + y_max) / 2.0 - 1 # according to darknet annotation
                box_w = x_max - x_min
                box_h = y_max - y_min
                box_x = box_x_center * 1. / width
                box_w = box_w * 1. / width
                box_y = box_y_center * 1. / height
                box_h = box_h * 1. / height

                b = [box_x, box_y, box_w, box_h]
                id = self.class_id[str(obj.find('name').text)]
                xywhc.append((box_x, box_y, box_w, box_h, id))
        return xywhc


def create_dataloader(img_path, label_path, train_proportion, val_proportion, test_proportion, batch_size, input_size,
                      S, B, num_classes):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    # create yolo dataset
    dataset = YOLODataset(img_path, label_path, S, B, num_classes, transforms=transform)

    dataset_size = len(dataset)
    train_size = int(dataset_size * train_proportion)
    val_size = int(dataset_size * val_proportion)
    # test_size = int(dataset_size * test_proportion)
    test_size = dataset_size - train_size - val_size

    # split dataset to train set, val set and test set three parts
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
