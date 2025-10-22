import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import random
from os import listdir
from skimage.transform import resize
from scipy import ndimage

class ImageGenerator:
    def __init__(self, file_path: str, label_path: str, batch_size: int, image_size: list[int], rotation=False, mirroring=False, shuffle=False):
        assert batch_size > 0

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.num_elements = len(listdir(file_path))
        self.cur_element = 0

        self.index_range = list(range(self.num_elements))
        self.cur_epoch = 0

        with open(label_path, 'r') as json_file:
            self.label_dict = json.load(json_file)

        self.label_names = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }

    def next(self):
        images = []
        labels = []

        if self.shuffle:
            for i in range(0, self.batch_size):
                index = -1
                if not self.index_range:
                    index = self.cur_element
                    self.cur_element = (self.cur_element + 1) % self.num_elements
                    
                    self.cur_epoch += 1
                    self.index_range = list(range(self.num_elements))
                    
                else:
                    index = random.sample(self.index_range, 1)[0]
                    self.index_range.remove(index)

                filename = f"{index}.npy"
                img = np.load(self.file_path + filename)
                img = resize(img, (self.image_size[0], self.image_size[1]))

                label = self.label_dict[str(index)]

                images.append(img)
                labels.append(label)
        else:
            for i in range(0, self.batch_size):
                filename = f"{self.cur_element}.npy"
                img = np.load(self.file_path + filename)
                img = resize(img, (self.image_size[0], self.image_size[1]))

                label = self.label_dict[str(self.cur_element)]

                images.append(img)
                labels.append(label)

                self.cur_element = (self.cur_element + 1) % self.num_elements
                
                if self.cur_element == 0:
                    self.cur_epoch += 1
                

        if self.mirroring:
            for i in range(len(images)):
                random_bool = random.choice([True, False])
                if random_bool:
                    images[i] = np.flip(images[i], axis=1)

        if self.rotation:
            for i in range(len(images)):
                random_angle = random.choice([90, 180, 270])
                images[i] = ndimage.rotate(images[i], random_angle, reshape=False, mode='nearest')

        return np.array(images), np.array(labels)

    def augment(self, img):
        random_transformation = random.choice(['mirror', 'rotation', 'mirror_and_rotation'])
        if 'mirror' in random_transformation:
            img = np.flip(img, axis=1)
        elif 'rotation' in random_transformation:
            img = ndimage.rotate(img, random_angle, reshape=False, mode='nearest')
        else:
            img = np.flip(img, axis=1)
            img = ndimage.rotate(img, random_angle, reshape=False, mode='nearest')
            
        return img

    def current_epoch(self):
        return self.cur_epoch

    def class_name(self, x):
        return self.label_names[x]

    def show(self):
        x, y = self.next()
        title = [self.class_name(label) for label in y]

        n_rows = 1
        n_cols = self.batch_size

        for i in range(n_rows * n_cols):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(x[i], cmap='gray')  # Adjust the colormap if needed
            plt.title(title[i])
            plt.axis('off')  # Turn off axis labels

        plt.tight_layout()  # Adjust subplot spacing
        plt.show()
