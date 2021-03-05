# v1.0.0
from glob import glob
from random import randint, choice
from cv2 import imread, resize, cvtColor, inRange, COLOR_BGR2GRAY, COLOR_BGR2HSV, COLOR_GRAY2RGB, INTER_AREA
from numpy import array

class gtsrb:
    def __init__(self, data_path, image_final_size):
        self.path = data_path
        self.size = image_final_size
        self.training_images = []
        self.training_labels = []
        self.test_images = []
        self.test_labels = []
        self.all_files = []

    def load_data(self, wb_mode):
        for i in range(6):
            self.all_files.append(glob(self.path + '\\' + str(i) + '\\*.ppm'))
        all_files_copy = self.all_files
        while all_files_copy != [[], [], [], [], [], []]:
            global_choice = randint(0, 5)
            try:
                prec_choice = choice(all_files_copy[global_choice])
            except IndexError:
                continue
            all_files_copy[global_choice].pop(all_files_copy[global_choice].index(prec_choice))
            image = imread(prec_choice)
            res_z = (self.size, self.size)
            image = resize(image, res_z, interpolation=INTER_AREA)
            if wb_mode:
                image = cvtColor(image, COLOR_BGR2GRAY)
            #image = 255 - inRange(image, (30, 0, 100), (255, 120, 255))
            #image = 255 - cvtColor(image, COLOR_BGR2GRAY)
            if randint(0, 7) == 0:
                self.test_images.append(image)
                self.test_labels.append(global_choice)
            else:
                self.training_images.append(image)
                self.training_labels.append(global_choice)

        return [(array(self.training_images), array(self.training_labels)), (array(self.test_images), array(self.test_labels))]
