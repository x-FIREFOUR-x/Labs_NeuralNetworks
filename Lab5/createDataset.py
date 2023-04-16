import cv2
import glob
import os
import shutil
import random

from arguments import DATASET_PATH, ORIGIN_DATA_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, OTHER_CLASS, CLASSES_NAMES

COUNT_FILES = 500
IMAGE_KERNEL = 11
MAX_SHIFT = 20
COMBINED_SIZE = 150
MAX_OPERATION = 3


def blur(img):
    blurred = cv2.GaussianBlur(img, (IMAGE_KERNEL, IMAGE_KERNEL), 0)
    return blurred, 'blurred'



def rotate90(img):
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return rotated, 'rotated_90'


def rotate180(img):
    rotated = cv2.rotate(img, cv2.ROTATE_180)
    return rotated, 'rotated_180'


def rotate270(img):
    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated, 'rotated_270'


def flip_vertical(img):
    flipped = cv2.flip(img, 0)
    return flipped, 'flip_0'


def flip_horizontal(img):
    flipped = cv2.flip(img, 1)
    return flipped, 'flip_1'


def grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    return rgb_img, 'gray'


def shift_x(img):
    rand_shift = random.randint(MAX_SHIFT, MAX_SHIFT)
    coppied = img.copy()
    width = coppied.shape[1]
    right_border = width + rand_shift
    left_border = abs(rand_shift)
    if rand_shift < 0:
        coppied[:, 0:right_border] = coppied[:, left_border:width]
    else:
        coppied[:, left_border:width] = coppied[:, 0:width - rand_shift]
    return coppied, 'shift_x_' + str(rand_shift)


def shift_y(img):
    rand_shift = random.randint(MAX_SHIFT, MAX_SHIFT)
    coppied = img.copy()
    height = coppied.shape[0]
    bottom_border = height + rand_shift
    top_border = abs(rand_shift)
    if rand_shift < 0:
        coppied[0:bottom_border] = coppied[top_border:height]
    else:
        coppied[top_border:height] = coppied[0:height - rand_shift]
    return coppied, 'shift_y_' + str(rand_shift)


def crop(img):
    coppied = img.copy()
    height = coppied.shape[0]
    width = coppied.shape[1]
    left_border = random.randint(1, MAX_SHIFT)
    right_border = width - random.randint(1, MAX_SHIFT)
    top_border = random.randint(1, MAX_SHIFT)
    bottom_border = height - random.randint(1, MAX_SHIFT)
    coppied = coppied[top_border:bottom_border, left_border:right_border]
    coppied = cv2.resize(coppied, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return coppied, 'cropped_' + str(top_border) + '_' + str(right_border) + '_' + str(bottom_border) + '_' + str(
        left_border)


def combine_with_others(img):
    files = glob.glob(ORIGIN_DATA_PATH + "/" + OTHER_CLASS + "/*.jpg")
    files_count = len(files)
    rand_file = random.randint(0, files_count - 1)
    name = os.path.split(files[rand_file])[-1]
    name = name.split('.')[0]
    resized_height = COMBINED_SIZE
    resized_width = COMBINED_SIZE
    coppied = cv2.resize(img, (resized_height, resized_width))
    rand_file_img = cv2.imread(files[rand_file])
    height, width = (IMAGE_HEIGHT, IMAGE_WIDTH)
    rand_file_img = cv2.resize(rand_file_img, (height, width))
    rand_file_pos_x = random.randint(0, width - resized_width)
    rand_file_pos_y = random.randint(0, height - resized_height)
    rand_file_img[rand_file_pos_y:rand_file_pos_y + resized_height,
    rand_file_pos_x:rand_file_pos_x + resized_width] = coppied
    return rand_file_img, 'combined_' + name + '_' + str(rand_file_pos_y) + '_' + str(rand_file_pos_x)


def generate(img, operations):
    random.shuffle(operations)
    num_of_operations = random.randint(1, MAX_OPERATION)
    new_image = img
    new_name = ''
    for i in range(num_of_operations):
        new_image, name = operations[i](new_image)
        new_name += name
    return new_image, new_name


def generate_dataset(num, input_folder, output_folder):
    init_files = glob.glob(input_folder + "/*.jpg")
    for i in range(len(init_files)):
        img = cv2.imread(init_files[i])
        name = os.path.split(init_files[i])[-1]
        cv2.imwrite(output_folder + "/" + name, img)

    generated_files = set()
    operations = [blur, rotate90, rotate180, rotate270, flip_vertical, flip_horizontal,
                  grayscale, shift_x, shift_y, crop, combine_with_others]
    while len(generated_files) < num:
        rand_file = random.randint(0, len(init_files) - 1)
        img = cv2.imread(init_files[rand_file])
        name = os.path.split(init_files[rand_file])[-1]
        name = name.split('.')[0]
        new_img, new_name = generate(img, operations)
        full_name = name + '_' + new_name
        if not (full_name in generated_files):
            generated_files.add(full_name)
            cv2.imwrite(output_folder + "/" + full_name + ".jpg", new_img)


if __name__ == '__main__':
    for Class in CLASSES_NAMES:
        input_folder = ORIGIN_DATA_PATH + "/" + Class
        output_folder = DATASET_PATH + "/" + Class
        if os.path.isdir(output_folder):
            shutil.rmtree(output_folder)
        os.mkdir(output_folder)

        generate_dataset(COUNT_FILES, input_folder, output_folder)
