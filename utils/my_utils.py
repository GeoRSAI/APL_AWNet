'''
this file contains some functions that used in train_and_val.py
'''
from PIL import Image
import numpy as np
import os
import csv
import cv2
from sklearn.utils import shuffle
import copy
import random
# import tensorflow as tf

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def write_csv_hdy(image_folder, csv_train_path, csv_test_path, ratio=0.2):
    """
    write the image path and category in the folder to a file
    default split 20% of all data into test csv
    :param image_folder:contains one subfolder for each category
    :param csv_train_path:path to save the train data
    :param csv_test_path:path to save the test data
    :param ratio: split ratio
    the csv is formatted by following
        image_name,image_path,category
    such as
        .\dataset\EuroSAT\AnnualCrop\AnnualCrop_1.jpg,0
    """

    writer_train = csv.writer(open(csv_train_path, 'a', newline='', encoding='utf8'))
    writer_test = csv.writer(open(csv_test_path, 'a', newline='', encoding='utf8'))

    current_category = 0

    for floder in os.listdir(image_folder):
        floder = os.path.join(image_folder, floder)
        each_img_files = os.listdir(floder)
        each_img_total = len(each_img_files)
        offset = int(each_img_total * ratio)
        # offset = 50
        # randomly shuffles list
        np.random.shuffle(each_img_files)
        test_img_files = each_img_files[:offset]
        tain_img_files = each_img_files[offset:]
        for testfile in test_img_files:
            print('writing test data: ' + testfile)
            abs_path = os.path.join(floder, testfile)
            writer_test.writerow([abs_path, current_category])

        for train_file in tain_img_files:
            print('writing train data: ' + train_file)
            abs_path = os.path.join(floder, train_file)
            writer_train.writerow([abs_path, current_category])
        current_category += 1

def split_train_val_hdy(csv_total_train, csv_train_split_path, csv_val_path, ratio=0.2):
    """
    write the image path and category in the folder to a file
    default split 20% of all data into test csv
    :param image_folder:contains one subfolder for each category
    :param csv_train_path:path to save the train data
    :param csv_test_path:path to save the test data
    :param ratio: split ratio
    the csv is formatted by following
        image_name,image_path,category
    such as
        .\dataset\EuroSAT\AnnualCrop\AnnualCrop_1.jpg,0
    """

    writer_train = csv.writer(open(csv_train_split_path, 'a', newline='', encoding='utf8'))
    writer_test = csv.writer(open(csv_val_path, 'a', newline='', encoding='utf8'))

    content = []
    dict_val = {}
    for img_path, current_category in csv.reader(open(csv_total_train, 'r', encoding='utf8')):
        if current_category in dict_val:
            content.append(img_path)
            print(img_path)
            dict_val[current_category] = copy.copy(content)
        else:
            content.clear()
            print(img_path)
            content.append(img_path)
            dict_val[current_category] = copy.copy(content)
        # content.append((img_path, current_category))
    sort_dic = sorted(dict_val.items(),key=lambda x:eval(x[0]))
    print('结束')
    for key, value in sort_dic:
        each_img_files = value
        each_img_total = len(each_img_files)

        offset = int(each_img_total * ratio)
        # offset = 50
        # randomly shuffles list
        random.shuffle(each_img_files)
        test_img_files = each_img_files[:offset]
        tain_img_files = each_img_files[offset:]
        for testfile in test_img_files:
            print('writing test data: ' + testfile)
            writer_test.writerow([testfile, key])

        for train_file in tain_img_files:
            print('writing train data: ' + train_file)
            writer_train.writerow([train_file, key])


def write_csv(image_folder, csv_train_path, csv_test_path):
    '''
    write the image path and category in the folder to a file
    default split 30% of all data into test csv
    :param image_folder:contains one subfolder for each category
    :param csv_train_path:path to save the train data
    :param csv_test_path:path to save the test data
    the csv is formatted by follow
        image_path,category
    such as
        .\dataset\EuroSAT\AnnualCrop\AnnualCrop_1.jpg,0
    '''
    writer_train = csv.writer(open(csv_train_path, 'a', newline='', encoding='utf8'))
    writer_test = csv.writer(open(csv_test_path, 'a', newline='', encoding='utf8'))

    count = 0  # for split test and train data
    current_category = 0

    for floder in os.listdir(image_folder):
        floder = os.path.join(image_folder, floder)
        np.random.shuffle(floder)
        for file in os.listdir(floder):
            print('writing down the ' + file)
            abs_path = os.path.join(floder, file)

            if count in [1, 2]:
                writer_test.writerow([abs_path, current_category])
                count += 1
            elif count == 9:
                writer_train.writerow([abs_path, current_category])
                count = 0
            else:
                writer_train.writerow([abs_path, current_category])
                count += 1
        current_category += 1


def write_all_csv(image_folder, csv_path):

    writer_csv = csv.writer(open(csv_path, 'a', newline='', encoding='utf8'))

    count = 0  # for split test and train data
    current_category = 0

    for floder in os.listdir(image_folder):
        floder = os.path.join(image_folder, floder)
        for file in os.listdir(floder):
            print('writing down the ' + file)
            abs_path = os.path.join(floder, file)

            writer_csv.writerow([abs_path, current_category])
        current_category += 1

def process_single(img_path, img_size):
    '''
    function to process single image,contains 'convert to rgb' and 'resize'
    :param img_path:single image path
    :param img_size:target image size
    :return: numpy array of input image
    '''
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize((img_size, img_size), Image.ANTIALIAS)
    img_array = np.array(img)
    return img_array


def batch_data_gen(csv_path, batch_size, num_classes, img_size):
    '''
    generator that yield shuffled data fit the batch size
    :param csv_path:path that contains input image data
    :param batch_size:target batch size
    :param num_classes:total number of classes.
    :param img_size:target image size
    :return:numpy ndarray (data,label),note that label is one-hot formation.
    '''
    content = []  # list for shuffle
    while 1:
        X = []
        Y = []

        for img_path, current_category in csv.reader(open(csv_path, 'r', encoding='utf8')):
            content.append((img_path, current_category))

        content = shuffle(content)  # do shuffle

        count = 0  # variable used to count the batch size
        for img_path, current_category in content:
            img_array = process_single(img_path, img_size)
            X.append(img_array)
            Y.append(current_category)
            count += 1
            if count == batch_size:
                count = 0
                data = np.stack(X, axis=0)
                label = np.stack(Y, axis=0)
                label = to_categorical(label, num_classes=num_classes)
                yield data, label
                X = []
                Y = []



def single_data_gen(csv_path, num_classes, img_size):
    '''
    generator that yield single image data
    :param csv_path:path that contains input image data
    :param num_classes:total number of classes.
    :param img_size:target image size
    :return:data, label, img_path;note that label is one-hot formation.
    '''
    # while 1:
    #     for img_path, current_category in csv.reader(open(csv_path, 'r', encoding='utf8')):
    #         data = process_single(img_path, img_size)
    #         label = to_categorical(current_category, num_classes=num_classes)
    #         yield data, label, img_path
    #     raise StopIteration
    for img_path, current_category in csv.reader(open(csv_path, 'r', encoding='utf8')):
        if img_path.strip():
            data = process_single(img_path, img_size)
            label = to_categorical(current_category, num_classes=num_classes)
            yield data, label, img_path
        # if img_path.strip():
        #     print(img_path)

def single_data_gen_low_level(csv_path, num_classes):
    '''
    generator that yield single image data
    :param csv_path:path that contains input image data
    :param num_classes:total number of classes.
    :param img_size:target image size
    :return:data, label, img_path;note that label is one-hot formation.
    '''
    while 1:
        for img_path, current_category in csv.reader(open(csv_path, 'r', encoding='utf8')):
            label = to_categorical(current_category, num_classes=num_classes)

            yield label, img_path
        raise StopIteration

def single_data_wsy(csv_path, num_classes):
    '''
    generator that yield single image data
    :param csv_path:path that contains input image data
    :param num_classes:total number of classes.
    :param img_size:target image size
    :return:data, label, img_path;note that label is one-hot formation.
    '''
    img_paths = []
    labels = []
    for img_path, current_category in csv.reader(open(csv_path, 'r', encoding='utf8')):
        label = to_categorical(current_category, num_classes=num_classes)

        img_paths.append(img_path)
        labels.append(label)

    return labels, img_paths


def load_data(csv_path, num_classes, img_size):
    '''
    this function is writing for small dataset that can be loaded into memory directly
    :param csv_path:path that contains input image data
    :param num_classes:total number of classes.
    :param img_size:target image size
    :return:numpy ndarray (data,label),note that label is one-hot formation.
    '''
    X = []
    Y = []
    for img_path, current_category in csv.reader(open(csv_path, 'r', encoding='utf8')):
        img_array = process_single(img_path, img_size)
        X.append(img_array)
        Y.append(current_category)

    data = np.stack(X, axis=0)
    label = np.stack(Y, axis=0)
    label = to_categorical(label, num_classes=num_classes)

    return data, label


def distance(featureA, featureB):
    '''
    Euclidean distance of two feature
    :param featureA:
    :param featureB:
    :return:Euclidean distance (float)
    '''
    featureA = featureA.flatten()
    featureB = featureB.flatten()

    a = np.square(featureA - featureB)
    # np.square获取矩阵的平方
    return np.sqrt(np.sum(np.square(featureA - featureB)))

def hamming_distance(featureA, featureB):
    '''
    Euclidean distance of two feature
    :param featureA:
    :param featureB:
    :return:Euclidean distance (float)
    '''
    featureA = featureA.flatten()
    featureB = featureB.flatten()
    dist = np.linalg.norm(featureA - featureB)
    return dist


def get_topK(k, dict, target_label,retrieval_result_file):
    '''
    get the top K images ranked by distance
    :param k:number of returned image
    :param dict:
        dict = {'image_path':image_path,
                content:{
                    'dis':distance,
                    'label':label
                }}
    :param target_label:the label of target image (int)
    '''
    f = open(retrieval_result_file,'a')

    num_right = 0
    num_total = 0
    for image_path, content in dict:
        distance = content['dis']
        label = content['label']

        result = image_path + ';  dis:' + str(distance) + '; label:' + str(label)+'\n'
        f.write(result)

        if target_label == label: num_right += 1

        num_total += 1

        if num_total == k: break

    correct = '正确率是' + str(num_right / num_total)
    f.write(correct)
    print(correct)

    f.close()

def get_lines_count(filename):
    '''
    Gets the count of lines in filename
    :return:count (int)
    '''
    count = 0
    fp = open(filename, "r", encoding='utf-8')
    while 1:
        buffer = fp.read(8 * 1024 * 1024)
        if not buffer:
            break
        count += buffer.count('\n')
    fp.close()
    return count


def cv2pil(cv_img_path):
    """将opencv格式的图片转换为PIL格式"""
    img = cv2.imread(cv_img_path)
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return pil_image


def pil2cv(pil_img_path):
    """将PIL格式图片转换为opencv格式"""
    image = Image.open(pil_img_path)
    cv_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return cv_img


def pil2cv_2(pil_img):
    """将PIL格式图片转换为opencv格式"""
    cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return cv_img


def covert_rgb_img(con_img_path):
    """将图片转换为RGB格式"""
    img = Image.open(con_img_path)
    img = img.convert('RGB')
    return img


def change_image_channels(image):
    """四通道转换为3通道图片"""
    # 4通道转3通道
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
        # image.save(image_path)
    elif image.mode != 'RGB':
        image = image.convert("RGB")
    return image


def change_image_rgb(image_path):
    """
    根据图片路径将图片转换为CV格式
    :param image_path:
    :return:
    """
    img = Image.open(image_path)
    pilime= change_image_channels(img)
    cv_img = pil2cv_2(pilime)
    return cv_img


# def stats_graph(graph):
#     flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
#     params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
#     print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

if __name__ == '__main__':
    image_folder = r'D:\txq00000000000000000000000000\图像检索\遥感数据集\rsir_torch\dataset\UCMD'
    # all_folder = r'E:\tif\bing'
    # csv_train_split_path = r'result\vgg16\mix\mix_train_split.csv'
    # csv_val_path = r'result\vgg16\mix\mix_train_val.csv'
    # csv_total_train = r'result\vgg16\mix\mix_train.csv'
    # csv_test_path = r'result\vgg16\AID50_20\mix_test.csv'
    csv_train_split_path = r'train.csv'
    csv_val_path = r'val.csv'
    csv_total_train = r'database.csv'
    csv_test_path = r'test.csv'


    img_size = 256

    write_all_csv(image_folder, 'merge.csv')

    # write_csv(image_folder, csv_train_path, csv_test_path)
    # write_csv(image_folder, csv_total_train, csv_test_path)
    # split_train_val_hdy(csv_total_train, csv_train_split_path, csv_val_path, ratio=0.2)
    write_csv_hdy(image_folder, csv_total_train, csv_test_path, ratio=0.2)
    split_train_val_hdy(csv_total_train, csv_train_split_path, csv_val_path, ratio=0.2)
    # single_data_gen(csv_test_path, 20, 256)


