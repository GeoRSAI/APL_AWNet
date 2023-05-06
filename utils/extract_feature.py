import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import os
import h5py
from .my_utils import single_data_gen_low_level, single_data_wsy

# 提取Hash特征
def index_hash(model, config_transform, classes, csv_path, index_file):


    if config_transform == "default":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = config_transform

    targets, imgs = single_data_wsy(csv_path, classes)


    img_paths = []
    preds = []
    labels = []
    for label, img_path in zip(targets, imgs):
        img = Image.open(img_path).convert('RGB')
        img = transform(img)

        c, h, w = img.size()
        img = img.view(-1, c, h, w)
        use_gpu = torch.cuda.is_available()
        img = img.cuda() if use_gpu else img
        input_img = Variable(img)

        # get the feature
        # pred = get_lbp_data(img_path)
        pred = model(input_img)
        # 量化
        pred = pred.data.cpu()
        pred = torch.sign(pred)
        pred = pred.numpy().flatten()


        # time.sleep(1)
        label = np.argmax(label)

        img_paths.append(img_path)
        labels.append(label)
        preds.append(pred)

        print('extracing features from %s ' % (img_path))

    print('Writing features information to the file')

    img_paths_encode = []
    for word in img_paths:
        img_paths_encode.append(word.encode())

    h5f = h5py.File(index_file, 'w')
    h5f.create_dataset('img_paths_encode', data=img_paths_encode)
    h5f.create_dataset('labels', data=labels)
    h5f.create_dataset('preds', data=preds)
    h5f.close()

# 提取Hash特征
def index_2output_hash(model, config_transform, classes, csv_path, index_file):

    if config_transform == "default":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = config_transform

    targets, imgs = single_data_wsy(csv_path, classes)


    img_paths = []
    preds = []
    labels = []
    for label, img_path in zip(targets, imgs):
        img = Image.open(img_path).convert('RGB')
        img = transform(img)

        c, h, w = img.size()
        img = img.view(-1, c, h, w)
        use_gpu = torch.cuda.is_available()
        img = img.cuda() if use_gpu else img
        input_img = Variable(img)

        # get the feature
        # pred = get_lbp_data(img_path)
        _, pred = model(input_img)
        # 量化
        pred = torch.sign(pred)
        pred = pred.data.cpu()
        # fnorm = torch.norm(pred,p=2,dim=1, keepdim=True)
        # pred = pred.div(fnorm.expand_as(pred))
        pred = pred.numpy().flatten()

        # time.sleep(1)
        label = np.argmax(label)

        img_paths.append(img_path)
        labels.append(label)
        preds.append(pred)

        print('extracing features from %s ' % (img_path))

    print('Writing features information to the file')

    img_paths_encode = []
    for word in img_paths:
        img_paths_encode.append(word.encode())

    h5f = h5py.File(index_file, 'w')
    h5f.create_dataset('img_paths_encode', data=img_paths_encode)
    h5f.create_dataset('labels', data=labels)
    h5f.create_dataset('preds', data=preds)
    h5f.close()
    print('done!')


# 提取 FAH Hash特征
def index_fah(model, config_transform, classes, csv_path, index_file):


    if config_transform == "default":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = config_transform

    targets, imgs = single_data_wsy(csv_path, classes)


    img_paths = []
    preds = []
    labels = []
    for label, img_path in zip(targets, imgs):
        img = Image.open(img_path).convert('RGB')
        img = transform(img)

        c, h, w = img.size()
        img = img.view(-1, c, h, w)
        use_gpu = torch.cuda.is_available()
        img = img.cuda() if use_gpu else img
        input_img = Variable(img)

        # get the feature
        # pred = get_lbp_data(img_path)
        gap_softmax, softmax, hash_codes = model(input_img)
        # 量化
        pred = hash_codes.data.cpu()

        pred = pred.numpy().flatten()
        pred = (pred > 0.5) / 1.0

        # time.sleep(1)
        label = np.argmax(label)

        img_paths.append(img_path)
        labels.append(label)
        preds.append(pred)

        print('extracing features from %s ' % (img_path))

    print('Writing features information to the file')

    img_paths_encode = []
    for word in img_paths:
        img_paths_encode.append(word.encode())

    h5f = h5py.File(index_file, 'w')
    h5f.create_dataset('img_paths_encode', data=img_paths_encode)
    h5f.create_dataset('labels', data=labels)
    h5f.create_dataset('preds', data=preds)
    h5f.close()
    print('done!')

# 提取 FAH Hash特征
def index_fah_2(model, config_transform, classes, csv_path, index_file):


    if config_transform == "default":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = config_transform

    targets, imgs = single_data_wsy(csv_path, classes)


    img_paths = []
    preds = []
    labels = []
    for label, img_path in zip(targets, imgs):
        img = Image.open(img_path).convert('RGB')
        img = transform(img)

        c, h, w = img.size()
        img = img.view(-1, c, h, w)
        use_gpu = torch.cuda.is_available()
        img = img.cuda() if use_gpu else img
        input_img = Variable(img)

        # get the feature
        # pred = get_lbp_data(img_path)
        softmax, hash_codes = model(input_img)
        # 量化
        pred = hash_codes.data.cpu()

        pred = pred.numpy().flatten()
        pred = (pred > 0.5) / 1.0

        # time.sleep(1)
        label = np.argmax(label)

        img_paths.append(img_path)
        labels.append(label)
        preds.append(pred)

        print('extracing features from %s ' % (img_path))

    print('Writing features information to the file')

    img_paths_encode = []
    for word in img_paths:
        img_paths_encode.append(word.encode())

    h5f = h5py.File(index_file, 'w')
    h5f.create_dataset('img_paths_encode', data=img_paths_encode)
    h5f.create_dataset('labels', data=labels)
    h5f.create_dataset('preds', data=preds)
    h5f.close()
    print('done!')


# 用神经网络提取特征
def index_high_level_hdy(model, config_transform, classes, csv_path, index_file):

    gen = single_data_gen_low_level(csv_path, classes)

    if config_transform == "default":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = config_transform

    img_paths = []
    preds = []
    labels = []
    while 1:
        try:
            # get the raw data
            label, img_path = gen.__next__()

            img = Image.open(img_path).convert('RGB')
            img = transform(img)

            c, h, w = img.size()
            img = img.view(-1, c, h, w)
            use_gpu = torch.cuda.is_available()
            img = img.cuda() if use_gpu else img
            input_img = Variable(img)

            # get the feature
            pred = model(input_img)
            pred = pred.cpu().detach().numpy().flatten()

            # time.sleep(1)
            label = np.argmax(label)

            img_paths.append(img_path)
            labels.append(label)
            preds.append(pred)

            print('extracing features from %s ' % (img_path))
        except StopIteration:
            break
    print('Writing features information to the file')

    img_paths_encode = []
    for word in img_paths:
        img_paths_encode.append(word.encode())

    h5f = h5py.File(index_file, 'w')
    h5f.create_dataset('img_paths_encode', data=img_paths_encode)
    h5f.create_dataset('labels', data=labels)
    h5f.create_dataset('preds', data=preds)
    h5f.close()
    print('done!')


# 用神经网络提取特征
def index_high_level(model, config_transform, classes, csv_path, index_file):

    if config_transform == "default":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = config_transform

    targets, imgs = single_data_wsy(csv_path, classes)


    img_paths = []
    preds = []
    labels = []
    for label, img_path in zip(targets, imgs):
        img = Image.open(img_path).convert('RGB')
        img = transform(img)

        c, h, w = img.size()
        img = img.view(-1, c, h, w)
        use_gpu = torch.cuda.is_available()
        img = img.cuda() if use_gpu else img
        input_img = Variable(img)

        # get the feature
        pred = model.get_feature(input_img)
        pred = pred.cpu().detach().numpy().flatten()

        # time.sleep(1)
        label = np.argmax(label)

        img_paths.append(img_path)
        labels.append(label)
        preds.append(pred)

        # print('extracing features from %s ' % (img_path))

    print('Writing features information to the file')

    img_paths_encode = []
    for word in img_paths:
        img_paths_encode.append(word.encode())

    h5f = h5py.File(index_file, 'w')
    h5f.create_dataset('img_paths_encode', data=img_paths_encode)
    h5f.create_dataset('labels', data=labels)
    h5f.create_dataset('preds', data=preds)
    h5f.close()
    print('done!')

# 用神经网络提取2个输出
def index_2output(model, config_transform, classes, csv_path, index_file, position=1):

    gen = single_data_gen_low_level(csv_path, classes)

    if config_transform == "default":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = config_transform

    img_paths = []
    preds = []
    labels = []
    while 1:
        try:
            # get the raw data
            label, img_path = gen.__next__()

            img = Image.open(img_path).convert('RGB')
            img = transform(img)

            c, h, w = img.size()
            img = img.view(-1, c, h, w)
            use_gpu = torch.cuda.is_available()
            img = img.cuda() if use_gpu else img
            input_img = Variable(img)

            # get the feature
            if position == 1:
                _, pred = model(input_img)
            else:
                pred, _ = model(input_img)

            pred = pred.cpu().detach().numpy().flatten()

            # time.sleep(1)
            label = np.argmax(label)

            img_paths.append(img_path)
            labels.append(label)
            preds.append(pred)

            print('extracing features from %s ' % (img_path))
        except StopIteration:
            break
    print('Writing features information to the file')

    img_paths_encode = []
    for word in img_paths:
        img_paths_encode.append(word.encode())

    h5f = h5py.File(index_file, 'w')
    h5f.create_dataset('img_paths_encode', data=img_paths_encode)
    h5f.create_dataset('labels', data=labels)
    h5f.create_dataset('preds', data=preds)
    h5f.close()
    print('done!')

# 提取低层级特征
def index_low_level(get_feature, classes,  csv_path, index_file):
    gen = single_data_gen_low_level(csv_path, classes)
    img_paths = []
    preds = []
    labels = []
    while 1:
        try:
            # get the raw data
            label, img_path = gen.__next__()

            # get the feature
            pred = get_feature(img_path)

            label = np.argmax(label)

            img_paths.append(img_path)
            labels.append(label)
            preds.append(pred)

            print('extracing features from %s ' % (img_path))
        except StopIteration:
            break
    print('Writing features information to the file')

    img_paths_encode = []
    for word in img_paths:
        img_paths_encode.append(word.encode())

    h5f = h5py.File(index_file, 'w')
    h5f.create_dataset('img_paths_encode', data=img_paths_encode)
    h5f.create_dataset('labels', data=labels)
    h5f.create_dataset('preds', data=preds)
    h5f.close()
    print('done!')

if __name__ == '__main__':
    # img_path = r'data\PatternNet\images\airplane\airplane001.jpg'
    # hist_result = get_lbp_data(img_path)
    classes = 38
    image_size = 256

    csv_test_path = r'pn_test.csv'
    csv_imageLib_path = r'pn_train.csv'

    base_path = r'result/pn/feature_test/'
    index_train_file = base_path + 'train_index.h5'
    index_test_file = base_path + 'test_index.h5'

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # model = AlexNet_Hash(hash_bit=64, pretrained=False)
    # model.cuda()
    # model.load_state_dict(torch.load('work_dir/DSH_AlexNet/epoch_100.pth'))
    # model.eval()
    #
    #
    # index_hash(model=model, config_transform="default", classes=38, csv_path=csv_test_path, index_file=index_test_file)


    # index_hash(image_size, classes, csv_imageLib_path, index_train_file)

    # index_low_level(image_size, classes, csv_all, index_test_file)