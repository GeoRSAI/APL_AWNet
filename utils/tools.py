import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import csv
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
from tqdm import tqdm
from scipy.spatial.distance import cdist

def config_dataset(config):

    if config["dataset"] == "PatternNet":
        config["n_class"] = 38
    elif config["dataset"] == "VGoogle":
        config["n_class"] = 38
    elif config["dataset"] == "VBing":
        config["n_class"] = 38
    elif config["dataset"] == "Merge":
        config["n_class"] = 38
    elif config["dataset"] == "VArcGIS":
        config["n_class"] = 38
    elif config["dataset"] == "NWPU":
        config["n_class"] = 45
    elif config["dataset"] == "UCMD":
        config["n_class"] = 21
    elif config["dataset"] == "Natural":
        config["n_class"] = 6
    elif config["dataset"] == "AID":
        config["n_class"] = 30
    else:
        config["n_class"] = 38

    transform = image_transform(config["transform"], config['dataset'])
    train_data = MyDataset_csv_onehot(txt=config["train_path"], transform=transform)

    config["num_train"] = len(train_data)

    return config

def text_create(path, name):
    full_path = path + name + '.txt'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')
    # file.write(msg)
    file.close()
    return full_path

def image_transform(config_transform, dataset):

    if config_transform == "default":
        if dataset == 'AID':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    else:
        transform = config_transform

    return transform


class MyDataset_csv_onehot(Dataset):

    def __init__(self, txt, transform=None, n_class=38):
        with open(txt, 'r', encoding='UTF-8') as fh:
            reader = csv.reader(fh)
            imgs = []
            label = []
            # for line in fh:
            for line in reader:
                imgs.append((line[0], int(line[1])))  # imgs中包含有图像路径和标签
                label.append(line[1])
        self.n_class = n_class
        self.imgs = imgs
        self.transform = transform
        self.label = label
    def __getitem__(self, index):
        img_path, label = self.imgs[index]

        # print(os.path.abspath(os.path.join(os.getcwd(), "../")))
        img_path = os.path.join("../", img_path) # 和图像文件放置相对路径有关
        # 调用定义的loader方法
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label_onehot = np.eye(self.n_class, dtype=np.int8)[np.array(label)]
        return img, label, label_onehot, index

    def __len__(self):
        return len(self.imgs)

def get_data(config):
    transform = image_transform(config["transform"], config['dataset'])

    train_data = MyDataset_csv_onehot(txt=config["train_path"], transform=transform, n_class=config['n_class'])
    val_data = MyDataset_csv_onehot(txt=config["val_path"], transform=transform, n_class=config['n_class'])
    test_data = MyDataset_csv_onehot(txt=config["test_path"], transform=transform, n_class=config['n_class'])
    database = MyDataset_csv_onehot(txt=config["database_path"], transform=transform, n_class=config['n_class'])

    train_loader = util_data.DataLoader(train_data, batch_size=config["batch_size"], num_workers=8, shuffle=True, drop_last=True)
    val_loader = util_data.DataLoader(val_data, batch_size=config["batch_size"], num_workers=8, shuffle=False)
    test_loader = util_data.DataLoader(test_data, batch_size=config["batch_size"], num_workers=8, shuffle=False)
    database_loader = util_data.DataLoader(database, batch_size=config["batch_size"], num_workers=8, shuffle=False)

    return train_loader, val_loader, test_loader, database_loader, \
           len(train_data), len(val_data), len(test_data), len(database)


def generate_binary_distribution(batchsize, dim):
    z_batch = np.zeros((batchsize, dim))
    for b in range(batchsize):
        value_zeros_ones = np.zeros((dim))
        for i in range(dim):
            if i < dim // 2:
                value_zeros_ones[i] = 0.
            else:
                value_zeros_ones[i] = 1.
        index = np.arange(dim)
        np.random.shuffle(index)
        z_batch[b, ...] = value_zeros_ones[index]
    return z_batch

# 计算分类精度
def calculate_classification_accuracy(predict, target):
    accu = (predict.argmax(dim=1) == target).float().mean()
    return accu

# draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
#               9000, 9500, 10000]

draw_range = [1, 5, 10, 20, 30, 50, 100, 500, 1000, 1500, 2000, 2500, 3000,
              3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]
# draw_range = [i * 10 for i in range(1, 100)]

def cos(A, B=None):
    """cosine"""
    An = normalize(A, norm='l2', axis=1)
    if (B is None) or (B is A):
        return np.dot(An, An.T)
    Bn = normalize(B, norm='l2', axis=1)
    return np.dot(An, Bn.T)


def hamming(A, B=None):
    """A, B: [None, bit]
    elements in {-1, 1}
    """
    if B is None: B = A
    bit = A.shape[1]
    return (bit - A.dot(B.T)) // 2


def euclidean(A, B=None, sqrt=False):
    aTb = np.dot(A, B.T)
    if (B is None) or (B is A):
        aTa = np.diag(aTb)
        bTb = aTa
    else:
        aTa = np.diag(np.dot(A, A.T))
        bTb = np.diag(np.dot(B, B.T))
    D = aTa[:, np.newaxis] - 2.0 * aTb + bTb[np.newaxis, :]
    if sqrt:
        D = np.sqrt(D)
    return D

# 计算P-R精度曲线
def pr_curve2(qF, rF, qL, rL, draw_range=draw_range):
    #  https://blog.csdn.net/HackerTom/article/details/89425729
    n_query = qF.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    Rank = np.argsort(CalcHammingDist(qF, rF))
    P, R = [], []
    for k in tqdm(draw_range):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all
        P.append(np.mean(p))
        R.append(np.mean(r))
    return P, R

# 画 P-R 曲线
def pr_curve(qF, rF, qL, rL, what=1, topK=-1):
    n_query = qF.shape[0]
    if topK == -1 or topK > rF.shape[0]:  # top-K 之 K 的上限
        topK = rF.shape[0]

    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    if what == 0:
        # Rank = np.argsort(cdist(qF, rF, 'cosine'))
        Rank = np.argsort(1 - cos(qF, rF))
    elif what == 1:
        Rank = np.argsort(hamming(qF, rF))
    elif what == 2:
        Rank = np.argsort(euclidean(qF, rF))

    P, R = [], []
    for k in range(1, topK + 1):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = np.zeros(n_query)  # 各 query sample 的 Precision@R
        r = np.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all

        P.append(np.mean(p))
        R.append(np.mean(r))

    # 画 P-R 曲线
    # fig = plt.figure(figsize=(5, 5))
    # plt.plot(R, P)  # 第一个是 x，第二个是 y
    # plt.grid(True)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # plt.legend()
    # plt.show()

    return P, R


# 计算hash特征
def compute_result_hash(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, _, cls, _ in tqdm(dataloader):
        clses.append(cls)
        hash = net(img.to(device))
        # a = hash.data.cpu().numpy()
        bs.append(hash.data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_hash_MBE(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, _, cls, _ in tqdm(dataloader):
        clses.append(cls)
        _, H, code = net(img.to(device))
        # code = torch.sign(H)
        # a = hash.data.cpu().numpy()
        bs.append(code.data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

# 计算非hash特征
def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, _, cls, _ in tqdm(dataloader):
        clses.append(cls)
        output = net(img.to(device))
        a = output.data.cpu().flatten(start_dim=1).numpy()
        bs.append(output.data.cpu().flatten(start_dim=1))
    return torch.cat(bs), torch.cat(clses)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(
            1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

