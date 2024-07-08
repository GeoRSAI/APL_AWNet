import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import datetime
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import logger
import torch.optim as optim
from utils.tools import get_data, config_dataset, calculate_classification_accuracy
from network.classify.resnet import resnet50, resnet18, resnet101
from utils.extract_feature import index_high_level
from utils.cal_precision_util import execute_retrieval
from network.loss.loss_new import NCEandMAE, NCEandRCE, SCELoss, NFLandRCE, NFLandMAE
from awnet import AWNet
import math as m

torch.multiprocessing.set_sharing_strategy('file_system')


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# seed_torch()

def sample_weight(output, target):
    output_sort = output.argsort(descending=True)
    output_sort_list = output_sort.cpu().numpy().tolist()
    target_list = target.tolist()
    weight = []
    for ele in output_sort_list:
        index = output_sort_list.index(ele)
        if output[index][ele[0]] == output[index][target_list[index]]:
            pn = output[index][ele[1]]
        else:
            pn = output[index][ele[0]]
        delta = output[index][target_list[index]] - pn
        weight.append(np.float64(delta))

    return weight

def get_config():
    config = {
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1.5e-4, "weight_decay": 1.5e-4}},
        "info": "[ResNet50]",
        # "info": "[ResNet18]",
        # "info": "[ResNet101]",
        'transform': 'default',
        "batch_size":128,
        "net": resnet50,
        # "net": resnet18,
        # "net": resnet101,
        "pretrained": False,
        "dataset": "UCMD",
        "train_path": '/home/sdb/txq/rsir_AWNet/dataset/UCMD/noisy_train_0.2.csv',
        "val_path": '/home/sdb/txq/rsir_AWNet/dataset/UCMD/noisy_val_0.2.csv',
        "test_path": '/home/sdb/txq/rsir_AWNet/dataset/UCMD/test.csv',
        "database_path": '/home/sdb/txq/rsir_AWNet/dataset/UCMD/database.csv',
        "save_path": "/home/sdb/txq/rsir_AWNet/multi_loss_NWPU_128_0.2/A-NCE&MAE/symmexc_0.2/",
        "epoch": 20,
        "test_map": 1,
        "device": torch.device("cuda:0")
    }
    config = config_dataset(config)
    return config

def train_val(config, criterion):

    log = logger.get_logger(log_path)
    log.info('start training! Time is {}'.format(log_time))

    device = config["device"]

    train_loader, val_loader, test_loader, dataset_loader, num_train, num_val, num_test, num_dataset = get_data(config)

    model = config["net"](
        num_classes=config['n_class'],
        pretrained=True
    ).to(device)

    # model = torchvision.models.densenet169(True, num_classes=config['n_class']).to(device)
    # model = torchvision.models.mobilenet_v3_large(True, num_classes=config['n_class']).to(device)
    # model = torchvision.models.mobilenet_v3_small(True, num_classes=config['n_class']).to(device)

    if config["pretrained"]:
        print('load model...')
        model.load_state_dict(torch.load(config["pretrained"], map_location='cuda:0'))
    awnet = AWNet(2, 100, 2).cuda()
    optimizer = config["optimizer"]["type"](model.parameters(), **(config["optimizer"]["optim_params"]))
    optimizer_awnet = torch.optim.Adam(awnet.params(), 1e-3, weight_decay=1e-4)

    best_value = 0
    for epoch in range(1, config["epoch"]+1):
        model.train()
        description = "train " + str(epoch) + "/" + str(config["epoch"])
        train_loss, train_accuracy = 0.0, 0.0
        with tqdm(train_loader, desc=description) as iterator:
            for i, (data, target, target_onehot, ind) in enumerate(iterator):
                optimizer.zero_grad() # 将梯度归0
                var_list = []
                entropy_list = []
                entropy_d_var_list = []

                data, target, target_onehot = data.to(device), target.to(device), target_onehot.to(device)
                output = model(data)
                output_ls = output.tolist()
                output_s = F.softmax(output)
                output_np = output_s.tolist()
                if epoch < 4:          #为提高样本评价结果可靠性，前3个epoch采用固定权重(α=1，β=1)
                    cost = NCEandMAE(1, 1, num_classes=config['n_class'])
                else:
                    weight = sample_weight(output_s, target)
                    for i in range(len(output_ls)):

                        var = m.sqrt(np.var(output_ls[i]))     # S
                        var_list.append(var)

                        entropy = -1.0*torch.sum(torch.tensor(output_np[i]*np.log(output_np[i])))   # Entropy
                        entropy = entropy.cpu().detach().numpy()
                        entropy_list.append(entropy.tolist())

                        entropy_d_var = entropy/var                          #EDS
                        entropy_d_var_list.append(entropy_d_var)

                    w_v = list(zip(entropy_d_var_list,weight))
                    w_v_tensor = torch.tensor(w_v, dtype=torch.float).to(device)
                    ab = awnet(w_v_tensor)
                    # 逐样本赋权
                    a, b = ab.chunk(2, 1)
                    a_tensor = a.squeeze()
                    b_tensor = b.squeeze()
                    cost = NCEandMAE(a_tensor, b_tensor, num_classes=config['n_class'])

                loss = cost(output, target)
                accuracy = calculate_classification_accuracy(output, target)

                # 主模型的参数更新
                loss.backward()
                optimizer.step()

                # AWNet参数更新
                output_2 = model(data)
                loss2 = cost(output_2, target)
                optimizer_awnet.zero_grad()
                loss2.backward()
                optimizer_awnet.step()

                information = "Loss: {:.4f}, Accuracy: {:.2f}".format(loss.item(), accuracy)
                iterator.set_postfix_str(information)

                train_loss += loss.item()
                train_accuracy += accuracy

        train_loss = train_loss / len(train_loader)
        train_accuracy = train_accuracy /len(train_loader)
        # log日志 训练信息写入
        log.info('Train Epoch:[{}/{}] loss={:.4f} Accuracy={:.4f}'.format(epoch, config["epoch"], train_loss, train_accuracy))
        # Tensorboard 训练信息写入
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # 测试检索精度
        if epoch % config["test_map"] == 0:
            # 保存模型
            torch.save(model.state_dict(), root_path + 'epoch_' + str(epoch) + '.pth')
            model.eval()
            description = "val " + str(epoch) + "/" + str(config["epoch"])
            val_loss, val_accuracy = 0.0, 0.0
            with torch.no_grad():
                with tqdm(val_loader, desc=description) as iterator:
                    for i, (data, target, target_onehot, ind) in enumerate(iterator):
                        data, target, target_onehot = data.to(device), target.to(device), target_onehot.to(device)
                        output = model(data)

                        loss = criterion(output, target)
                        accuracy = calculate_classification_accuracy(output, target)

                        information = "Val Loss: {:.4f}, Accuracy: {:.2f}".format(loss.item(), accuracy)
                        iterator.set_postfix_str(information)

                        val_loss += loss.item()
                        val_accuracy += accuracy

            val_loss = val_loss / len(val_loader)
            val_accuracy = val_accuracy / len(val_loader)

            log.info('Val Epoch:[{}/{}] loss={:.4f} Accuracy={:.4f}'.format(epoch, config["epoch"], val_loss,
                                                                              val_accuracy))
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)

            # 保存精度最优模型
            if val_accuracy > best_value:
                best_value = val_accuracy
                torch.save(model.state_dict(), root_path + 'best_model.pth')
            os.remove(root_path + 'epoch_' + str(epoch) + '.pth')
    # 关闭Summary_Writer，写入Log日志
    writer.close()
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    log.info('End training! End Time is {} Elapsed Time is {}'.format(end_time, elapsed_time))

    model.load_state_dict(torch.load(config["save_path"]+'best_model.pth', map_location='cuda:0'))

    model.eval()

    train_index = config["save_path"] + 'train_index.h5'
    test_index = config["save_path"] + 'test_index.h5'

    index_high_level(model=model, config_transform=config["transform"], classes=config["n_class"], csv_path=config["test_path"], index_file=test_index)
    index_high_level(model=model, config_transform=config["transform"], classes=config["n_class"], csv_path=config["database_path"], index_file=train_index)

    log.info('End Extract feature! End Time is {}'.format(datetime.datetime.now()))

    execute_retrieval(save_path=config["save_path"], pools=10, classes=config["n_class"])
    log.info('End Retrieval! End Time is {}'.format(datetime.datetime.now()))

if __name__ == "__main__":

    config = get_config()

    root_path = config["save_path"]
    start_time = datetime.datetime.now()
    log_time = datetime.datetime.strftime(start_time, '%Y-%m-%d %H-%M-%S')
    log_path = root_path + log_time + '.log'

    # 定义Summary_Writer
    writer_path = root_path + log_time
    writer = SummaryWriter(writer_path)

    criterion = nn.CrossEntropyLoss()
    train_val(config=config, criterion=criterion)
