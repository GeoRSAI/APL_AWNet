import os
import random
import numpy as np
import torch
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import logger
import torch.optim as optim
from utils.tools import get_data, config_dataset, calculate_classification_accuracy, text_create
from network.classify.resnet import resnet50, resnet18, resnet101
from utils.extract_feature import index_high_level
from utils.cal_precision_util import execute_retrieval
from network.loss.loss_new import NCEandMAE, NCEandRCE, NFLandRCE, NFLandMAE, SCELoss, FocalLoss,\
    NormalizedFocalLoss, NormalizedCrossEntropy, GeneralizedCrossEntropy, ReverseCrossEntropy, \
    MeanAbsoluteError
torch.multiprocessing.set_sharing_strategy('file_system')

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# seed_torch()

def get_config():
    config = {
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1.5e-4, "weight_decay": 1.5e-4}},
        "info": "[ResNet50]",
        # "info": "[ResNet18]",
        # "info": "[ResNet101]",
        'transform': 'default',
        "batch_size":128,
        # "net": resnet18,
        "net": resnet50,
        # "net": resnet101,
        "dataset": 'UCMD',
        "train_path": '/home/sdb/txq/rsir_AWNet/dataset/UCMD/nosiy_train_0.2.csv',
        "val_path": '/home/sdb/txq/rsir_AWNet/dataset/UCMD/nosiy_val_0.2.csv',
        "test_path": '/home/sdb/txq/rsir_AWNet/dataset/UCMD/test.csv',
        'database_path': '/home/sdb/txq/rsir_AWNet/dataset/UCMD/database.csv',
        "save_path": "/home/sdb/txq/rsir_AWNet/multi_loss_UCMD_128_0.2/SCELoss/symmexc_0.2/",
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

    optimizer = config["optimizer"]["type"](model.parameters(), **(config["optimizer"]["optim_params"]))

    best_value = 0
    for epoch in range(1, config["epoch"]+1):
        model.train()
        description = "train " + str(epoch) + "/" + str(config["epoch"])
        train_loss, train_accuracy = 0.0, 0.0

        with tqdm(train_loader, desc=description) as iterator:
            for i, (data, target, target_onehot, ind) in enumerate(iterator):
                optimizer.zero_grad() # 将梯度归0

                data, target, target_onehot = data.to(device), target.to(device), target_onehot.to(device)
                output = model(data)

                loss = criterion(output, target)
                accuracy = calculate_classification_accuracy(output, target)

                loss.backward()  # 反向传播计算得到每个参数的梯度值
                optimizer.step() # 通过梯度下降执行一步参数更新

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

    #pr_path = text_create(root_path, 'pr')

    # print(config)

    # criterion = nn.CrossEntropyLoss()
    criterion = SCELoss(1, 1, num_classes=config['n_class'])
    # criterion = FocalLoss()
    # criterion = NormalizedFocalLoss(num_classes=config['n_class'])
    # criterion = GeneralizedCrossEntropy(num_classes=config['n_class'])
    # criterion = ReverseCrossEntropy(num_classes=config['n_class'])
    # criterion = NormalizedCrossEntropy(num_classes=config['n_class'])
    # criterion = MeanAbsoluteError(num_classes=config['n_class'])
    # criterion = NCEandRCE(1, 0.1, num_classes=config['n_class'])
    # criterion = NCEandMAE(10, 0.1, num_classes=config['n_class'])
    # criterion = NFLandRCE(10, 0.1, num_classes=config['n_class'])
    # criterion = NFLandMAE(10, 0.1, num_classes=config['n_class'])
    train_val(config=config, criterion=criterion)
