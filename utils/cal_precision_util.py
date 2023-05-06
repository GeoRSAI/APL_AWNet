import csv
import h5py
import utils.my_utils as my_utils
import numpy as np
import utils.metric as metric
from collections import Counter
import time
import os

from multiprocessing import Process,Manager
from multiprocessing import Pool


def retrieval_h5py_thread(query_feature, query_labels,query_img_names,database_feature,database_labels,database_img_names):
    """
    为实现多线程，构造的函数：一类查询一个线程
    :param query_feature:
    :param query_labels:
    :param query_img_names:
    :param database_feature:
    :param database_labels:
    :param database_img_names:
    :return:
    """
    print(str(query_labels[0]) + '类别图像开始检索...')
    accus = []
    for index, (query_temp_feature, query_temp_label, query_temp_name) in enumerate(zip(query_feature, query_labels, query_img_names)):

        query_result_dict = {}
        query_result_dict.setdefault(query_temp_name, []).append(-1)
        query_result_dict.setdefault(query_temp_name, []).append(query_temp_label)
        query_result_dict.setdefault(query_temp_name, []).append(1)

        for database_temp_feature, database_temp_label, data_temp_name in zip(database_feature,database_labels,database_img_names):
            relevance_flag = 0 # 0表示不相关；1表示相关

            distance = my_utils.distance(database_temp_feature, query_temp_feature)

            query_result_dict.setdefault(data_temp_name, []).append(distance)
            query_result_dict.setdefault(data_temp_name, []).append(database_temp_label)
            if query_temp_label == database_temp_label:
                relevance_flag = 1
            query_result_dict.setdefault(data_temp_name, []).append(relevance_flag)

        sort_result = sorted(query_result_dict.items(), key=lambda x: x[1][0])

        accu = []
        for i, val in enumerate(sort_result):
            rele = val[1][2]
            accu.append(rele)

        if index == 0:
            accu_np = np.array(accu)
            accu_np = np.expand_dims(accu_np, axis=0)
            accus = accu_np
        else:
            accu_np = np.array(accu)
            accu_np = np.expand_dims(accu_np, axis=0)
            accus = np.concatenate((accus, accu_np), axis=0)

    accus = np.delete(accus, [0], axis=1)

    print(str(query_labels[0]) + '类别图像检索结束')

    return accus, str(query_labels[0])



def retrieval_h5py_by_thread(db_index_file, query_index_file, classes, pools=10):
    """
    利用h5f文件检索计算所有查询的距离
    :param db_index_file: 图像数据库的索引文件路径
    :param distance_file: 待保存的距离字典路径，距离字典文件形式为[(图片名称,[距离，实际类标签，是否相关])]
    :return: 无返回
    """
    h5f = h5py.File(db_index_file, 'r')
    img_paths_encode = h5f['img_paths_encode'][:]
    database_feature = h5f['preds'][:]
    database_labels = h5f['labels'][:]
    h5f.close()
    database_img_names = np.char.decode(img_paths_encode.astype(np.string_))

    #读取待检索图片的特征
    h5f = h5py.File(query_index_file, 'r')
    query_img_names = h5f['img_paths_encode'][:]
    query_feature = h5f['preds'][:]
    query_labels = h5f['labels'][:]
    h5f.close()
    query_img_names = np.char.decode(query_img_names.astype(np.string_))

    query_count_dict = Counter(query_labels)
    print(query_count_dict)
    query_class_count = len(query_count_dict)
    threads = []
    start_query = 0
    end_query = 0

    main_pool = Pool(pools)
    results = []
    for i in range(query_class_count):
        if i == 0:
            result = main_pool.apply_async(retrieval_h5py_thread,args=(
                query_feature[:query_count_dict[i]],query_labels[:query_count_dict[i]],
                query_img_names[:query_count_dict[i]],database_feature,database_labels,database_img_names, ))

            results.append(result)
            start_query = start_query + query_count_dict[i]
        else:
            end_query = start_query + query_count_dict[i]

            result = main_pool.apply_async(retrieval_h5py_thread, args=(
                query_feature[start_query:end_query], query_labels[start_query:end_query],
                query_img_names[start_query:end_query], database_feature, database_labels, database_img_names,))

            results.append(result)
            start_query = end_query
    main_pool.close()
    main_pool.join()

    names = locals()
    for i in results:
        rele, ind = i.get()
        names['prec' + str(ind)] = rele

    matrix = []
    for i in range(classes):
        if i == 0:
            temp = names.get('prec' + str(i))
            matrix = temp
        else:
            temp = names.get('prec' + str(i))
            matrix = np.concatenate((matrix, temp), axis=0)

    print('..........检索结束..........')
    return matrix

def get_ng_k(database_file):
    """
    计算在图像库中与查询qi实际相关的个数ng
    计算k=min(4*ng,2M) m=max{Ng(q1),Ng(q2)，....,Ng(qn)}
    :param database_file:数据库的索引
    :return:标签：个数字典和所有查询对应的最大相似个数
    """
    h5f = h5py.File(database_file, 'r')
    database_labels = h5f['labels'][:]
    h5f.close()
    label_count_dict = Counter(database_labels)
    all_query_max_count = max(label_count_dict.values())
    return label_count_dict, all_query_max_count


def get_query_label(query_file):
    """
    返回待查询图片的标签列表
    :param query_file: 待查询图片的索引列表
    :return: 返回标签列表
    """
    h5f = h5py.File(query_file, 'r')
    query_labels = h5f['labels'][:]
    h5f.close()
    return query_labels

def cal_precision(all_rel_list, database_index_path, query_index_path, metric_path):
    """
    根据已保存的距离文件、图像数据特征库、待查询图像特征库计算各种精度
    包括ANMRR/mAP/P@5/P@10/P@20/P@50/P@100/P@1000
    :param distance_path: 排序后的距离文件，由retrieval函数火点
        :param database_index_path: 图像数据特征库文件路径
    :param query_index_path: 待查询图像特征库文件
    :param metric_path: 保存精度的csv文件路径
    :return:
    """
    writer_metric = csv.writer(open(metric_path, 'a', newline='', encoding='utf8'))
    label_count, max_count = get_ng_k(database_index_path)
    query_label_list = get_query_label(query_index_path)

    nmrr_list = []
    writer_metric.writerow(['NMRR', 'AP', 'P@5', 'P@10', 'P@20', 'P@50', 'P@100', 'P@1000'])
    p_5_list = []
    p_10_list = []
    p_20_list = []
    p_50_list = []
    p_100_list = []
    p_1000_list = []
    q_ap_list = []
    for q_rel, q_label in zip(all_rel_list, query_label_list):
        p_5 = metric.precision_at_k(q_rel, 5)
        p_10 = metric.precision_at_k(q_rel, 10)
        p_20 = metric.precision_at_k(q_rel, 20)
        p_50 = metric.precision_at_k(q_rel, 50)
        p_100 = metric.precision_at_k(q_rel, 100)
        p_1000 = metric.precision_at_k(q_rel, 1000)
        q_ap = metric.average_precision(q_rel)
        k_value = min(4 * label_count[q_label], 2 * max_count)
        q_avr, q_mrr, q_nmrr = metric.nmrr(q_rel, label_count[q_label], k_value)
        writer_metric.writerow([q_nmrr, q_ap, p_5, p_10, p_20, p_50, p_100, p_1000])
        nmrr_list.append(q_nmrr)
        p_5_list.append(p_5)
        p_10_list.append(p_10)
        p_20_list.append(p_20)
        p_50_list.append(p_50)
        p_100_list.append(p_100)
        p_1000_list.append(p_1000)
        q_ap_list.append(q_ap)
    a_p_5 = np.mean(p_5_list)
    a_p_10 = np.mean(p_10_list)
    a_p_20 = np.mean(p_20_list)
    a_p_50 = np.mean(p_50_list)
    a_p_100 = np.mean(p_100_list)
    a_p_1000 = np.mean(p_1000_list)
    m_q_ap = np.mean(q_ap_list)
    q_anmrr = metric.anmrr(nmrr_list)
    writer_metric.writerow(['ANMRR', 'mAP', 'a_P@5', 'a_P@10', 'a_P@20', 'a_P@50', 'a_P@100', 'a_P@1000'])
    writer_metric.writerow([q_anmrr, m_q_ap, a_p_5, a_p_10, a_p_20, a_p_50, a_p_100, a_p_1000])

def execute_retrieval(save_path, dimension=1, pools=10, classes=38):
    if dimension == 1:
        query_index_file = r'' + save_path + 'test_index.h5'
        database_index_file = r'' + save_path + 'train_index.h5'
        metric_file_path = r'' + save_path + 'metric.csv'

        # query_index_file = r'' + save_path + 'test_index_val.h5'
        # database_index_file = r'' + save_path + 'train_index_val.h5'
        # metric_file_path = r'' + save_path + 'metric_val.csv'
    else:
        query_index_file = r'' + save_path + 'test_index' + str(dimension) + '.h5'
        database_index_file = r'' + save_path + 'train_index' + str(dimension) + '.h5'
        metric_file_path = r'' + save_path + 'metric' + str(dimension) + '.csv'

        # query_index_file = r'' + save_path + 'test_index_val_' + str(dimension) + '.h5'
        # database_index_file = r'' + save_path + 'train_index_val_' + str(dimension) + '.h5'
        # metric_file_path = r'' + save_path + 'metric_val' + str(dimension) + '.csv'


    start = time.clock()
    # 1 计算距离和排序
    matrix = retrieval_h5py_by_thread(database_index_file, query_index_file, classes, pools)
    # 2 计算精度
    cal_precision(matrix, database_index_file, query_index_file, metric_file_path)

    elapsed = (time.clock() - start)
    print("Retrieval time used:", elapsed)

if __name__ == "__main__":

    execute_retrieval(save_path="../work_dir/NWPU/resnet50_pretrain_symmexc_0_NCEandRCE/", dimension=1, pools=10, classes=45)