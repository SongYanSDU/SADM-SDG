# -- coding: utf-8 --
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import random
# import nlpaug.augmenters.audio as naa


def data_shuffle(data1, label1):
    index1 = np.arange(np.size(data1, 0))
    np.random.shuffle(index1)
    data1 = data1[index1, :]
    label1 = label1[index1, ]
    return data1, label1


def create_final_subset(data, labels):
    final_data = []
    final_labels = []
    for cls in range(0, 5):
        # 筛选当前类别的样本
        cls_mask = labels == cls
        cls_data = data[cls_mask]
        cls_labels = labels[cls_mask]
        if cls_data.shape[0] == 0:
            print("子数据集没有样本。")
            continue
        # 选择前 per_class_samples 个样本
        selected_data = cls_data[:1000]
        selected_labels = cls_labels[:1000]
        final_data.append(selected_data)
        final_labels.append(selected_labels)
        # 合并所有选取的样本
    final_data = np.vstack(final_data)
    final_labels = np.hstack(final_labels)
    return final_data, final_labels


def bjut_dataset():
    pd1 = sio.loadmat(
        'F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_20.mat')
    data1 = pd1['bjut_speed_20'][:, 1:]
    label1 = pd1['bjut_speed_20'][:, 0]
    data1, label1 = create_final_subset(data1, label1)
    data1, label1 = data_shuffle(data1, label1)

    pd2 = sio.loadmat(
        'F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_25.mat')
    data2 = pd2['bjut_speed_25'][:, 1:]
    label2 = pd2['bjut_speed_25'][:, 0]
    data2, label2 = create_final_subset(data2, label2)
    data2, label2 = data_shuffle(data2, label2)

    pd3 = sio.loadmat(
        'F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_30.mat')
    data3 = pd3['bjut_speed_30'][:, 1:]
    label3 = pd3['bjut_speed_30'][:, 0]
    data3, label3 = create_final_subset(data3, label3)
    data3, label3 = data_shuffle(data3, label3)

    pd4 = sio.loadmat(
        'F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_35.mat')
    data4 = pd4['bjut_speed_35'][:, 1:]
    label4 = pd4['bjut_speed_35'][:, 0]
    data4, label4 = create_final_subset(data4, label4)
    data4, label4 = data_shuffle(data4, label4)

    pd5 = sio.loadmat(
        'F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_40.mat')
    data5 = pd5['bjut_speed_40'][:, 1:]
    label5 = pd5['bjut_speed_40'][:, 0]
    data5, label5 = create_final_subset(data5, label5)
    data5, label5 = data_shuffle(data5, label5)

    pd6 = sio.loadmat(
        'F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_45.mat')
    data6 = pd6['bjut_speed_45'][:, 1:]
    label6 = pd6['bjut_speed_45'][:, 0]
    data6, label6 = create_final_subset(data6, label6)
    data6, label6 = data_shuffle(data6, label6)

    pd7 = sio.loadmat(
        'F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_50.mat')
    data7 = pd7['bjut_speed_50'][:, 1:]
    label7 = pd7['bjut_speed_50'][:, 0]
    data7, label7 = create_final_subset(data7, label7)
    data7, label7 = data_shuffle(data7, label7)

    pd8 = sio.loadmat(
        'F:\\数据集\\BJUT-WT-planetary-gearbox-dataset-master\\bjut_speed_55.mat')
    data8 = pd8['bjut_speed_55'][:, 1:]
    label8 = pd8['bjut_speed_55'][:, 0]
    data8, label8 = create_final_subset(data8, label8)
    data8, label8 = data_shuffle(data8, label8)
    return (data1, label1, data2, label2, data3, label3, data4, label4, data5, label5,
            data6, label6, data7, label7, data8, label8)

def Lboro_dataset():
    def load_and_filter(filepath):
        pd = sio.loadmat(filepath)
        data = pd[list(pd.keys())[-1]][:, 1:]   # 通常动态地获取变量名
        label = pd[list(pd.keys())[-1]][:, 0]
        # 删除标签为4的数据
        '''mask = label != 4
        # 将标签为5的数据改为4
        label[label == 5] = 4
        # 应用mask
        data, label = data[mask], label[mask]
        # 只保留标签为0到4的数据
        mask = (label >= 0) & (label <= 4)
        data, label = data[mask], label[mask]'''
        return data_shuffle(data, label)

    data1, label1 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_900.mat')
    data2, label2 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_1500.mat')
    data3, label3 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_2000.mat')
    data4, label4 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_1500_2000.mat')
    data5, label5 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_900_1500.mat')
    data6, label6 = load_and_filter('F:\\数据集\\拉夫堡\\lboro_900_2000.mat')
    return data1, label1, data2, label2, data3, label3, data4, label4, data5, label5, data6, label6