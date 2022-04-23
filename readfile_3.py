import math
import random
import numpy as np
import waveforms_processing

time_shift = [-500, -300,  0, 300, 500]
wave_length = 7000

def load_train_data(ep_record, ss_record, num):
    X_train = []
    Y_train = []
    usesize = [1, wave_length]
    lsep = np.zeros([1, wave_length, 3])
    # 爆破 1,0
    index_ep_list = list(range(0, int(len(ep_record)/3)))
    random.shuffle(index_ep_list)
    for i in range(num):
        for j in range(3):
            index = index_ep_list[i] * 3 + j
            onset = int(ep_record[index].split()[0])
            ear = ep_record[index].replace('\n', '').split()[1]
            if '河南' in ear:
                random.shuffle(time_shift)
                onset = onset + time_shift[0]
            fr = open(ear)
            fread = fr.read()
            fr.close()
            ori_data = [float(k) for k in fread.split()[:]]
            data = waveforms_processing.qushi10_guiyihua(ori_data)[onset:onset + wave_length]
            lsep[:, :, j] = np.array(data).reshape(usesize)
        X_train.append(lsep.copy())
        Y_train.append([1, 0])

    # 塌陷 0，1
    index_ss_list = list(range(0, int(len(ss_record)/3)))
    random.shuffle(index_ss_list)
    for k in range(num):
        for l in range(3):
            index = index_ss_list[k] * 3 + l
            onset = int(ss_record[index].split()[0])
            ear = ss_record[index].replace('\n', '').split()[1]
            if '河南' in ear:
                random.shuffle(time_shift)
                onset = onset + time_shift[0]
            fr = open(ear)
            fread = fr.read()
            fr.close()
            ori_data = [float(k) for k in fread.split()[:]]
            data = waveforms_processing.qushi10_guiyihua(ori_data)[onset:onset + wave_length]
            lsep[:, :, l] = np.array(data).reshape(usesize)
        X_train.append(lsep.copy())
        Y_train.append([0, 1])
    return X_train, Y_train


def load_test_data(ep_record, ss_record, num):
    X_test = []
    Y_test = []
    usesize = [1, wave_length]
    lsep = np.zeros([1, wave_length, 3])
    # 爆破 1,0
    index_ep_list = list(range(0, int(len(ep_record)/3)))
    random.shuffle(index_ep_list)
    for i in range(num):
        for j in range(3):
            index = index_ep_list[i] * 3 + j
            onset = int(ep_record[index].split()[0])
            ear = ep_record[index].replace('\n', '').split()[1]
            if '河南' in ear:
                random.shuffle(time_shift)
                onset = onset + time_shift[0]
            fr = open(ear)
            fread = fr.read()
            fr.close()
            ori_data = [float(k) for k in fread.split()[:]]
            data = waveforms_processing.qushi10_guiyihua(ori_data)[onset:onset + wave_length]
            lsep[:, :, j] = np.array(data).reshape(usesize)
        X_test.append(lsep.copy())
        Y_test.append([1, 0])

    # 塌陷测试
    index_ss_list = list(range(0, int(len(ss_record)/3)))
    random.shuffle(index_ss_list)
    for k in range(num):
        for l in range(3):
            index = index_ss_list[k] * 3 + l
            onset = int(ss_record[index].split()[0])
            ear = ss_record[index].replace('\n', '').split()[1]
            if '河南' in ear:
                random.shuffle(time_shift)
                onset = onset + time_shift[0]
            fr = open(ear)
            fread = fr.read()
            fr.close()
            ori_data = [float(k) for k in fread.split()[:]]
            data = waveforms_processing.qushi10_guiyihua(ori_data)[onset:onset + wave_length]
            lsep[:, :, l] = np.array(data).reshape(usesize)
        X_test.append(lsep.copy())
        Y_test.append([0, 1])
    return X_test, Y_test