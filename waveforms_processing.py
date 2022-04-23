import numpy as np

T = 0.01

# 数值微分-加速度
def acceleration(list_v):
    a = []
    for i in range(len(list_v)):
        if i == 0:
            a.append(list_v[0])
        else:
            a.append(list_v[i] - list_v[i-1])
    return a

# 数值积分-位移
def distance(list_v):
    d = []
    for i in range(len(list_v)):
        if i == 0:
            d.append(0.0)
        else:
            d.append(d[i-1] + (list_v[i] + list_v[i-1])/2*T)
    return d

# 十次多项式拟合去势，差分，归一化
def chafen_qushi_guiyihua(ori_list):
    # 去势
    a = np.polyfit(range(len(ori_list)), ori_list[:], 10)  # 用10次多项式拟合x，y数组
    b = np.poly1d(a)  # 拟合完之后用这个函数来生成多项式对象
    c = b(range(len(ori_list)))  # 生成多项式对象之后，就是获取x在这个多项式处的值
    qushi = [(ori_list[i] - c[i]) for i in range(len(ori_list))]
    # 差分
    cf = []
    for i in range(len(ori_list)):
        if i == len(ori_list) - 1:
            cf.append(ori_list[i] - c[i])
        else:
            cf.append((ori_list[i + 1] - c[i + 1]) - (ori_list[i] - c[i]))
    # 归一化
    mean = np.mean(np.array(cf))
    std = np.std(np.array(cf), ddof=1)
    nor = []
    for i in range(len(cf)):
        nor.append((cf[i] - mean) / std)
    return nor


# 十次多项式拟合去势，归一化
def qushi10_guiyihua(ori_list):
    # 去势
    a = np.polyfit(range(len(ori_list)), ori_list[:], 10)  # 用10次多项式拟合x，y数组
    b = np.poly1d(a)  # 拟合完之后用这个函数来生成多项式对象
    c = b(range(len(ori_list)))  # 生成多项式对象之后，就是获取x在这个多项式处的值
    qushi = [(ori_list[i] - c[i]) for i in range(len(ori_list))]
    # 归一化
    mean = np.mean(np.array(qushi))
    std = np.std(np.array(qushi), ddof=1)
    nor = []
    for i in range(len(qushi)):
        nor.append((qushi[i] - mean) / std)
    return nor


# 均值去势，归一化
def qushi_mean_guiyihua(ori_list):
    # 去势
    mean = np.mean(np.array(ori_list))
    qushi = [(ori_list[i] - mean) for i in range(len(ori_list))]
    # 归一化
    mean = np.mean(np.array(qushi))
    std = np.std(np.array(qushi), ddof=1)
    nor = []
    for i in range(len(qushi)):
        nor.append((qushi[i] - mean) / std)
    return nor


# 均值去势，scalesize
def qushi_mean_scalesize(ori_list):
    # 去势
    mean = np.mean(np.array(ori_list))
    qushi = [(ori_list[i] - mean) for i in range(len(ori_list))]
    # scalesize
    scalesize = 1e-5
    scale = [(qushi[i] * scalesize) for i in range(len(ori_list))]
    return scale


def detrend_normalization_max(ori_list):
    # 去势
    a = np.polyfit(range(len(ori_list)), ori_list[:], 10)  # 用10次多项式拟合x，y数组
    b = np.poly1d(a)  # 拟合完之后用这个函数来生成多项式对象
    c = b(range(len(ori_list)))  # 生成多项式对象之后，就是获取x在这个多项式处的值
    qushi = [(ori_list[i] - c[i]) for i in range(len(ori_list))]
    # 归一化——最大值
    max_value = max([abs(qushi[i]) for i in range(len(qushi))])
    nor = [(qushi[i] / max_value) for i in range(len(qushi))]
    return nor