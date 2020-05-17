from random import random, randint
import math


# 根据等级和年代对价格进行模拟
def wineprice(rating, age):
    peak_age = rating - 50

    # 根据等级计算价格
    price = rating / 2
    if age > peak_age:
        # 经过“峰值年”，后续5年里其品质将会变差
        price = price * (5 - (age - peak_age) / 2)
    else:
        # 价格在接近“峰值年”时会增加到原值的5倍
        price = price * (5 * ((age + 1) / peak_age))
    if price < 0: price = 0
    return price


# 生成一批数据代表样本数据集
def wineset1():
    rows = []
    for i in range(300):
        # 随机生成年代和等级
        rating = random() * 50 + 50
        age = random() * 50

        # 得到一个参考价格
        price = wineprice(rating, age)

        # 添加一些噪音
        price *= (random() * 0.2 + 0.9)

        # 加入数据集
        rows.append({'input': (rating, age), 'result': price})
    return rows


# 使用欧几里得距离，定义距离
def euclidean(v1, v2):
    d = 0.0
    for i in range(len(v1)):
        d += (v1[i] - v2[i]) ** 2
    return math.sqrt(d)


# 计算待测商品和样本数据集中任一商品间的距离。data样本数据集，vec1待测商品
def getdistances(data, vec1):
    distancelist = []

    # 遍历样本数据集中的每一项
    for i in range(len(data)):
        vec2 = data[i]['input']

        # 添加距离到距离列表
        distancelist.append((euclidean(vec1, vec2), i))

    # 距离排序
    distancelist.sort()
    return distancelist  # 返回距离列表


# 对距离值最小的前k个结果求平均
def knnestimate(data, vec1, k=5):
    # 得到经过排序的距离值
    dlist = getdistances(data, vec1)
    avg = 0.0

    # 对前k项结果求平均
    for i in range(k):
        idx = dlist[i][1]
        avg += data[idx]['result']
    avg = avg / k
    return avg


def inverseweight(dist, num=1.0, const=0.1):
    return num / (dist + const)


def subtractweight(dist, const=1.0):
    if dist > const:
        return 0
    else:
        return const - dist


def gaussian(dist, sigma=5.0):
    return math.e ** (-dist ** 2 / (2 * sigma ** 2))


# 对距离值最小的前k个结果求加权平均
def weightedknn(data, vec1, k=5, weightf=gaussian):
    # 得到距离值
    dlist = getdistances(data, vec1)
    avg = 0.0
    totalweight = 0.0

    # 得到加权平均
    for i in range(k):
        dist = dlist[i][0]
        idx = dlist[i][1]
        weight = weightf(dist)
        avg += weight * data[idx]['result']
        totalweight += weight
    if totalweight == 0: return 0
    avg = avg / totalweight
    return avg


# 划分数据。test待测集占的比例。其他数据为训练集
def dividedata(data, test=0.05):
    trainset = []
    testset = []
    for row in data:
        if random() < test:
            testset.append(row)
        else:
            trainset.append(row)
    return trainset, testset


# 对使用算法进行预测的结果的误差进行统计，以此判断算法好坏。algf为算法函数，trainset为训练数据集，testset为待测数据集
def testalgorithm(algf, trainset, testset):
    error = 0.0
    for row in testset:
        guess = algf(trainset, row['input'])  # 这一步要和样本数据的格式保持一致，列表内个元素为一个字典，每个字典包含input和result两个属性。而且函数参数是列表和元组
        error += (row['result'] - guess) ** 2
        # print row['result'],guess
    # print error/len(testset)
    return error / len(testset)


# 将数据拆分和误差统计合并在一起。对数据集进行多次划分，并验证算法好坏
def crossvalidate(algf, data, trials=100, test=0.1):
    error = 0.0
    for i in range(trials):
        trainset, testset = dividedata(data, test)
        error += testalgorithm(algf, trainset, testset)
    return error / trials


if __name__ == '__main__':  # 只有在执行当前模块时才会运行此函数
    data = wineset1()  # 创建第一批数据集
    print(data)
    error = crossvalidate(knnestimate, data)  # 对直接求均值算法进行评估
    print('平均误差：' + str(error))


    def knn3(d, v): return knnestimate(d, v, k=3)  # 定义一个函数指针。参数为d列表，v元组


    error = crossvalidate(knn3, data)  # 对直接求均值算法进行评估
    print('平均误差：' + str(error))


    def knninverse(d, v): return weightedknn(d, v, weightf=inverseweight)  # 定义一个函数指针。参数为d列表，v元组


    error = crossvalidate(knninverse, data)  # 对使用反函数做权值分配方法进行评估
    print('平均误差：' + str(error))
