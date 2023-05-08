# 我的代码
import numpy as np
from matplotlib import pyplot as plt

# 规范化矩形描述方式
# 传入四点或两点坐标返回，一点的坐标加宽高
def get_axis_aligned_bbox(region):
    region = np.asarray(region)
    nv = len(region)
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
        return (cx - w / 2, cy - h / 2, w, h)
    else:
        return (region[0], region[1], region[2] - region[0], region[3] - region[1])

# print(get_axis_aligned_bbox([28.788,57.572,97.714,57.116,98.27,141.12,29.344,141.58]))

# 传入两个矩形的左上角和右下角的坐标，得出相交面积，与面积
#（xmin,ymin,xmax,ymax)
def computeArea(rect1, rect2):
    #  让rect 1 靠左
    if rect1[0] > rect2[0]:
        return computeArea(rect2, rect1)
    # 没有重叠
    if rect1[1] >= rect2[3] or rect1[3] <= rect2[1] or rect1[2] <= rect2[0]:
        return 0 #，rect1[2] * rect1[3] + rect2[2] * rect2[3]
    #用最小的xmax减去最大的xmin，最小ymax减去最大的ymin得到交区
    #并区由总面积减去交区
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    intersection = abs(x1 - x2) * abs(y1 - y2) #abs是绝对值

    rect1w = rect1[2] - rect1[0]
    rect1h = rect1[3] - rect1[1]
    rect2w = rect2[2] - rect2[0]
    rect2h = rect2[3] - rect2[1]
    sunarea = rect1w * rect1h + rect2w * rect2h - intersection
    iou = intersection / sunarea *1.0
    return iou

#print(computeArea([-3,0,3,4], [0,-1,9,2]))


def getCenter(region):
    #输入(left,top,w,h)，返回中心坐标
    return (region[0] + region[2] / 2, region[1] + region[3] / 2)


def computePrecision(myData, trueData, x):
    # 获取中心差
    cen_gap = []
    for i in range(len(myData)):
        # x1 = myData[i][0]
        # y1 = myData[i][1]
        # x2 = trueData[i][0]
        # y2 = trueData[i][1]
        (x1,y1) = getCenter(myData[i])
        (x2, y2) = getCenter(trueData[i])
        cen_gap.append(np.sqrt((x2-x1)**2+(y2-y1)**2))
    # 计算百分比
    precision = []
    for i in range(len(x)):
        gap = x[i] #阈值
        count = 0
        for j in range(len(cen_gap)):
            if cen_gap[j] < gap:
                count += 1
        precision.append(count/len(cen_gap)) #占视频总帧的比重

    return precision

#(left,top,w,h)
def computeSuccess(myData, trueData, x):
    frames = len(trueData)
    # 获取重合率得分
    overlapScore = []
    for i in range(frames):
        one = [myData[i][0], myData[i][1], myData[i][0] +
               myData[i][2], myData[i][1] + myData[i][3]]
        two = [trueData[i][0], trueData[i][1], trueData[i][0] +
               trueData[i][2], trueData[i][1] + trueData[i][3]]
        iou = computeArea(one, two)
        overlapScore.append(iou)

    # 计算百分比
    precision = []
    for i in range(len(x)):
        gap = x[i] #阈值
        count = 0
        for j in range(frames):
            if overlapScore[j] > gap:
                count += 1
        precision.append(count/frames)

    return precision


def showPrecision(myData, trueData, algorithm, colors):
    # 生成阈值，在[start, stop]范围内计算，返回num个(默认为50)均匀间隔的样本
    xPrecision = np.linspace(0, 160, 81, endpoint=True) #等差数列，6个值
    yPrecision = []
    for i in myData: #多个算法
        # 分别存放所有点的横坐标和纵坐标，一一对应
        yPrecision.append(computePrecision(i, trueData, xPrecision)) #返回该算法下所有阈值下的比重

    # 创建图并命名
    plt.figure('Precision plot in different algorithms')
    ax = plt.gca() #获取坐标轴对象
    # 设置x轴、y轴名称
    ax.set_xlabel('Location error threshold')
    ax.set_ylabel('Precision')

    for i in range(len(myData)):
        # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
        # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
        ax.plot(xPrecision, yPrecision[i], color=colors[i], linewidth=1,
                alpha=0.6, label=algorithm[i] + "[%.3f]" % yPrecision[i][-1])

    # 设置图例的最好位置
    plt.legend(loc="best")
    plt.show()


def showSuccess(myData, trueData, algorithm, colors):
    # 生成阈值，在[start, stop]范围内计算，返回num个(默认为50)均匀间隔的样本
    xSuccess = np.linspace(0, 1, 51, endpoint=True)
    ySuccess = []
    for i in myData:
        # 分别存放所有点的横坐标和纵坐标，一一对应
        ySuccess.append(computeSuccess(i, trueData, xSuccess))
    # 创建图并命名
    plt.figure('Success plot in different algorithms')
    ax = plt.gca()
    # 设置x轴、y轴名称
    ax.set_xlabel('Overlap threshold')
    ax.set_ylabel('Success')

    for i in range(len(myData)):
        # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
        # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
        ax.plot(xSuccess, ySuccess[i], color=colors[i], linewidth=1,
                alpha=0.6, label=algorithm[i] + "[%.3f]" % ySuccess[i][0])

    # 设置图例的最好位置
    plt.legend(loc="best")
    plt.show()

# 从文件读入坐标 [left,top,w,h]
def readData(path, separator, need):
    reader = open(path, "r", encoding='utf-8')
    ans = []
    lines = reader.readlines()
    for i in range(len(lines)):
        t = lines[i].split(separator)[-4:]
        t = [float(i) for i in t]
        if need:
            ans.append(get_axis_aligned_bbox(t))
        else:
            ans.append(t)
    return ans

if __name__ == "__main__":
    girlData1 = readData('/home/bolin/zhao/proj/icvs2017_dataset/hallway_1/results/CNN_V1_result.txt', "	", False)
    girlData2 = readData('/home/bolin/zhao/proj/icvs2017_dataset/hallway_1/results/CNN_V2_result.txt', "	",
                        False)
    girlData3 = readData('/home/bolin/zhao/proj/icvs2017_dataset/hallway_1/results/CNN_V3_result.txt', "	",
                        False)
    girlData4 = readData('/home/bolin/zhao/proj/icvs2017_dataset/hallway_1/results/video_hallway_1.txt', "	",
                         False)
    girlDataTrue = readData(
        '/home/bolin/zhao/proj/icvs2017_dataset/hallway_1/GroundTruth.txt', "	", False)

    showPrecision([girlData1,girlData2,girlData3,girlData4], girlDataTrue, ["CNN_V1","CNN_V2","CNN_V3",'hallway_1'], ["c","m","y",'g'])
    showSuccess([girlData1,girlData2,girlData3,girlData4], girlDataTrue, ["CNN_V1","CNN_V2","CNN_V3",'hallway_1'],["c","m","y",'g'])



