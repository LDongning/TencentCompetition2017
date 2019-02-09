import sys
import pandas as pd
import scipy as sp
from pyspark import SparkContext
from math import e
import math
# from pyspark.sql import Row
import numpy as np

outputfile = sys.argv[1]
inputfile = sys.argv[2]


def fea_extra_one(df):
    train = df.split(" ")  # .map(lambda field: Row(id=field[0], tracks=field[1], aim=field[2], label=field[3]))
    points = []
    for ele in train[1][:-1].split(';'):
        point = ele.split(',')
        points.append(((float(point[0]), float(point[1])), float(point[2])))
    if len(points) == 1:
        lastPoint = points[0]
        points.append(lastPoint)
    if len(points) == 2:
        lastPoint = points[1]
        points.append(lastPoint)

    xs = np.array([point[0][0] for point in points])
    ys = np.array([point[0][1] for point in points])

    aim = train[2].split(',')
    aim = np.array([float(aim[0]), float(aim[1])])

    distance_deltas = np.array([math.sqrt(
        math.pow(points[i][0][0] - points[i + 1][0][0], 2) + math.pow(points[i][0][1] - points[i + 1][0][1], 2)) for i
                                 in range(len(points) - 1)])
    time_deltas = np.array([points[i + 1][1] - points[i][1] for i in range(len(points) - 1)])

    # xs_deltas = xs.diff(1).dropna()
    # ys_deltas = ys.diff(1).dropna()
    xs_deltas = np.diff(xs)
    ys_deltas = np.diff(ys)

    speeds = np.array(
        [np.log1p(distance) - np.log1p(abs(delta)) for (distance, delta) in zip(distance_deltas, time_deltas)])

    angles = np.array(
        [np.log1p(abs(points[i + 1][0][1] - points[i][0][1])) - np.log1p(abs(points[i + 1][0][0] - points[i][0][0])) for i
         in range(len(points) - 1)])

    # speed_diff = speeds.diff(1).dropna()
    # angle_diff = angles.diff(1).dropna()
    speed_diff = np.diff(speeds)
    angle_diff = np.diff(angles)

    distance_aim_deltas = np.array(
        [math.sqrt(math.pow(points[i][0][0] - aim[0], 2) + math.pow(points[i][0][1] - aim[1], 2)) for i in
         range(len(points))])

    # distance_aim_deltas_diff = distance_aim_deltas.diff(1).dropna()
    distance_aim_deltas_diff = np.diff(distance_aim_deltas)

    x_last6 = xs[len(xs)-6:]
    x_last6_aimX_value = np.array([ele - aim[0] for ele in x_last6])

    fea_list = []
    fea_list.append(pd.Series(speed_diff).median())
    fea_list.append(speed_diff.mean())
    fea_list.append(speed_diff.var())
    fea_list.append(speed_diff.max())
    fea_list.append(angle_diff.var())

    fea_list.append(time_deltas.min())
    fea_list.append(time_deltas.max())
    fea_list.append(time_deltas.var())
    fea_list.append(distance_deltas.max())

    fea_list.append(distance_aim_deltas[-1])
    fea_list.append(distance_aim_deltas_diff.max())
    fea_list.append(distance_aim_deltas_diff.var())

    fea_list.append(speeds.mean())
    fea_list.append(speeds.max())
    fea_list.append(speeds.var())

    fea_list.append(angles.max())
    fea_list.append(angles.var())
    fea_list.append(pd.Series(angles).kurt())

    fea_list.append(ys.min())
    fea_list.append(ys.max())
    fea_list.append(ys.var())

    fea_list.append(xs.min())
    fea_list.append(xs.max())
    fea_list.append(xs.var())

    xs_deltas_overZero = [(ele > 0.00001) and ele for ele in xs_deltas]
    xs_deltas_lowZero = [(ele < 0.00001) and ele for ele in xs_deltas]
    fea_list.append(min(len(xs_deltas_lowZero), len(xs_deltas_overZero)))

    ys_deltas_overZero = [(ele > 0.00001) and ele for ele in ys_deltas]
    ys_deltas_lowZero = [(ele < 0.00001) and ele for ele in ys_deltas]
    fea_list.append(min(len(ys_deltas_lowZero), len(ys_deltas_overZero)))

    fea_list.append(xs_deltas.var())
    fea_list.append(xs_deltas.max())
    fea_list.append(xs_deltas.min())

    # fea_list.append(xs[0:6].mean())
    # fea_list.append(xs[0:6].var())
    #
    fea_list.append(x_last6_aimX_value.var())
    fea_list.append(x_last6_aimX_value.min())
    #
    # xs_stop = [(abs(ele) <= 0.00001) and ele for ele in xs_deltas]
    # ys_stop = [(abs(ele) <= 0.00001) and ele for ele in ys_deltas]
    # fea_list.append(len(xs_stop))
    # fea_list.append(len(ys_stop))
    #
    aim_angle = np.array([np.log1p( abs(point[0][1] - aim[1]) ) - np.log1p( abs(point[0][0] - aim[0]) ) for point in points])
    aim_angle_diff = np.diff(aim_angle)
    #
    # aim_angle_speed = np.array([np.log1p(abs(aim_angle_diff_v)) - np.log1p(abs(time_deltas[index])) for index, aim_angle_diff_v in enumerate(aim_angle_diff)])
    # aim_angle_speed_diff = np.diff(aim_angle_speed)
    #
    fea_list.append(aim_angle[-1])
    fea_list.append(aim_angle_diff.max())
    fea_list.append(aim_angle_diff.var())
    #
    # fea_list.append(aim_angle_speed_diff.var())
    # fea_list.append(aim_angle_speed_diff.mean())
    #
    # fea_list.append(aim_angle_speed.var())
    # fea_list.append(aim_angle_speed.mean())

    fea_str_list = [' {}:{:.8f}'.format(i + 1, j) for i, j in enumerate(fea_list)]
    fea_str_list_new = [ele.replace('nan', '-1.00000000') for ele in fea_str_list]

    fea_str_list_new = [str(train[0])] + fea_str_list_new
    fea_str = ' '.join(fea_str_list_new)

    return fea_str


if __name__ == "__main__":
    sc = SparkContext(appName="test")
    # rdd = sc.textFile('dsjtzs_txfz_training.txt')
    rdd = sc.textFile(inputfile)
    result = rdd.map(lambda line: fea_extra_one(line))
    result.saveAsTextFile(outputfile)
    # result.saveAsTextFile('sv')
