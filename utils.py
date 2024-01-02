import cv2
import numpy as np
import math
import csv
from matplotlib import pyplot as plt
from collections import defaultdict


def readMultiLayerCSV(file_name):
    """
    将存有轮廓点的csv转化为一组描述曲线的点序列字典，
    曲线名与采样点序列为key:value。
    :param file_name: csv文件名
    :return: 一组描述曲线的点序列字典
    """
    layers = {}
    layer = None
    for index, line in enumerate(open(file_name, "r")):
        line = line.strip()
        # 曲线名
        if ("MainCurve" in line) or ("Curve" in line):
            # print(line)
            layer = line
            layers[layer] = []
        # 采样点
        else:
            point = line.split(",")
            layers[layer].append([float(point[0]), float(point[1])-200])
    return layers


class UnionFind(object):
    """
    构造函数。初始化一个并查集，其中的元素为0到n-1。
    father列表储存每个元素的父节点；初始化时，每个元素父节点都是其自身。
    """

    def __init__(self, n):
        self.father = [i for i in range(n)]

    def find(self, x):
        """
        查找元素x的根节点
        :return: 元素x的根节点
        """
        # 如果一个元素的根节点就是其自身，那么这个元素就是根节点
        if x == self.father[x]:
            return x
        else:
            # 递归地查找x的根节点
            self.father[x] = self.find(self.father[x])
            return self.father[x]

    def union(self, x, y):
        """
        合并元素x和元素y所在的集合，将元素x的根节点的根节点设为元素y的根节点
        """
        self.father[self.find(x)] = self.find(y)

    def groups(self):
        """
        返回所有集合。
        通过遍历每个元素，将每个元素添加到其根节点所对应的列表中，完成分组。
        :return: 一个列表，其中每个元素都是一个列表，包含其代表集合中所有元素。
        """
        groups = defaultdict(list)
        for i, _ in enumerate(self.father):
            groups[self.find(i)].append(i)
        return list(groups.values())


def judgeDilationDirection(buffers):
    """
    判断膨胀方向。
    如果有一个膨胀方向为inside，则整体为inside，否则为outside。
    :param buffers: 一组膨胀后的LinearRing对象
    :return: 膨胀方向
    """
    # 如果有一个膨胀方向为inside，则整体为inside，否则为outside
    # for buffer_info in buffers:
    #     print(buffer_info.dilation_direction)
    # print("--------------")
    for buffer_info in buffers:
        if buffer_info.dilation_direction == "inside":
            return "inside"
    return "outside"


class ImageProcessor:
    def __init__(self):
        self.image_path = "test1.png"
        self.output_path = "test1.csv"

    @staticmethod
    def contour_similarity(c1, c2):
        return cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I3, 0)

    def process_image(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path

        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(131), plt.imshow(image, cmap='gray'), plt.axis("off"), plt.title("Original Image")

        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        edges = cv2.Canny(binary, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        selected_contours = [contours[0]]
        for idx, contour in enumerate(contours[1:]):
            # 计算当前轮廓与已选轮廓的相似性
            similarity_scores = [self.contour_similarity(contour, selected) for selected in selected_contours]
            # 如果相似性较低，说明是一个新的轮廓，将其加入已选轮廓列表
            if min(similarity_scores) > 0.2:
                selected_contours.append(contour)

        with open(self.output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for idx, contour in enumerate(selected_contours):
                print(f"Processing contour {idx + 1}")
                contour_name = f"Curve{idx + 1}" if idx > 0 else "MainCurve"
                writer.writerow([contour_name])
                for point in contour:
                    x, y = point[0]
                    # y轴取图像高度减去y轴坐标，使得坐标原点在左上角
                    writer.writerow([x, image.shape[1] - y])

        # 创建和原始图像一样大小的空图像
        contours = np.ones_like(image)
        cv2.drawContours(contours, selected_contours, -1, (0, 0, 255), 2)

        # Show the image with all contours
        plt.subplot(132), plt.imshow(edges, cmap='gray'), plt.axis("off"), plt.title("Canny Edge Detection")
        plt.subplot(133), plt.imshow(contours, cmap='gray'), plt.axis("off"), plt.title("Contours")
        plt.show()
        # print(image.shape)


# def pointPlot(result, title):
#     for points in result:
#         # 在800×600的空图像上显示散点
#         plt.scatter(points[:, 0], points[:, 1], s=1, color="black")
#     plt.axis("off"), plt.title(title)
#     plt.show()


def pointPlot(result, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    for points in result:
        x = points[:, 0]
        y = points[:, 1]
        ax.scatter(x, y, s=1, color='black')  # Scatter plot

    plt.xlim(0, 800)  # x-axis limits
    plt.ylim(0, 600)  # y-axis limits
    plt.axis('off'), plt.title(title)
    plt.show()


# def pointProject(target_size, current_size, result):
#     """
#     将点的坐标映射到新的坐标系中
#     :param target_size: 目标坐标系的大小
#     :param current_size: 当前坐标系的大小
#     :param result: 需要映射的点的坐标
#     :return: 映射后的点的坐标
#     """
#     # 计算x和y轴的缩放比率
#     x_ratio = target_size[0] / current_size[0]
#     y_ratio = target_size[1] / current_size[1]
#
#     mapped_result = []
#     mapped_ring = []
#
#     # 遍历result中的所有点，进行映射，并将映射后的结果添加到新的结果列表中
#     for ring in result:
#         for point in ring:
#             mapped_ring.append([point[0] * x_ratio, point[1] * y_ratio])
#             mapped_result.append(mapped_ring)
#             # print(mapped_ring)
#
#     print("Projection finished")
#     return mapped_result


def pointProject(center_new, r, k, result):
    mapped_result = []

    # 遍历result中的所有点，进行映射，并将映射后的结果添加到新的结果列表中
    for ring in result:
        points = []

        for point in ring:
            point_old = point - [400, 300]
            x_new, y_new = center_new

            # 使用给定的新中心、缩小比例和斜率创建一个新的基于旋转的仿射矩阵
            theta = np.arctan(k)  # 从斜率计算角度
            scale = np.array([[r, 0],
                              [0, r]])  # 缩放矩阵
            rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])  # 旋转矩阵
            translation = np.array([x_new, y_new])  # 平移矩阵

            # 在旧点上应用缩放、旋转和平移变换
            point_new = scale @ np.array(point_old)  # 应用缩放
            point_new = rotation @ point_new  # 应用旋转
            point_new += translation  # 应用平移
            points.append(tuple(point_new))

        mapped_ring = np.array(list(points), dtype="float32")
        mapped_result.append(mapped_ring)

    print("Projection finished")
    return mapped_result


def getRectangle(contour):
    # 使用minAreaRect获取最小面积的矩形(该矩形可能会旋转)
    rect = cv2.minAreaRect(contour)

    # 获取矩形的中心点坐标、宽度、高度和旋转角度
    center, (width, height), angle = rect

    # 根据最小矩形的宽度和高度确定长轴和斜率
    if width < height:
        long_axis_slope = np.tan(np.deg2rad(90 + angle))
    else:
        long_axis_slope = np.tan(np.deg2rad(angle))

    # 计算高度与宽度之比，并依据比例判断返回值
    ratio = height / width
    if ratio > 3 / 4:
        is_width = True
        dimension = width
    else:
        is_width = False
        dimension = height

    return center, long_axis_slope, is_width, dimension


def pointSample(gap, result):
    """
    对点进行等间距采样
    :param gap: 采样间隔
    :param result: 需要采样的点
    :return: 采样后的点
    """
    sampled_result = []

    for ring in result:
        ring_len = 0
        # 计算ring的总长度
        for i in range(1, len(ring)):
            point1, point2 = ring[i - 1], ring[i]
            ring_len += math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

        # 计算保留点的数量，留意可能的除0错误
        num_points = int(ring_len // gap) if ring_len != 0 else len(ring)

        # 求得每隔interval个点取一个点，可能的除以0错误也需要注意
        interval = len(ring) // num_points if num_points != 0 else len(ring)
        # print(len(ring), num_points, interval)
        # 对ring进行采样
        if len(ring) > interval and interval > 0:
            sampled_ring = ring[::interval]
            sampled_result.append(sampled_ring)

    print("Sampling finished")
    return sampled_result


if __name__ == '__main__':
    # 使用示例
    processor = ImageProcessor()
    processor.process_image('test1.png', 'test1.csv')
