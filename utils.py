import cv2
import numpy as np
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
            layers[layer].append([float(point[0]), float(point[1])])
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
        self.image_path = "test.png"
        self.output_path = "test.csv"

    @staticmethod
    def contour_similarity(c1, c2):
        return cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I2, 0)

    def process_image(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path

        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(131), plt.imshow(image, cmap='gray')

        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        edges = cv2.Canny(binary, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        selected_contours = [contours[0]]
        for idx, contour in enumerate(contours[1:]):
            # 计算当前轮廓与已选轮廓的相似性
            similarity_scores = [self.contour_similarity(contour, selected) for selected in selected_contours]
            # 如果相似性较低，说明是一个新的轮廓，将其加入已选轮廓列表
            if min(similarity_scores) > 0.5:
                selected_contours.append(contour)

        with open(self.output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for idx, contour in enumerate(selected_contours):
                print(f"Processing contour {idx + 1}")
                contour_name = f"Curve{idx + 1}" if idx > 0 else "MainCurve"
                writer.writerow([contour_name])
                for point in contour:
                    x, y = point[0]
                    writer.writerow([x, y])

        # 创建和原始图像一样大小的空图像
        white = np.ones_like(image)
        cv2.drawContours(white, selected_contours, -1, (0, 0, 255), 2)

        # Show the image with all contours
        plt.subplot(132), plt.imshow(edges, cmap='gray')
        plt.subplot(133), plt.imshow(white, cmap='gray')
        plt.show()


if __name__ == '__main__':
    # 使用示例
    processor = ImageProcessor()
    processor.process_image('test1.png', 'test1.csv')
