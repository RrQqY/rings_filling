# -*- coding:utf-8 -*-
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import *
from shapely.ops import unary_union
from queue import Queue
from collections import namedtuple
from utils import *


class fillRing(object):
    def __init__(self, dilation_radius: float = 0.1):
        super().__init__()
        self.dilation_radius = dilation_radius  # 膨胀系数
        self.title_style = {"size": 15}
        self.plot_style = {"color": "red"}
        self.legend_style = {"loc": "lower right", "prop": {"size": 10}}

    def fillSingleRing(self, file_path):
        """
        没有内部空腔的单层环情况
        :param file_path: csv文件路径
        :return: 一组内环的点序列
        """
        # 读入单个环的采样点
        rings = readMultiLayerCSV(file_path).values()
        points = list(rings)[0]
        points = np.array(points, dtype="float32")
        # 显示原始环
        plt.plot(points[:, 0], points[:, 1], "black")

        # 将采样点转换为LinearRing类
        ring = LinearRing(points)
        # 使用一个栈来模拟递归
        stk = [ring]
        result = []

        # 深度优先搜索，直到没有子环入栈，表示所有内环都已经遍历完毕
        while len(stk) > 0:
            ring = stk.pop()
            # buffer方法用于膨胀，interiors用于获取内环
            for inter_ring in ring.buffer(distance=self.dilation_radius).interiors:
                # 子内轮廓入栈
                stk.append(inter_ring)
                # 将子内轮廓转换为数组
                points = np.array(list(inter_ring.coords), dtype="float32")
                result.append(points)
                # 显示子内轮廓
                plt.plot(points[:, 0], points[:, 1], "red")

        plt.axis("off")
        plt.show()
        return result

    # 用于存储LinearRing对象和膨胀方向的元组
    Ring = namedtuple('Ring', ['ring', 'dilation_direction'])

    def fillMultipleRing(self, file_path):
        """
        有内部空腔的多层环情况
        :param file_path: csv文件路径
        :return: 一组内环的点序列
        """
        # 读入多个环的采样点
        rings = readMultiLayerCSV(file_path)
        base_ring_names = list(rings.keys())
        # 原有的环
        base_rings = []

        q = Queue()
        result = []

        # 对于读取的每个环，将其转换为LinearRing对象，存入队列q中
        for name in base_ring_names:
            points = np.array(rings[name], dtype="float32")
            ring = LinearRing(points)

            # 如果name是MainCurve则为外轮廓，否则为内轮廓
            # ring_dd = "outside" if name == "MainCurve" else "inside"
            ring_dd = "inside" if name == "MainCurve" else "outside"

            ring_info = fillRing.Ring(ring, ring_dd)
            base_rings.append(ring_info)
            q.put(ring_info)

        ring_sequences = []
        # 深度优先搜索
        while not q.empty():
            # 获取队列中的所有环
            ring_infos = [q.get() for _ in range(q.qsize())]
            all_cur_buffers = []

            # 对于每个环，计算其膨胀区域
            for ring_info in ring_infos:
                ring = ring_info.ring
                dilation_direction = ring_info.dilation_direction
                out_half_buffer = ring.buffer(self.dilation_radius) - Polygon(ring)
                # 外半膨胀区域 = 全膨胀区域 - 原始环
                if dilation_direction == "outside":
                    all_cur_buffers.append(fillRing.Ring(out_half_buffer, "outside"))
                # 内半膨胀区域 = 全膨胀区域 - 外半膨胀区域
                else:
                    inside_buffer = ring.buffer(self.dilation_radius) - out_half_buffer
                    all_cur_buffers.append(fillRing.Ring(inside_buffer, "inside"))

            buffer_after_merge = []
            union_set = UnionFind(len(ring_infos))

            # 如果两个膨胀出的环相交，则将其合并
            for i in range(len(ring_infos)):
                for j in range(len(ring_infos)):
                    if i != j and all_cur_buffers[i].ring.intersects(all_cur_buffers[j].ring):
                        union_set.union(i, j)

            # 获取膨胀后的所有环
            for buffer_index_list in union_set.groups():
                # 合并为一个集合的所有环
                buffer_list = [all_cur_buffers[index] for index in buffer_index_list]
                # 对一个集合中的所有环进行合并，得到一个单独的环
                total_buffer = unary_union([buffer.ring for buffer in buffer_list])
                # print(total_buffer)

                # 如果膨胀方向为inside，则需要将内环入栈
                if judgeDilationDirection(buffer_list) == "inside" and hasattr(total_buffer, "interiors"):
                    for inter_ring in total_buffer.interiors:
                        # 将内环转换为LinearRing对象
                        inter_ring_info = fillRing.Ring(inter_ring, "inside")
                        # 将内环入栈
                        q.put(inter_ring_info)
                        ring_sequences.append(inter_ring_info)
                # 如果膨胀方向为outside，则需要将外环入栈
                if judgeDilationDirection(buffer_list) == "outside" and hasattr(total_buffer, "exterior"):
                    # 将外环转换为LinearRing对象
                    outer_ring = total_buffer.exterior
                    outer_ring_info = fillRing.Ring(outer_ring, "outside")
                    # 将外环入栈
                    q.put(outer_ring_info)
                    ring_sequences.append(outer_ring_info)

        # 原有的环
        base_ring_region = [Polygon(rings[name]) for name in base_ring_names]
        # 筛选ring，根据包围其的曲线数量，包围它的曲线必须是奇数
        for ring_info in ring_sequences:
            # 包围其的原有环数量
            include_num = sum([1 if region.covers(ring_info.ring) else 0 for region in base_ring_region])
            # 判断曲线和原有环是否有接触
            is_intersecting = any([base_ring.ring.intersects(ring_info.ring) for base_ring in base_rings])

            # 去除所有包围其的曲线数量为偶数的曲线和与原有环有接触的曲线
            if include_num % 2 == 1 and (not is_intersecting):
                points = np.array(list(ring_info.ring.coords), dtype="float32")
                result.append(points)
                plt.plot(points[:, 0], points[:, 1], **self.plot_style)

        # 显示原始环
        for name in base_ring_names:
            points = np.array(rings[name], dtype="float32")
            # plt.subplot(121), plt.plot(points[:, 0], points[:, 1], "black")
            plt.plot(points[:, 0], points[:, 1], "black")

        # plt.subplot(121), plt.title("Before Filling", **self.title_style), plt.axis("off")
        plt.title("After Filling"), plt.axis("off")
        plt.show()
        return result


if __name__ == '__main__':
    img_path = 'test1.png'
    csv_path = 'test1.csv'
    processor = ImageProcessor()
    processor.process_image(img_path, csv_path)

    # 没有内部空腔的单层环情况
    # fill = fillRing(dilation_radius=8)
    # fill.fillSingleRing(file_path="test.csv")

    # 有内部空腔的多层环情况
    fill = fillRing(dilation_radius=16)
    result = fill.fillMultipleRing(file_path=csv_path)

    # 点采样
    result = pointSample(gap=16, result=result)
    pointPlot(result=result, title="After Sampling")
    # 显示result中y轴坐标最大值
    max_y = max(max(points[:, 1]) for points in result)
    print(max_y)

    # 点映射
    result = pointProject(center_new=(500, 400), r=0.4, k=-1.2, result=result)
    pointPlot(result=result, title="After Projection")
