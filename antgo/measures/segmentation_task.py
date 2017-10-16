# encoding=utf-8
# @Time    : 17-7-12
# @File    : segmentation_task.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
from antgo.task.task import *
from antgo.measures.base import *
from antgo.dataflow.common import *
from antgo.utils.utils import get_sort_index
import cv2
import time


class AntPixelAccuracySeg1(AntMeasure):
    def __init__(self, task):
        # paper: Jonathan Long, Evan Shelhamer, etc. Fully Convolutional Networks for Semantic Segmentation
        # formular: \sum_i{n_{ii}}/\sum_i t_i
        super(AntPixelAccuracySeg1, self).__init__(task, 'PixelAccuracy')
        assert(task.task_type == 'SEGMENTATION')

        self.is_support_rank = True

    def eva(self, data, label):
        classes_num = len(self.task.class_label)

        sum_nii = np.zeros((1, classes_num))
        sum_ti = np.zeros((1, classes_num))

        if label is not None:
            data = zip(data, label)
        val_list = []
        for predict, gt in data:
            single_nii = np.zeros((1, classes_num))
            single_ti = np.zeros((1, classes_num))
            gt_labels = set(gt.flatten())
            for l in gt_labels:
                l = int(l)
                if l == 0:
                    continue

                p = np.where(gt == l)
                single_ti[l - 1] += len(p[0])
                sum_ti[l - 1] += len(p[0])

                predicted_l = predict[p]
                nii = len(np.where(predicted_l == l)[0])
                single_nii[l - 1] += nii
                sum_nii[l - 1] += nii
            single_val = np.sum(single_nii) / np.sum(single_ti)
            val_list.append(single_val)
        val = np.sum(sum_nii) / np.sum(sum_ti)
        # val_index = get_sort_index(val_list)[0:10]
        return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type':'SCALAR'},
                                                           {'name': 'AntPixelAccuracySeg list', 'value': val_list, 'type': 'TABLE'}]}}
        # return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type':'SCALAR'},
        #                                                    {'name': 'val_list', 'value': val_index, 'type': 'TABLE'}]}}


class AntMeanAccuracySeg(AntMeasure):
    def __init__(self, task):
        # paper: Jonathan Long, Evan Shelhamer, etc. Fully Convolutional Networks for Semantic Segmentation
        # formular: (1/n_{cl}) / \sum_i n_{ii}/t_i
        super(AntMeanAccuracySeg, self).__init__(task, 'MeanAccuracy')
        assert(task.task_type == 'SEGMENTATION')

        self.is_support_rank = True

    def eva(self, data, label):
        classes_num = len(self.task.class_label)

        sum_nii = np.zeros((1, classes_num))
        sum_ti = np.zeros((1, classes_num))

        if label is not None:
            data = zip(data, label)
        val_list = []
        for predict, gt in data:
            single_nii = np.zeros((1, classes_num))
            single_ti = np.zeros((1, classes_num))
            gt_labels = set(gt.flatten())
            for l in gt_labels:
                l = int(l)
                if l == 0:
                    continue

                p = np.where(gt == l)
                single_ti[l - 1] += len(p[0])
                sum_ti[l - 1] += len(p[0])

                predicted_l = predict[p]
                nii = len(np.where(predicted_l == l)[0])
                single_nii[l - 1] += nii
                sum_nii[l - 1] += nii
            single_val = np.sum(single_nii) / np.sum(single_ti)
            val_list.append(single_val)
        val = np.mean(sum_nii / sum_ti)
        # val_index = get_sort_index(val_list)[0:10]
        return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type':'SCALAR'},
                                                           {'name': 'AntMeanAccuracySeg list', 'value': val_list, 'type': 'TABLE'}]}}


class AntMeanIOUSeg(AntMeasure):
    def __init__(self, task):
        # paper: Jonathan Long, Evan Shelhamer, etc. Fully Convolutional Networks for Semantic Segmentation
        # formular: (1/n_{cl}) / \sum_i n_{ii}/(t_i+\sum_j n_{ji}-n_{ii})

        super(AntMeanIOUSeg, self).__init__(task, 'MeanIOU')
        assert(task.task_type == 'SEGMENTATION')

        self.is_support_rank = True

    def eva(self, data, label):
        classes_num = len(self.task.class_label)

        sum_nii = np.zeros((1, classes_num))
        sum_ti = np.zeros((1, classes_num))
        sum_ji = np.zeros((1, classes_num))

        if label is not None:
            data = zip(data, label)
        val_list = []
        for predict, gt in data:
            gt_labels = set(gt.flatten())
            single_nii = np.zeros((1, classes_num))
            single_ti = np.zeros((1, classes_num))
            single_ji = np.zeros((1, classes_num))
            for l in gt_labels:
                l = int(l)
                if l == 0:
                    continue
                p = np.where(gt == l)
                single_ti[l - 1] += len(p[0])
                sum_ti[l - 1] += len(p[0])

                predicted_l = predict[p]
                nii = len(np.where(predicted_l == l)[0])
                single_nii[l - 1] += nii
                sum_nii[l - 1] += nii

                single_ji[l - 1] += len(np.where(predict == l)[0])
                sum_ji[l - 1] += len(np.where(predict == l)[0])
            single_val = np.mean(single_nii / (single_ti + single_ji - single_nii))
            val_list.append(single_val)

        val = np.mean(sum_nii / (sum_ti + sum_ji - sum_nii))
        # val_index = get_sort_index(val_list)[0:10]
        return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type':'SCALAR'},
                                                           {'name': 'AntMeanIOUSeg list', 'value': val_list, 'type': 'TABLE'}]}}


class AntFrequencyWeightedIOUSeg(AntMeasure):
    def __init__(self, task):
        # paper: Jonathan Long, Evan Shelhamer, etc. Fully Convolutional Networks for Semantic Segmentation
        # formular: (\sum_kt_k)^{-1} / \sum_i t_in_{ii}/(t_i+\sum_j n_{ji}-n_{ii})

        super(AntFrequencyWeightedIOUSeg, self).__init__(task, 'FrequencyWeightedIOU')
        assert(task.task_type == 'SEGMENTATION')

        self.is_support_rank = True

    def eva(self, data, label):
        classes_num = len(self.task.class_label)

        sum_nii = np.zeros((1, classes_num))
        sum_ti = np.zeros((1, classes_num))
        sum_ji = np.zeros((1, classes_num))

        if label is not None:
            data = zip(data, label)
        val_list = []
        for predict, gt in data:
            gt_labels = set(gt.flatten())
            single_nii = np.zeros((1, classes_num))
            single_ti = np.zeros((1, classes_num))
            single_ji = np.zeros((1, classes_num))
            for l in gt_labels:
                l = int(l)
                if l == 0:
                    continue
                p = np.where(gt == l)
                single_ti[l - 1] += len(p[0])
                sum_ti[l - 1] += len(p[0])

                predicted_l = predict[p]
                nii = len(np.where(predicted_l == l)[0])
                single_nii[l - 1] += nii
                sum_nii[l - 1] += nii

                single_ji[l - 1] += len(np.where(predict == l)[0])
                sum_ji[l - 1] += len(np.where(predict == l)[0])
            single_val = np.sum(single_ti * single_nii / (single_ti + single_ji - single_nii)) / np.sum(single_ti)
            val_list.append(single_val)

        val = np.sum(sum_ti * sum_nii / (sum_ti + sum_ji - sum_nii)) / np.sum(sum_ti)
        # val_index = get_sort_index(val_list)[0:10]
        return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type':'SCALAR'},
                                                           {'name': 'AntFrequencyWeightedIOUSeg list', 'value': val_list, 'type': 'TABLE'}]}}


class AntMeanIOUBoundary(AntMeasure):
    def __init__(self, task):
        # paper: Liang-Chieh Chen, etc. Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs
        # formular: (1/n_{cl}) / \sum_i n_{ii}/(t_i+\sum_j n_{ji}-n_{ii})

        super(AntMeanIOUBoundary, self).__init__(task, 'MeanIOUBoundary')
        assert(task.task_type == 'SEGMENTATION')

        self.is_support_rank = True

    def eva(self, data, label):
        classes_num = len(self.task.class_label)

        sum_nii = np.zeros((1, classes_num))
        sum_ti = np.zeros((1, classes_num))
        sum_ji = np.zeros((1, classes_num))

        trimap_width = int(getattr(self.task, 'trimap_width', 3))

        offset_x, offset_y = np.meshgrid(np.arange(-trimap_width, trimap_width + 1),
                                         np.arange(-trimap_width, trimap_width + 1))
        offset = np.column_stack((offset_x.flatten(), offset_y.flatten()))

        if label is not None:
            data = zip(data, label)
        val_list = []
        for predict, gt in data:
            single_nii = np.zeros((1, classes_num))
            single_ti = np.zeros((1, classes_num))
            single_ji = np.zeros((1, classes_num))

            gt_labels = set(gt.flatten())
            rows, cols = gt.shape[:2]
            # generate trimap for objects (predict)
            predict_boundary = np.where((predict[0:-1, 0:-1] - predict[1:, 1:]) != 0)
            predict_boundary = np.column_stack(predict_boundary)
            predict_band_boundary = np.expand_dims(predict_boundary, 1) + offset
            predict_band_boundary = predict_band_boundary.reshape(-1, 2)
            index = np.where((predict_band_boundary[:, 0] > 0) &
                             (predict_band_boundary[:, 1] > 0) &
                             (predict_band_boundary[:, 0] < rows) &
                             (predict_band_boundary[:, 1] < cols))
            predict_trimap = np.zeros((rows, cols), dtype=np.int32)
            predict_trimap[predict_band_boundary[index, 0], predict_band_boundary[index, 1]] = \
                predict[predict_band_boundary[index, 0], predict_band_boundary[index, 1]]

            for l in gt_labels:
                l = int(l)
                if l == 0:
                    continue
                # generate trimap for object (gt)
                obj_map = np.zeros((rows, cols), dtype=np.uint32)
                obj_map[np.where(gt == l)] = 1

                gt_boundary = np.where((obj_map[0:-1, 0:-1] - obj_map[1:, 1:]) != 0)
                gt_boundary = np.column_stack(gt_boundary)
                gt_band_boundary = np.expand_dims(gt_boundary, 1) + offset
                gt_band_boundary = gt_band_boundary.reshape(-1, 2)
                gt_band_index = np.where((gt_band_boundary[:, 0] > 0) &
                                         (gt_band_boundary[:, 1] > 0) &
                                         (gt_band_boundary[:, 0] < rows) &
                                         (gt_band_boundary[:, 1] < cols))

                # trimap = np.zeros((rows, cols), dtype=np.int32)
                # trimap[gt_band_boundary[gt_band_index,0],gt_band_boundary[gt_band_index,1]] = 1
                # cv2.imshow("DD", (trimap * 255).astype(np.uint8))
                # cv2.imshow("ZZ", (gt * 255).astype(np.uint8))
                # cv2.waitKey(0)
                single_ti[l - 1] += len(gt_band_index[0])
                sum_ti[l - 1] += len(gt_band_index[0])

                predicted_l = predict_trimap[gt_band_boundary[gt_band_index, 0], gt_band_boundary[gt_band_index, 1]]
                nii = len(np.where(predicted_l == l)[0])
                single_nii[l - 1] += nii
                sum_nii[l - 1] += nii

                single_ji[l - 1] += len(np.where(predict_trimap == l)[0])
                sum_ji[l - 1] += len(np.where(predict_trimap == l)[0])
            single_val = np.mean(single_nii / (single_ti + single_ji - single_nii))
            val_list.append(single_val)

        val = np.mean(sum_nii / (sum_ti + sum_ji - sum_nii))
        # val_index = get_sort_index(val_list)[0:10]
        return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type':'SCALAR'},
                                                           {'name': 'AntMeanIOUBoundary list', 'value': val_list, 'type': 'TABLE'}]}}


        # def main():
#     predict_data = cv2.imread('/home/mi/res/mask_3.png')
#     predict_data = predict_data[:,:,0]
#     obj_pos = np.where(predict_data>10)
#     predict_data[obj_pos] = 1
#
#     label_data = cv2.imread('/home/mi/res/gt_3.png')
#     label_data = label_data[:,:,0]
#     obj_pos = np.where(label_data > 0)
#     label_data[obj_pos] = 1
#
#     pa = pixel_accuracy([predict_data,predict_data],[label_data,label_data],1)
#     print('%f\n',pa)
#
#     ma = mean_accuracy([predict_data,predict_data],[label_data,label_data],1)
#     print('%f\n',ma)
#
#     mi = mean_iou([predict_data,predict_data],[label_data,label_data],1)
#     print('%f\n',mi)
#
#     fwi = frequency_weighted_iou([predict_data,predict_data],[label_data,label_data],1)
#     print('%f\n',fwi)
#
#     miab = mean_iou_along_boundary([predict_data,predict_data],[label_data,label_data],1,3)
#     print('%f\n',miab)
#
# if __name__ == '__main__':
#     main()

class AntPixelAccuracySeg(AntMeasure):
    def __init__(self, task):
        # paper: Jonathan Long, Evan Shelhamer, etc. Fully Convolutional Networks for Semantic Segmentation
        # formular: \sum_i{n_{ii}}/\sum_i t_i
        super(AntPixelAccuracySeg, self).__init__(task, 'PixelAccuracy')
        assert(task.task_type == 'SEGMENTATION')

        self.is_support_rank = True

    def eva(self, data, label):

        def find_mask_xor(gtm, prm):
            xor1 = gtm - prm
            xor2 = prm - gtm
            # gtm[xor1 == 255] = 254
            # gtm[xor2 == 255] = 1
            gtm[xor1 == 255] = 1
            gtm[xor2 == 255] = 254
            return gtm

        def seed_fill(gtm):
            mask = np.zeros((gtm.shape[0]+2, gtm.shape[1]+2), np.uint8)
            gtm_copy = gtm.copy()
            gtm_copy.dtype=np.uint8
            for x in range(gtm.shape[0]):
                for y in range(gtm.shape[1]):
                    if gtm[x, y] == 0:
                        cv2.floodFill(gtm_copy, mask, (y, x), (0,), (2,), (2,))
                    if gtm[x, y] == 255:
                        cv2.floodFill(gtm_copy, mask, (y, x), (255,), (2,), (2,), flags=cv2.FLOODFILL_FIXED_RANGE)

            return gtm_copy

        def find_hole(gtm, prm):
            gtm = find_mask_xor(gtm, prm)
            # cv2.imwrite('ttt3.png', gt)
            # print 'gtm', gtm
            # print gt.shape

            frame = seed_fill(gtm)
            # cv2.imwrite('tttbefore.png', gt)
            # res = gt.copy()
            # res[frame == 1,0] = 0
            # res[frame == 254,1] = 255
            return frame

        def compute_error_percent(mask):
            tmp = np.zeros(mask.shape)
            tmp[mask == 1] = 1
            tmp[mask == 254] = 1
            error = tmp.sum()
            res = float(error) / (mask.shape[0] * mask.shape[1])
            return res


        def find_edge(gtm, fill_hole):
            edge = find_mask_xor(gtm, fill_hole)
            return edge

        classes_num = len(self.task.class_label)

        sum_nii = np.zeros((1, classes_num))
        sum_ti = np.zeros((1, classes_num))

        if label is not None:
            data = zip(data, label)
        hole_list = []
        edge_list = []
        for pr, gt in data:
            print('start')
            time1 = time.time()
            pr[pr == 1] = 255
            gt[gt == 1] = 255
            time2 = time.time()
            print('change to 255: ', time2-time1)

            pr3c = cv2.merge([pr, pr, pr])
            gt3c = cv2.merge([gt, gt, gt])
            time3 = time.time()
            print('merge: ', time3-time2)
            # print(pr.shape, pr3c.shape, gt.shape, gt3c.shape)

            frame = find_hole(gt, pr)
            time4 = time.time()
            print('find hole: ', time4-time3)
            hole_error = compute_error_percent(frame)
            time5 = time.time()
            print('compute hole error: ', time5-time4)
            # print 'hole error: ', hole_error
            res = pr3c.copy()
            fill_hole = pr.copy()
            time6 = time.time()
            print('copy: ', time6-time5)
            # print(res.shape, frame.shape)
            # print(frame == 1)
            res[frame == 1, 0] = 255
            res[frame == 254, 1] = 0
            fill_hole[frame == 1] = 255
            fill_hole[frame == 254] = 0
            time7 = time.time()
            print('change value: ', time7-time6)
            cv2.imwrite('ttt.png', res)
            cv2.imwrite('fillhole.png', fill_hole)
            time8 = time.time()
            print('save 2 image: ', time8-time7)

            edge = find_edge(gt, fill_hole)
            time9 = time.time()
            print('find edge: ', time9-time8)
            edge_error = compute_error_percent(edge)
            time10 = time.time()
            print('compute edge error: ', time10-time9)
            # print 'edge error', edge_error
            # print 'edge gt', gt
            # print 'fill hole', fill_hole
            # print 'edge', edge
            edge_pr = fill_hole.copy()
            edge_pr = cv2.merge([edge_pr, edge_pr, edge_pr])
            edge_pr[edge == 1, 0] = 120
            edge_pr[edge == 254, 1] = 120
            cv2.imwrite('edge_pr.png', edge_pr)
            hole_list.append(hole_error)
            edge_list.append(edge_error)
            time11 = time.time()
            print('last other: ', time11-time10)
        print('hole list: ', hole_list)
        print('edge list: ', edge_list)
        return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': 0, 'type':'SCALAR'},
                                                           {'name': 'AntPixelAccuracySeg list', 'value': hole_list, 'type': 'TABLE'}]}}
        # return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type':'SCALAR'},
        #                                                    {'name': 'val_list', 'value': val_index, 'type': 'TABLE'}]}}

