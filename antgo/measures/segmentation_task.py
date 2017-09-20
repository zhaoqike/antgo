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


class AntPixelAccuracySeg(AntMeasure):
    def __init__(self, task):
        # paper: Jonathan Long, Evan Shelhamer, etc. Fully Convolutional Networks for Semantic Segmentation
        # formular: \sum_i{n_{ii}}/\sum_i t_i
        super(AntPixelAccuracySeg, self).__init__(task, 'PixelAccuracy')
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
                                                           {'name': 'AntPixelAccuracySeg bad 10 list', 'value': val_list, 'type': 'TABLE'}]}}
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
                                                           {'name': 'AntMeanAccuracySeg bad 10 list', 'value': val_list, 'type': 'TABLE'}]}}


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
                                                           {'name': 'AntMeanIOUSeg bad 10 list', 'value': val_list, 'type': 'TABLE'}]}}


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
                                                           {'name': 'AntFrequencyWeightedIOUSeg bad 10 list', 'value': val_list, 'type': 'TABLE'}]}}


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
                                                           {'name': 'AntMeanIOUBoundary bad 10 list', 'value': val_list, 'type': 'TABLE'}]}}


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