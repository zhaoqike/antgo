# encoding=utf-8

from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
import json

tag_num = [3,2,4,3,3,2,2,2,4,2,3,3]



num_of_people = 0
one = 0
two = 1
more = 2

in_out_door = 1
in_door = 0
out_door = 1

light = 2
daylight = 0
dim = 1
color = 2
night = 3

gender = 3
male = 0
female = 1

age = 4
young = 0
old = 1
child = 2

gesture = 5
front = 0
side = 1

position = 6
middle = 0
not_middle = 1


bodylean = 7
straight = 0
lean = 1

decorate = 8
no_dec = 0
hat = 1
face_mask = 2
other_dec = 3

limb = 9
show_limb = 0
hide_limb = 1

shelter = 10
face_shelter = 0
body_shelter = 1

background = 11
simple_bg = 0
regular_bg = 1
complex_bg = 2

# tag_name = [
#     ['单人', '两人', '三人或以上'],
#     ['室内', '室外'],
#     ['白天', '昏暗', '彩色光', '夜间'],
#     ['女', '男', '男女都有'],
#     ['中轻年人', '老年人', '儿童'],
#     ['正脸', '侧脸'],
#     ['居中', '不居中'],
#     ['不倾斜', '倾斜'],
#     ['无明显装饰物', '帽子', '口罩面具', '其他装饰物'],
#     ['有手部肢体露出', '无露出'],
#     ['无遮挡', '面部遮挡', '身体遮挡'],
#     ['背景简单', '正常', '背景有不相关路人']
# ]



tag_name = [
    ['one person', 'two', 'three or more'],
    ['indoor', 'outdoor'],
    ['day', 'dim', 'color', 'night'],
    ['female', 'male', 'both'],
    ['young', 'old', 'child'],
    ['front face', 'side face'],
    ['middle', 'not middle'],
    ['not lean', 'lean'],
    ['no decorate', 'hat', 'mask', 'other decorate'],
    ['show limb', 'hide limb'],
    ['no shelter', 'face shelter', 'body shelter'],
    ['simple bg', 'regular', 'complex bg']
]
#
# row_title = tag_name[0]
# print(row_title)
# row_title = list(map(lambda x: x.decode('utf-8'), row_title))
# row_title = json.dumps(row_title, ensure_ascii=False)
# print(row_title)
#
# row = tag_name[0][0]
# print(row, type(row))
# # row = json.dumps(row, ensure_ascii=False)
# row1 = row.decode('utf-8')
# #
# print(row, row1, type(row), type(row1))
#
# row2 = row.decode('unicode-escape')
# print(row2, type(row2))
# import six