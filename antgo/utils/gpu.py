#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gpu.py
# Author: jian(jian@mltalker.com)
from __future__ import unicode_literals

import os
import re
from antgo.utils.utils import change_env
import subprocess

__all__ = ['change_gpu', 'get_nr_gpu', 'get_gpus','gpu_running_info']


def change_gpu(val):
    return change_env('CUDA_VISIBLE_DEVICES', str(val))


def get_nr_gpu():
    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env is None:
        return 0

    return len(env.split(','))


def get_gpus():
    """ return a list of GPU physical id"""
    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env is None:
        return ''
    return map(int, env.strip().split(','))


def gpu_running_info(pid=None):
    try:
        content = subprocess.check_output('nvidia-smi')
    except:
        return None

    driver_version = re.findall('(?<=Driver Version: )[\d.]+',content)[0]
    gpu_basic_info = re.findall('(?<=\|)[ ]+\d[ ]+\w+ \w+ \w+[ ]+\w+[ ]+(?=\|)',content)
    gpu_num = len(gpu_basic_info)
    gpus=[]
    for gpu_index in range(gpu_num):
        result = re.split('\s+',gpu_basic_info[gpu_index])
        gpus.append(''.join([result[1],' ',result[2],' ',result[3]]))

    gpu_pwr_info = re.findall('\d+W / \d+W',content)
    gpu_pwr_usage=[]
    gpu_pwr_cap=[]
    for gpu_index in range(gpu_num):
        result = re.split('/',gpu_pwr_info[gpu_index])
        pwr_usage = re.findall('\d+',result[0])[0]
        pwr_cap = re.findall('\d+',result[1])[0]
        gpu_pwr_usage.append(int(pwr_usage))
        gpu_pwr_cap.append(int(pwr_cap))

    gpu_mem_info = re.findall('\d+MiB / \d+MiB',content)
    gpu_mem_usage=[]
    gpu_mem_max=[]
    for gpu_index in range(gpu_num):
        result = re.split('/',gpu_mem_info[gpu_index])
        mem_usage = re.findall('\d+',result[0])[0]
        mem_max = re.findall('\d+',result[1])[0]
        gpu_mem_usage.append(int(mem_usage))
        gpu_mem_max.append(int(mem_max))

    gpu_util = re.findall('\d+(?=%)',content)
    gpu_util = [int(util) for util in gpu_util]

    occupy_gpus = []
    if pid is not None:
        pattern = '\d\s+(?={pid})'.format(pid=pid)
        terms = re.findall(pattern,content)
        for term in terms:
            occupy_gpus.append(int(term))

    return {'gpus': gpus, 'driver-version': driver_version,
            'gpu_pwr_usage': gpu_pwr_usage, 'gpu_pwr_cap': gpu_pwr_cap,
            'gpu_mem_usage': gpu_mem_usage, 'gpu_mem_max': gpu_mem_max,
            'gpu_util': gpu_util, 'occupy_gpus': occupy_gpus}
