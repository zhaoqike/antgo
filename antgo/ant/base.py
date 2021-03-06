# encoding=utf-8
# @Time    : 17-3-3
# @File    : common.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from ..utils.serialize import loads,dumps
from ..utils import logger
import zmq
import uuid
import time
import json
import sys
import sys
if sys.version > '3':
  PY3 = True
else:
  PY3 = False


class AntBase(object):
  def __init__(self, ant_name, ant_context=None, ant_token=None):
    self.ant_name = ant_name
    self.app_token = os.environ.get('APP_TOKEN', ant_token)
    self.app_connect = os.environ.get('APP_CONNECT', 'tcp://127.0.0.1:2345')

    # config zmq connect
    self.zmq_context = zmq.Context()
    self.zmq_socket = self.zmq_context.socket(zmq.REQ)
    self.zmq_socket.connect(self.app_connect)
    
    # server flag
    self.app_server = self.__class__.__name__
    if not PY3:
      self.app_server = unicode(self.app_server)

    # core
    self.ant_context = None
    if ant_context is not None:
      self.ant_context = ant_context
      self.ant_context.ant = self

    # time
    self.ant_time_stamp = time.time()

  def send(self, data, stage):
    if self.app_token is not None:
      # 0.step add extra data
      data['APP_TOKEN'] = self.app_token
      data['APP_TIME'] = self.ant_time_stamp
      if self.ant_context is not None:
        if self.ant_context.params is not None:
          data['APP_HYPER_PARAMETER'] = json.dumps(self.ant_context.params)
      data['APP_RPC'] = "INFO"
      data['APP_STAGE'] = stage
      data['APP_NOW_TIME'] = time.time()
      data["APP_NAME"] = self.ant_name
      data["APP_SERVER"] = self.app_server

      # 1.step send info
      self.zmq_socket.send(dumps(data))

      # 2.step ignore any receive info
      self.zmq_socket.recv(copy=False)

  def rpc(self, cmd=""):
    if self.app_token is not None:
      # 0.step config data
      data = {}
      data['APP_TOKEN'] = self.app_token
      data['APP_TIME'] = self.ant_time_stamp
      data['APP_RPC'] = cmd
      data['APP_STAGE'] = 'RPC'
      data['APP_NOW_TIME'] = time.time()
      data["APP_NAME"] = self.ant_name
      data['APP_SERVER'] = self.app_server

      # 1.step send rpc
      self.zmq_socket.send(dumps(data))

      # 2.step receive info
      try:
        response = loads(self.zmq_socket.recv(copy=False))
        if len(response) == 0:
          return None
        return response
      except:
        return None

    return None

  @property
  def stage(self):
    return self.ant_context.stage
  @stage.setter
  def stage(self, val):
    self.ant_context.stage = val

  @property
  def token(self):
    return self.app_token
  @token.setter
  def token(self, val):
    self.app_token = val

  @property
  def name(self):
    return self.ant_name

  @property
  def context(self):
    return self.ant_context

  @context.setter
  def context(self, val):
    self.ant_context = val
    self.ant_context.ant = self

  @property
  def time_stamp(self):
    return self.ant_time_stamp
