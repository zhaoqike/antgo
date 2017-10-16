# -*- coding: UTF-8 -*-
# File: utils.py
# Author: jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os, sys
from contextlib import contextmanager
import inspect, functools
from datetime import datetime
import time
import collections
import numpy as np
import six
import tarfile


__all__ = ['change_env',
        'map_arg',
        'get_rng', 'memoized',
        'get_tqdm_kwargs',
        'targz'
        ]

#def expand_dim_if_necessary(var, dp):
#    """
#    Args:
#        var: a tensor
#        dp: a numpy array
#    Return a reshaped version of dp, if that makes it match the valid dimension of var
#    """
#    shape = var.get_shape().as_list()
#    valid_shape = [k for k in shape if k]
#    if dp.shape == tuple(valid_shape):
#        new_shape = [k if k else 1 for k in shape]
#        dp = dp.reshape(new_shape)
#    return dp

@contextmanager
def change_env(name, val):
  oldval = os.environ.get(name, None)
  os.environ[name] = val
  yield
  if oldval is None:
    del os.environ[name]
  else:
    os.environ[name] = oldval

class memoized(object):
  '''Decorator. Caches a function's return value each time it is called.
  If called later with the same arguments, the cached value is returned
  (not reevaluated).
  '''
  def __init__(self, func):
    self.func = func
    self.cache = {}

  def __call__(self, *args):
    if not isinstance(args, collections.Hashable):
      # uncacheable. a list, for instance.
      # better to not cache than blow up.
      return self.func(*args)
    if args in self.cache:
      return self.cache[args]
    else:
      value = self.func(*args)
      self.cache[args] = value
      return value

  def __repr__(self):
    '''Return the function's docstring.'''
    return self.func.__doc__

  def __get__(self, obj, objtype):
    '''Support instance methods.'''
    return functools.partial(self.__call__, obj)


#_GLOBAL_MEMOIZED_CACHE = dict()
#def global_memoized(func):
    #""" Make sure that the same `memoized` object is returned on different
        #calls to global_memoized(func)
    #"""
    #ret = _GLOBAL_MEMOIZED_CACHE.get(func, None)
    #if ret is None:
        #ret = _GLOBAL_MEMOIZED_CACHE[func] = memoized(func)
    #return ret

def map_arg(**maps):
  """
  Apply a mapping on certains argument before calling original function.
  maps: {key: map_func}
  """
  def deco(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      argmap = inspect.getcallargs(func, *args, **kwargs)
      for k, map_func in six.iteritems(maps):
        if k in argmap:
          argmap[k] = map_func(argmap[k])
      return func(**argmap)
    return wrapper
  return deco

def get_rng(obj=None, seed=None):
  """ obj: some object to use to generate random seed"""
  if seed is None:
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
  return np.random.RandomState(seed)


def get_tqdm_kwargs(**kwargs):
    default = dict(
            smoothing=0.5,
            dynamic_ncols=True,
            ascii=True,
            bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_noinv_fmt}]'
            )
    f = kwargs.get('file', sys.stderr)
    if f.isatty():
        default['mininterval'] = 0.5
    else:
        default['mininterval'] = 60
    default.update(kwargs)
    return default


def get_sort_index(lst, reverse=False):
    lst = [(lst[i], i) for i in range(len(lst))]
    sort_lst = sorted(lst, reverse=reverse)
    index = list(map(lambda value_index: value_index[1], sort_lst))
    return index


def targz(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
