# encoding=utf-8
# @Time    : 17-5-9
# @File    : challenge.py
# @Author  :
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

from antgo.html.html import *
from .base import *
from ..dataflow.common import *
from ..measures.statistic import *
from ..task.task import *
from ..utils import logger
from ..dataflow.recorder import *
import shutil
from antgo.utils.seg_tag import *
import copy
import json

class AntChallenge(AntBase):
  def __init__(self, ant_context,
               ant_name,
               ant_data_folder,
               ant_dump_dir,
               ant_token,
               ant_task_config=None):
    super(AntChallenge, self).__init__(ant_name, ant_context, ant_token)
    self.ant_data_source = ant_data_folder
    self.ant_dump_dir = ant_dump_dir
    self.ant_context.ant = self
    self.ant_task_config = ant_task_config

  def start(self):
    # 0.step loading challenge task
    running_ant_task = None
    if self.token is not None:
      # 0.step load challenge task
      challenge_task_config = self.rpc("TASK-CHALLENGE")
      if challenge_task_config is None:
        logger.error('couldnt load challenge task')
        exit(0)
      elif challenge_task_config['status'] == 'SUSPEND':
        # prohibit submit challenge task frequently
        logger.error('prohibit submit challenge task frequently')
        exit(0)
      elif challenge_task_config['status'] == 'UNAUTHORIZED':
        # unauthorized submit challenge task
        logger.error('unauthorized submit challenge task')
        exit(0)
      elif challenge_task_config['status'] == 'OK':
        challenge_task = create_task_from_json(challenge_task_config)
        if challenge_task is None:
          logger.error('couldnt load challenge task')
          exit(0)
        running_ant_task = challenge_task

    if running_ant_task is None:
      # 0.step load custom task
      custom_task = create_task_from_xml(self.ant_task_config, self.context)
      if custom_task is None:
        logger.error('couldnt load custom task')
        exit(0)
      running_ant_task = custom_task

    assert(running_ant_task is not None)

    # 1.step loading test dataset
    logger.info('loading test dataset %s'%running_ant_task.dataset_name)
    ant_test_dataset = running_ant_task.dataset('test',
                                                 os.path.join(self.ant_data_source, running_ant_task.dataset_name),
                                                 running_ant_task.dataset_params)
    
    with safe_recorder_manager(ant_test_dataset):
      # split data and label
      data_annotation_branch = DataAnnotationBranch(Node.inputs(ant_test_dataset))
      self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))
  
      self.stage = "INFERENCE"
      logger.info('start infer process')
      now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(self.time_stamp))
      infer_dump_dir = os.path.join(self.ant_dump_dir, now_time, 'inference')
      if not os.path.exists(infer_dump_dir):
        os.makedirs(infer_dump_dir)
      else:
        shutil.rmtree(infer_dump_dir)
        os.makedirs(infer_dump_dir)
      
      with safe_recorder_manager(self.context.recorder):
        with running_statistic(self.ant_name):
          self.context.call_infer_process(data_annotation_branch.output(0), infer_dump_dir)

      task_running_statictic = get_running_statistic(self.ant_name)
      task_running_statictic = {self.ant_name: task_running_statictic}
      task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
      task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
          task_running_elapsed_time / float(ant_test_dataset.size)
  
      self.stage = 'EVALUATION'
      logger.info('start evaluation process')
      evaluation_measure_result = []

      filelist_path = os.path.join(self.ant_data_source, running_ant_task.dataset_name, 'test/filelist.txt')
      taglist_path = os.path.join(self.ant_data_source, running_ant_task.dataset_name, 'test/taglist.txt')

      filelist = open(filelist_path).readlines() if os.path.exists(filelist_path) else None
      taglist = open(taglist_path).readlines() if os.path.exists(taglist_path) else None
      taglist = list(map(lambda x: list(map(lambda y: int(y), x.split(','))), taglist)) if taglist is not None else None

      def compute_subtag_measures(val_list, tag_list):
        # print(len(val_list), len(tag_list))
        # print(type(val_list), type(tag_list))
        assert(len(val_list) == len(tag_list))
        all_tag_measure = []
        for ti in range(len(tag_list[0])):
          tag_measure = [-1] * tag_num[ti]
          for tv in range(tag_num[ti]):
            fv = filter(lambda val_tag: val_tag[1][ti] == tv, zip(val_list, tag_list))
            fv = list(map(lambda val_tag: val_tag[0], fv))
            if(len(fv) != 0):
              tag_measure[tv] = sum(fv) / len(fv)
          all_tag_measure.append(tag_measure)
        return all_tag_measure


      def tag_measure_to_table(all_tag_measure):
        max_tag_len = max(tag_num)
        table = []
        for i in range(len(all_tag_measure)):
          tag_measure = copy.deepcopy(all_tag_measure[i])
          row_title = copy.deepcopy(tag_name[i])
          # print(row_title)
          # row_title = list(map(lambda xx: xx.decode('utf-8'), row_title))
          # row_title = map(lambda xx: xx.decode('utf-8'), row_title)
          # title = []
          # for t in row_title:
          #   title.append(json.dumps(t, ensure_ascii=False))
          # row_title = title
          # fuck = ['单人', '两人', '三人或以上']
          # print(fuck)
          # fuck = map(lambda xx: xx.decode('utf-8'), fuck)
          # print(fuck)
          # print(tag_measure, row_title)
          # print(len(tag_measure), len(row_title), type(tag_measure), type(row_title))
          assert(len(tag_measure) == len(row_title))

          row_title.extend(["-"] * (max_tag_len-len(tag_measure)))
          row = tag_measure
          row = list(map(lambda x: str(x)[:10], row))
          row.extend(["-"] * (max_tag_len-len(tag_measure)))

          table.append(row_title)
          table.append(row)
        return table


  
      with safe_recorder_manager(RecordReader(infer_dump_dir)) as record_reader:
        for measure in running_ant_task.evaluation_measures:
          record_generator = record_reader.iterate_read('predict', 'groundtruth')
          result = measure.eva(record_generator, None)
          # if 'bad 10 list' in result['statistic']['value'][1]['name']:
          val_list = result['statistic']['value'][1]['value']
          val_index = get_sort_index(val_list)[0:10]
          result['statistic']['value'][1]['value'] = [[filelist[i], val_list[i]] for i in val_index]
          evaluation_measure_result.append(result)

          if taglist is not None:
            all_tag_measure = compute_subtag_measures(val_list, taglist)
            table = tag_measure_to_table(all_tag_measure)
            all_tag_measure_context = {'name': measure.name + ' subtag list', 'value': table, 'type': 'TABLE'}
            result['statistic']['value'].append(all_tag_measure_context)
        task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result


      time_path = os.path.join(os.path.join(self.ant_dump_dir, now_time), 'inference', 'time.txt')

      def getavgpercent(times, percent):
        sort_times = sorted(times)
        lenp = int(len(sort_times) * percent)
        timesp = sort_times[:lenp]
        avgp = sum(timesp) / len(timesp)
        return avgp


      if os.path.exists(time_path):
        f = open(time_path)
        times = f.readlines()
        times = list(map(lambda x: float(x.strip()), times))
        index = get_sort_index(times, reverse=True)[0:10]
        sumt = sum(times)
        avgt = sumt / len(times)
        avgt95 = getavgpercent(times, 0.95)
        avgt99 = getavgpercent(times, 0.99)
        values = [{'name': 'time cost info', 'value': [['sum', 'avg', '95%', '99%'], [sumt, avgt, avgt95, avgt99]], 'type': 'TABLE'},
                  {'name': 'time cost bad 10 list', 'value': [[filelist[i], times[i]] for i in index], 'type': 'TABLE'}]
        # {'name': 'AntFrequencyWeightedIOUSeg bad 10 list', 'value': [[i, val_list[i]] for i in val_index], 'type': 'TABLE'}
        all_time_statistic = {'statistic':{'name': 'time cost',
                                           'value': values}}
        task_running_statictic[self.ant_name]['timecostmost'] = all_time_statistic
      
      logger.info('generate model evaluation report')
      # performace statistic
      everything_to_html(task_running_statictic, os.path.join(self.ant_dump_dir, now_time))

      # compare statistic
      logger.info('start compare process')
      # print(self.ant_data_source, self.ant_context, self.ant_name, self.ant_task_config, self.ant_dump_dir, self.ant_task_config)


      # notify
      self.context.job.send({'DATA': {'STATISTIC': task_running_statictic}})
