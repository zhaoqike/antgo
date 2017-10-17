# encoding=utf-8
# @Time    : 17-5-9
# @File    : challenge.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

from antgo.html.html import *
from antgo.ant.base import *
from antgo.dataflow.common import *
from antgo.measures.statistic import *
from antgo.measures.segmentation_task import *
from antgo.task.task import *
from antgo.utils import logger
from antgo.dataflow.recorder import *
from antgo.measures.deep_analysis import *
import shutil
from antgo.utils.seg_tag import *
from antgo.utils.sds import *
import copy
import json
import tarfile
from datetime import datetime

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
        # invalid token
        logger.error('couldnt load challenge task')
        self.token = None
      elif challenge_task_config['status'] == 'SUSPEND':
        # prohibit submit challenge task frequently
        logger.error('prohibit submit challenge task frequently')
        exit(-1)
      elif challenge_task_config['status'] == 'OK':
        # maybe user token or task token
        if 'task' in challenge_task_config:
          challenge_task = create_task_from_json(challenge_task_config)
          if challenge_task is None:
            logger.error('couldnt load challenge task')
            exit(-1)
          running_ant_task = challenge_task
      else:
        # unknow error
        logger.error('unknow error')
        exit(-1)

    if running_ant_task is None:
      # 0.step load custom task
      custom_task = create_task_from_xml(self.ant_task_config, self.context)
      if custom_task is None:
        logger.error('couldnt load custom task')
        exit(0)
      running_ant_task = custom_task

    assert(running_ant_task is not None)

    # now time stamp
    # now_time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(self.time_stamp))
    now_time_stamp = datetime.now().strftime('%Y%m%d.%H%M%S.%f')

    
    # 0.step warp model (main_file and main_param)
    self.stage = 'CHALLENGE-MODEL'
    # - backup in dump_dir
    main_folder = FLAGS.main_folder()
    main_param = FLAGS.main_param()
    main_file = FLAGS.main_file()

    if not os.path.exists(os.path.join(self.ant_dump_dir, now_time_stamp)):
      os.makedirs(os.path.join(self.ant_dump_dir, now_time_stamp))

    goldcoin = os.path.join(self.ant_dump_dir, now_time_stamp, '%s-goldcoin.tar.gz' % self.ant_name)

    if os.path.exists(goldcoin):
      os.remove(goldcoin)

    tar = tarfile.open(goldcoin, 'w:gz')
    tar.add(os.path.join(main_folder, main_file), arcname=main_file)
    if main_param is not None:
      tar.add(os.path.join(main_folder, main_param), arcname=main_param)
    tar.close()

    # - backup in cloud
    if os.path.exists(goldcoin):
      file_size = os.path.getsize(goldcoin) / 1024.0
      if file_size < 500:
        if not PY3 and sys.getdefaultencoding() != 'utf8':
          reload(sys)
          sys.setdefaultencoding('utf8')
        # model file shouldn't too large (500KB)
        with open(goldcoin, 'rb') as fp:
          self.context.job.send({'DATA': {'MODEL': fp.read()}})

    # 1.step loading test dataset
    logger.info('loading test dataset %s'%running_ant_task.dataset_name)
    ant_test_dataset = running_ant_task.dataset('test',
                                                 os.path.join(self.ant_data_source, running_ant_task.dataset_name),
                                                 running_ant_task.dataset_params)

    with safe_recorder_manager(ant_test_dataset):
      # split data and label
      data_annotation_branch = DataAnnotationBranch(Node.inputs(ant_test_dataset))
      self.context.recorder = RecorderNode(Node.inputs(data_annotation_branch.output(1)))

      self.stage = "CHALLENGE"
      logger.info('start infer process')
      infer_dump_dir = os.path.join(self.ant_dump_dir, now_time_stamp, 'inference')
      if not os.path.exists(infer_dump_dir):
        os.makedirs(infer_dump_dir)
      else:
        shutil.rmtree(infer_dump_dir)
        os.makedirs(infer_dump_dir)

      intermediate_dump_dir = os.path.join(self.ant_dump_dir, now_time_stamp, 'record')
      with safe_recorder_manager(self.context.recorder):
        self.context.recorder.dump_dir = intermediate_dump_dir
        with running_statistic(self.ant_name):
          self.context.call_infer_process(data_annotation_branch.output(0), infer_dump_dir)

      task_running_statictic = get_running_statistic(self.ant_name)
      task_running_statictic = {self.ant_name: task_running_statictic}
      task_running_elapsed_time = task_running_statictic[self.ant_name]['time']['elapsed_time']
      task_running_statictic[self.ant_name]['time']['elapsed_time_per_sample'] = \
          task_running_elapsed_time / float(ant_test_dataset.size)

      logger.info('start evaluation process')
      evaluation_measure_result = []

      filelist_path = os.path.join(self.ant_data_source, running_ant_task.dataset_name, 'test/filelist.txt')
      taglist_path = os.path.join(self.ant_data_source, running_ant_task.dataset_name, 'test/taglist.txt')

      filelist = open(filelist_path).readlines() if os.path.exists(filelist_path) else None
      filelist = map(lambda x: os.path.basename(x.strip()), filelist)
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
          assert(len(tag_measure) == len(row_title))

          row_title.extend(["-"] * (max_tag_len-len(tag_measure)))
          row = tag_measure
          row = list(map(lambda x: str(x)[:10], row))
          row.extend(["-"] * (max_tag_len-len(tag_measure)))

          table.append(row_title)
          table.append(row)
        return table



      def getavgpercent(times, percent, reverse=False):
        sort_times = sorted(times, reverse=reverse)
        lenp = int(len(sort_times) * percent)
        timesp = sort_times[:lenp]
        avgp = sum(timesp) / len(timesp)
        return avgp

      with safe_recorder_manager(RecordReader(intermediate_dump_dir)) as record_reader:
        for measure in running_ant_task.evaluation_measures:
          record_generator = record_reader.iterate_read('predict', 'groundtruth')
          result = measure.eva(record_generator, None)
          result['statistic']['value'][0]['type'] = 'TABLE'
          all = result['statistic']['value'][0]['value']
          val_list = result['statistic']['value'][1]['value']
          avgm = sum(val_list) / len(val_list)
          avgm95 = getavgpercent(val_list, 0.95, reverse=True)
          avgm99 = getavgpercent(val_list, 0.99, reverse=True)
          result['statistic']['value'][0]['value'] = [['all', 'avg', '99%', '95%'], [all, avgm, avgm99, avgm95]]
          val_list = result['statistic']['value'][1]['value']
          val_index = get_sort_index(val_list)[0:10]
          # result['statistic']['value'][1]['value'] = [[filelist[i], val_list[i]] for i in val_index]
          result['statistic']['value'].append({'name': result['statistic']['value'][1]['name'] + ' bad 10', 'value': [[filelist[i], val_list[i]] for i in val_index], 'type': 'TABLE'})
          measure_path = os.path.join(os.path.join(self.ant_dump_dir, now_time_stamp), 'inference', measure.name)
          os.makedirs(measure_path)
          for i in val_index:
              orig = ant_test_dataset.at(i)[0]
              pr = record_reader.read(i)[0]
              pr[pr == 1] = 255
              pr = cv2.merge([pr, pr, pr])
              gt = record_reader.read(i)[1]
              gt[gt == 1] = 255
              gt = cv2.merge([gt, gt, gt])
              all = np.hstack((orig, gt, pr))
              print(all.shape)
              # print(orig.shape, pr.shape, gt.shape)
              cv2.imwrite(os.path.join(measure_path, 'pic'+str(i)+'.png'), orig)
              cv2.imwrite(os.path.join(measure_path, 'pr'+str(i)+'.png'), pr)
              cv2.imwrite(os.path.join(measure_path, 'gt'+str(i)+'.png'), gt)
              cv2.imwrite(os.path.join(measure_path, 'all'+str(i)+'.png'), all)

          if measure.is_support_rank:
            # compute confidence interval
            confidence_interval = bootstrap_confidence_interval(record_reader, time.time(), measure, 2)
            result['statistic']['value'][0]['interval'] = confidence_interval

          evaluation_measure_result.append(result)
          if taglist is not None:
            all_tag_measure = compute_subtag_measures(val_list, taglist)
            table = tag_measure_to_table(all_tag_measure)
            all_tag_measure_context = {'name': measure.name + ' subtag list', 'value': table, 'type': 'TABLE'}
            result['statistic']['value'].append(all_tag_measure_context)
        task_running_statictic[self.ant_name]['measure'] = evaluation_measure_result

        # add hole and edge error pictures
        record_generator = record_reader.iterate_read('predict', 'groundtruth')
        result = AntHoleEdgeSeg(running_ant_task).eva(record_generator, None)
        hole_list = result['statistic']['value'][1]['value']
        hole_pic_list = result['statistic']['value'][2]['value']
        edge_list = result['statistic']['value'][3]['value']
        edge_pic_list = result['statistic']['value'][4]['value']
        hole_list_bad10_index = get_sort_index(hole_list, reverse=True)[0:10]
        edge_list_bad10_index = get_sort_index(edge_list, reverse=True)[0:10]
        hole_path = os.path.join(os.path.join(self.ant_dump_dir, now_time_stamp), 'inference', 'hole')
        edge_path = os.path.join(os.path.join(self.ant_dump_dir, now_time_stamp), 'inference', 'edge')

        os.makedirs(hole_path)
        os.makedirs(edge_path)
        for i in hole_list_bad10_index:
            pr = record_reader.read(i)[0]
            pr[pr == 1] = 255
            pr = cv2.merge([pr, pr, pr])
            gt = record_reader.read(i)[1]
            gt[gt == 1] = 255
            gt = cv2.merge([gt, gt, gt])
            orig = ant_test_dataset.at(i)[0]
            all = np.hstack((orig, gt, pr, hole_pic_list[i]))
            # print(os.path.join(hole_path, 'hole'+str(i)+'.png'))
            cv2.imwrite(os.path.join(hole_path, 'hole'+str(i)+'.png'), hole_pic_list[i])
            cv2.imwrite(os.path.join(hole_path, 'orig'+str(i)+'.png'), orig)
            cv2.imwrite(os.path.join(hole_path, 'all'+str(i)+'.png'), all)
        for i in edge_list_bad10_index:
            pr = record_reader.read(i)[0]
            pr[pr == 1] = 255
            pr = cv2.merge([pr, pr, pr])
            gt = record_reader.read(i)[1]
            gt[gt == 1] = 255
            gt = cv2.merge([gt, gt, gt])
            orig = ant_test_dataset.at(i)[0]
            all = np.hstack((orig, gt, pr, edge_pic_list[i]))
            # print(os.path.join(edge_path, 'edge'+str(i)+'.png'))
            cv2.imwrite(os.path.join(edge_path, 'edge'+str(i)+'.png'), edge_pic_list[i])
            cv2.imwrite(os.path.join(edge_path, 'orig'+str(i)+'.png'), ant_test_dataset.at(i)[0])
            cv2.imwrite(os.path.join(edge_path, 'all'+str(i)+'.png'), all)


      time_path = os.path.join(os.path.join(self.ant_dump_dir, now_time_stamp), 'inference', 'time.txt')


      if os.path.exists(time_path):
        f = open(time_path)
        times = f.readlines()
        times = list(map(lambda x: float(x.strip()), times))
        index = get_sort_index(times, reverse=True)[0:10]
        sumt = sum(times)
        avgt = sumt / len(times)
        avgt95 = getavgpercent(times, 0.95)
        avgt99 = getavgpercent(times, 0.99)
        values = [{'name': 'time cost info', 'value': [['sum', 'avg', '99%', '95%'], [sumt, avgt, avgt99, avgt95]], 'type': 'TABLE'},
                  {'name': 'time cost bad 10 list', 'value': [[filelist[i], times[i]] for i in index], 'type': 'TABLE'}]
        # {'name': 'AntFrequencyWeightedIOUSeg bad 10 list', 'value': [[i, val_list[i]] for i in val_index], 'type': 'TABLE'}
        all_time_statistic = {'statistic':{'name': 'time cost',
                                           'value': values}}
        task_running_statictic[self.ant_name]['timecostmost'] = all_time_statistic

      task_running_statictic[self.ant_name]['task'] = os.path.basename(self.ant_task_config)
      
      logger.info('generate model evaluation report')
      # performace statistic
      db_info = {}
      db_info['test_name'] = self.ant_name
      db_info['model_name'] = self.ant_context.model_name if hasattr(self.ant_context, 'model_name') else 'model'
      db_info['task']=os.path.basename(self.ant_task_config)
      for measure in task_running_statictic[self.ant_name]['measure']:
        measure_name = measure['statistic']['name']
        [all, avg, avg99, avg95] = measure['statistic']['value'][0]['value'][1]
        db_info[measure_name] = {}
        db_info[measure_name]['all'] = all
        db_info[measure_name]['avg'] = avg
        db_info[measure_name]['avg99'] = avg99
        db_info[measure_name]['avg95'] = avg95
        db_info[measure_name]['list'] = zip(filelist, measure['statistic']['value'][1]['value'])
        # db_info[measure_name]['list'] = map(lambda t: '<name>%s</name><value>%f</value>'%(t[0], t[1]), db_info[measure_name]['list'])
        # db_info[measure_name]['list'] = '<xml>' + ''.join(db_info[measure_name]['list']) + '</xml>'
        db_info[measure_name]['list'] = map(lambda t: '%s;%f'%(t[0], t[1]), db_info[measure_name]['list'])
        db_info[measure_name]['list'] = '|'.join(db_info[measure_name]['list'])
        db_info[measure_name]['bad10'] = measure['statistic']['value'][2]['value']
        # db_info[measure_name]['bad10'] = map(lambda t: '<name>%s</name><value>%f</value>'%(t[0], t[1]), db_info[measure_name]['bad10'])
        # db_info[measure_name]['bad10'] = '<xml>' + ''.join(db_info[measure_name]['bad10']) + '</xml>'
        db_info[measure_name]['bad10'] = map(lambda t: '%s;%f'%(t[0], t[1]), db_info[measure_name]['bad10'])
        db_info[measure_name]['bad10'] = '|'.join(db_info[measure_name]['bad10'])


      if task_running_statictic[self.ant_name].has_key('timecostmost'):
        [sumt, avg, avg99, avg95] = task_running_statictic[self.ant_name]['timecostmost']['statistic']['value'][0]['value'][1]
        db_info['time'] = {}
        db_info['time']['sum'] = sumt
        db_info['time']['avg'] = avg
        db_info['time']['avg99'] = avg99
        db_info['time']['avg95'] = avg95
        db_info['time']['bad10'] = task_running_statictic[self.ant_name]['timecostmost']['statistic']['value'][1]['value']
        db_info['time']['bad10'] = map(lambda t: '<name>%s</name><value>%f</value>'%(t[0], t[1]), db_info['time']['bad10'])
        db_info['time']['bad10'] = '<xml>' + '|'.join(db_info['time']['bad10']) + '</xml>'
      everything_to_html(task_running_statictic, os.path.join(self.ant_dump_dir, now_time_stamp))
      everything_to_db(db_info, os.path.join(self.ant_dump_dir, now_time_stamp))

      # compare statistic
      logger.info('start compare process')
      # print(self.ant_data_source, self.ant_context, self.ant_name, self.ant_task_config, self.ant_dump_dir, self.ant_task_config)

      # compare statistic
      logger.info('start checking significance difference')
      # finding benchmark model from server
      benchmark_info = self.rpc("TASK-BENCHMARK")
      benchmark_model_data = {}
      if benchmark_info is not None:
        for bmd in benchmark_info:
          benchmark_name = bmd['benchmark_name']
          benchmark_record = bmd['benchmark_record']
          benchmark_report = bmd['benchmark_report']

          # download benchmark record from url
          benchmark_record = self.download(benchmark_record,
                                           target_path=os.path.join(self.ant_dump_dir, now_time_stamp, 'benchmark'),
                                           target_name='%s.tar.gz'%benchmark_name,
                                           archive=benchmark_name)

          if 'record' not in benchmark_model_data:
            benchmark_model_data['record'] = {}
            benchmark_model_data['record'][benchmark_name] = benchmark_record

          if 'report' not in benchmark_model_data:
            benchmark_model_data['report'] = {}
            benchmark_model_data['report'][benchmark_name] = benchmark_report

      if benchmark_model_data is not None and 'record' in benchmark_model_data:
        benchmark_model_record = benchmark_model_data['record']

        task_running_statictic[self.ant_name]['significant_diff'] = {}
        for measure in running_ant_task.evaluation_measures:
          if measure.is_support_rank:
            significant_diff_score = []
            for benchmark_model_name, benchmark_model_address in benchmark_model_record.items():
              with safe_recorder_manager(RecordReader(intermediate_dump_dir)) as record_reader:
                with safe_recorder_manager(RecordReader(benchmark_model_address)) as benchmark_record_reader:
                  s = bootstrap_ab_significance_compare([record_reader, benchmark_record_reader], time.time(), measure, 50)
                  significant_diff_score.append({'name': benchmark_model_name, 'score': s})
            task_running_statictic[self.ant_name]['significant_diff'][measure.name] = significant_diff_score

      # deep analysis
      logger.info('start deep analysis')
      # finding all reltaed model result from server
      benchmark_model_statistic = None
      if benchmark_model_data is not None and 'report' in benchmark_model_data:
        benchmark_model_statistic = benchmark_model_data['report']
      
      # task_running_statictic={self.ant_name:
      #                           {'measure':[
      #                             {'statistic': {'name': 'MESR',
      #                                            'value': [{'name': 'MESR', 'value': 0.4, 'type':'SCALAR'}]},
      #                                            'info': [{'id':0,'score':0.8,'category':1},
      #                                                     {'id':1,'score':0.3,'category':1},
      #                                                     {'id':2,'score':0.9,'category':1},
      #                                                     {'id':3,'score':0.5,'category':1},
      #                                                     {'id':4,'score':1.0,'category':1}]},
      #                             {'statistic': {'name': "SE",
      #                                            'value': [{'name': 'SE', 'value': 0.5, 'type': 'SCALAR'}]},
      #                                            'info': [{'id':0,'score':0.4,'category':1},
      #                                                     {'id':1,'score':0.2,'category':1},
      #                                                     {'id':2,'score':0.1,'category':1},
      #                                                     {'id':3,'score':0.5,'category':1},
      #                                                     {'id':4,'score':0.23,'category':1}]}]}}
      
      for measure_result in task_running_statictic[self.ant_name]['measure']:
        if 'info' in measure_result and len(measure_result['info']) > 0:
          measure_name = measure_result['statistic']['name']
          measure_data = measure_result['info']
          
          # independent analysis per category for classification problem
          measure_data_list = []
          if running_ant_task.class_label is not None and len(running_ant_task.class_label) > 1:
            for cl in running_ant_task.class_label:
              measure_data_list.append([md for md in measure_data if md['category'] == cl])
          
          if len(measure_data_list) == 0:
            measure_data_list.append(measure_data)
          
          for category_id, category_measure_data in enumerate(measure_data_list):
            if 'analysis' not in task_running_statictic[self.ant_name]:
              task_running_statictic[self.ant_name]['analysis'] = {}
            
            if measure_name not in task_running_statictic[self.ant_name]['analysis']:
              task_running_statictic[self.ant_name]['analysis'][measure_name] = {}
  
            # reorganize as list
            method_samples_list = [{'name': self.ant_name, 'data': category_measure_data}]
            if benchmark_model_statistic is not None:
              # extract statistic data from benchmark
              for benchmark_model_data in benchmark_model_statistic:
                for benchmark_name, benchmark_statistic_data in benchmark_model_data.items():
                  # finding corresponding measure
                  for benchmark_measure_result in benchmark_statistic_data['measure']:
                    if benchmark_measure_result['statistic']['name'] == measure_name:
                      benchmark_measure_data = benchmark_measure_result['info']
                      
                      # finding corresponding category
                      sub_benchmark_measure_data = None
                      if running_ant_task.class_label is not None and len(running_ant_task.class_label) > 1:
                        sub_benchmark_measure_data = \
                          [md for md in benchmark_measure_data if md['category'] == running_ant_task.class_label[category_id]]
                      if sub_benchmark_measure_data is None:
                        sub_benchmark_measure_data = benchmark_measure_data
                      
                      method_samples_list.append({'name': benchmark_name, 'data': sub_benchmark_measure_data})
                      
                      break
                  break
  
            # reorganize data as score matrix
            method_num = len(method_samples_list)
            samples_num = len(method_samples_list[0]['data'])
            method_measure_mat = np.zeros((method_num, samples_num))
            samples_id = np.zeros((samples_num), np.uint64)
  
            for method_id, method_measure_data in enumerate(method_samples_list):
              if method_id == 0:
                # record sample id
                for sample_id, sample in enumerate(method_measure_data['data']):
                  samples_id[sample_id] = sample['id']
  
              for sample_id, sample in enumerate(method_measure_data['data']):
                  method_measure_mat[method_id, sample_id] = sample['score']
  
            # check method_measure_mat is binary (0 or 1)
            is_binary = False
            check_data = method_samples_list[0]['data'][0]['score']
            if type(check_data) == int:
              is_binary = True
  
            # score matrix analysis
            if not is_binary:
              s, ri, ci, lr_samples, mr_samples, hr_samples = \
                continuous_multi_model_measure_analysis(method_measure_mat, samples_id.tolist(), ant_test_dataset)
              
              analysis_tag = 'Global'
              if len(measure_data_list) > 1:
                analysis_tag = 'Global-Category ('+str(running_ant_task.class_label[category_id]) + ')'
              
              model_name_ri = [method_samples_list[r]['name'] for r in ri]
              task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag] = \
                            {'value': s,
                             'type': 'MATRIX',
                             'x': ci,
                             'y': model_name_ri,
                             'sampling': [{'name': 'High Score Region', 'data': hr_samples},
                                          {'name': 'Middle Score Region', 'data': mr_samples},
                                          {'name': 'Low Score Region', 'data': lr_samples}]}
  
              # group by tag
              tags = getattr(ant_test_dataset, 'tag', None)
              if tags is not None:
                for tag in tags:
                  g_s, g_ri, g_ci, g_lr_samples, g_mr_samples, g_hr_samples = \
                    continuous_multi_model_measure_analysis(method_measure_mat,
                                                            samples_id.tolist(),
                                                            ant_test_dataset,
                                                            filter_tag=tag)
                  
                  analysis_tag = 'Group'
                  if len(measure_data_list) > 1:
                    analysis_tag = 'Group-Category (' + str(running_ant_task.class_label[category_id]) + ')'

                  if analysis_tag not in task_running_statictic[self.ant_name]['analysis'][measure_name]:
                    task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag] = []

                  model_name_ri = [method_samples_list[r]['name'] for r in g_ri]
                  tag_data = {'value': g_s,
                              'type': 'MATRIX',
                              'x': g_ci,
                              'y': model_name_ri,
                              'sampling': [{'name': 'High Score Region', 'data': g_hr_samples},
                                           {'name': 'Middle Score Region', 'data': g_mr_samples},
                                           {'name': 'Low Score Region', 'data': g_lr_samples}]}
  
                  task_running_statictic[self.ant_name]['analysis'][measure_name][analysis_tag].append((tag, tag_data))
            else:
              s, ri, ci, region_95, region_52, region_42, region_13, region_one, region_zero = \
                discrete_multi_model_measure_analysis(method_measure_mat,
                                                      samples_id.tolist(),
                                                      ant_test_dataset)
              task_running_statictic[self.ant_name]['analysis'][measure_name]['global'] = \
                              {'value': s,
                               'type': 'MATRIX',
                               'x': ci,
                               'y': ri,
                               'sampling': [{'name': '95%', 'data': region_95},
                                            {'name': '52%', 'data': region_52},
                                            {'name': '42%', 'data': region_42},
                                            {'name': '13%', 'data': region_13},
                                            {'name': 'best', 'data': region_one},
                                            {'name': 'zero', 'data': region_zero}]}
  
              # group by tag
              tags = getattr(ant_test_dataset, 'tag', None)
              if tags is not None:
                for tag in tags:
                  g_s, g_ri, g_ci, g_region_95, g_region_52, g_region_42, g_region_13, g_region_one, g_region_zero = \
                    discrete_multi_model_measure_analysis(method_measure_mat,
                                                            samples_id.tolist(),
                                                            ant_test_dataset,
                                                            filter_tag=tag)
                  if 'group' not in task_running_statictic[self.ant_name]['analysis'][measure_name]:
                    task_running_statictic[self.ant_name]['analysis'][measure_name]['group'] = []
  
                  tag_data = {'value': g_s,
                              'type': 'MATRIX',
                              'x': g_ci,
                              'y': g_ri,
                              'sampling': [{'name': '95%', 'data': region_95},
                                           {'name': '52%', 'data': region_52},
                                           {'name': '42%', 'data': region_42},
                                           {'name': '13%', 'data': region_13},
                                           {'name': 'best', 'data': region_one},
                                           {'name': 'zero', 'data': region_zero}]}
  
                  task_running_statictic[self.ant_name]['analysis'][measure_name]['group'].append((tag, tag_data))


      # notify
      self.context.job.send({'DATA': {'REPORT': task_running_statictic, 'RECORD': intermediate_dump_dir}})

      # generate report html
      logger.info('generate model evaluation report')
      everything_to_html(task_running_statictic, os.path.join(self.ant_dump_dir, now_time_stamp), ant_test_dataset)
