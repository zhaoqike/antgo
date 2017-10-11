
import json
import MySQLdb

import random
import time
import sys

# from sds.client.clientfactory import ClientFactory
# from sds.client.datumutil import datum
# from sds.client.datumutil import values
# from sds.client.tablescanner import scan_iter
# from sds.auth.ttypes import Credential
# from sds.auth.ttypes import UserType
# from sds.common.constants import ADMIN_SERVICE_PATH
# from sds.common.constants import TABLE_SERVICE_PATH
# from sds.common.constants import DEFAULT_ADMIN_CLIENT_TIMEOUT
# from sds.common.constants import DEFAULT_CLIENT_TIMEOUT
# from sds.errors.ttypes import ServiceException
# from sds.errors.constants import ErrorCode
# from sds.table.constants import DataType
# from sds.table.ttypes import KeySpec, EntityGroupSpec, LocalSecondaryIndexSpec, SecondaryIndexConsistencyMode
# from sds.table.ttypes import TableSchema
# from sds.table.ttypes import TableQuota
# from sds.table.ttypes import TableMetadata
# from sds.table.ttypes import TableSpec
# from sds.table.ttypes import ProvisionThroughput
# from sds.table.ttypes import PutRequest
# from sds.table.ttypes import GetRequest
# from sds.table.ttypes import ScanRequest
# from sds.client.clientfactory import ClientFactory
#
# # change default encodings if unicode is used
# reload(sys)
# sys.setdefaultencoding('utf-8')
#
# endpoint = "http://cnbj-s0.sds.api.xiaomi.com"
# # Set your AppKey and AppSecret
# appKey = "AKNII6ZHBKRQLCQF6L"
# appSecret = "pPyjGHForthlhiwuZzpKaO+gQDflUVLvbXmNHJjQ"
# credential = Credential(UserType.APP_SECRET, appKey, appSecret)
# client_factory = ClientFactory(credential, True)
# # Clients are not thread-safe
# admin_client = client_factory.new_admin_client(endpoint + ADMIN_SERVICE_PATH,
#                                                DEFAULT_ADMIN_CLIENT_TIMEOUT)
# table_client = client_factory.new_table_client(endpoint + TABLE_SERVICE_PATH,
#                                                DEFAULT_CLIENT_TIMEOUT)
#
# table_spec = TableSpec(
#     schema=TableSchema(entityGroup=EntityGroupSpec(attributes=[KeySpec(attribute='userId', asc=True)], enableHash=True),
#                        primaryIndex=[KeySpec(attribute='noteId', asc=False)],
#                        secondaryIndexes={
#                            'mtime_index': LocalSecondaryIndexSpec(indexSchema=[KeySpec(attribute='mtime', asc=False)],
#                                                                   projections=['title', 'noteId'],
#                                                                   consistencyMode=SecondaryIndexConsistencyMode.EAGER),
#                            'category_index': LocalSecondaryIndexSpec(indexSchema=[KeySpec(attribute='category')],
#                                                                      consistencyMode=SecondaryIndexConsistencyMode.LAZY)},
#                        attributes={
#                            'userId': DataType.STRING,
#                            'noteId': DataType.INT64,
#                            'title': DataType.STRING,
#                            'content': DataType.STRING,
#                            'version': DataType.INT64,
#                            'mtime': DataType.INT64,
#                            'category': DataType.STRING_SET
#                        }),
#     metadata=TableMetadata(quota=TableQuota(size=100 * 1024 * 1024),
#                            throughput=ProvisionThroughput(readCapacity=20, writeCapacity=20)))
#
# table_name = "python-test-note"
# categories = ['work', 'travel', 'food']
# M = 20
# try:
#     admin_client.dropTable(table_name)
# except ServiceException, se:
#     assert se.errorCode == ErrorCode.RESOURCE_NOT_FOUND, "Unexpected error: %s" % se.errorCode
#
# admin_client.createTable(table_name, table_spec)
#
# for i in range(0, M):
#     version = 0
#     put = PutRequest(tableName=table_name,
#                      record={
#                          'userId': datum('user1'),
#                          'noteId': datum(i, DataType.INT64),
#                          'title': datum('Title' + str(i)),
#                          'content': datum('note ' + str(i)),
#                          'version': datum(version, DataType.INT64),
#                          'mtime': datum(i * i % 10, DataType.INT64),
#                          'category': datum([categories[i % len(categories)], categories[(i + 1) % len(categories)]],
#                                            DataType.STRING_SET)
#                      })
#     table_client.put(put)
#     print "put record %d" % i
#
# # random access
# print "================= get note by id ===================="
# get = GetRequest(tableName=table_name,
#                  keys={
#                      'userId': datum('user1'),
#                      'noteId': datum(random.randint(0, M), DataType.INT64)
#                  },
#                  attributes=['title', 'content'])
# print "get record: %s" % values(table_client.get(get).item)
#
# # scan by last modify time
# print "================= scan by last modify time ===================="
# start_key = {'userId': datum('user1')}
# stop_key = start_key
# scan = ScanRequest(tableName=table_name,
#                    indexName='mtime_index',
#                    startKey=start_key,  # None or unspecified means begin of the table
#                    stopKey=stop_key,  # None or unspecified means end of the table
#                    attributes=['noteId', 'title', 'mtime'],  # scan all attributes if not specified
#                    limit=M)  # batch size
#
# for record in scan_iter(table_client, scan):
#     print record
#
# # get noteId which contain category food
# print "================= get notes which contain category food ===================="
# start_key = {'userId': datum('user1'), 'category': datum('food')}
# stop_key = start_key
# scan = ScanRequest(tableName=table_name,
#                    indexName='category_index',
#                    startKey=start_key,  # None or unspecified means begin of the table
#                    stopKey=stop_key,  # None or unspecified means end of the table
#                    attributes=['noteId', 'category'],  # scan all attributes if not specified
#                    limit=M)  # batch size
# for record in scan_iter(table_client, scan):
#     print record
#
# admin_client.dropTable(table_name)

def everything_to_db(db_info, dump_dir):
    # print json.dumps(db_info)
    # for ant_name, ant_info in db_info.items():
        # timecostmost = ant_info['timecostmost']['statistic']['value'][0]['value'][1]
        # print 'timecostmost ', json.dumps(timecostmost)
        #
        # task = ant_info['task']
        # print 'task ', task
        #
        # measures = ant_info['measure']
        # for measure in measures:
        #     name = measure['statistic']['name']
        #     value = measure['statistic']['value'][0]['value']
        #     print 'measure ', name, value

    conn = MySQLdb.connect(host='localhost',port=3306,user='root',passwd='1',db='bokeh_test',charset='utf8')
    cursor=conn.cursor()

    # print(conn)
    # print(cursor)
    sql_command = """
        INSERT INTO bokeh_test.bokeh_test_tbl
            (bokeh_test_title,
            model_name,
            pixel_accuracy,
            pixel_accuracy_top_99,
            pixel_accuracy_top_95,
            pixel_accuracy_list,
            pixel_accuracy_bad_10,
            mean_accuracy,
            mean_accuracy_top_99,
            mean_accuracy_top_95,
            mean_accuracy_list,
            mean_accuracy_bad_10,
            mean_iou,
            mean_iou_top_95,
            mean_iou_top_99,
            mean_iou_list,
            mean_iou_bad_10,
            frequency_weighted_iou,
            frequency_weighted_iou_top_99,
            frequency_weighted_iou_top_95,
            frequency_weighted_iou_list,
            frequency_weighted_iou_bad_10,
            mean_iou_boundary,
            mean_iou_boundary_top_99,
            mean_iou_boundary_top_95,
            mean_iou_boundary_list,
            mean_iou_boundary_bad_10,
            cost_time,
            cost_time_sum,
            cost_time_avg,
            cost_time_top_99,
            cost_time_top_95,
            cost_time_list,
            cost_time_bad_10,
            bokeh_test_time)
            VALUES
            ('%s',
            '%s',
            %f,
            %f,
            %f,
            '%s',
            '%s',
            %f,
            %f,
            %f,
            '%s',
            '%s',
            %f,
            %f,
            %f,
            '%s',
            '%s',
            %f,
            %f,
            %f,
            '%s',
            '%s',
            %f,
            %f,
            %f,
            '%s',
            '%s',
            %f,
            %f,
            %f,
            %f,
            %f,
            '%s',
            '%s',
            now());
            """%(db_info['test_name'], db_info['model_name'],
                 db_info['PixelAccuracy']['all'], db_info['PixelAccuracy']['avg99'], db_info['PixelAccuracy']['avg95'], db_info['PixelAccuracy']['list'], db_info['PixelAccuracy']['bad10'],
                 db_info['MeanAccuracy']['all'], db_info['MeanAccuracy']['avg99'], db_info['MeanAccuracy']['avg95'], db_info['MeanAccuracy']['list'], db_info['MeanAccuracy']['bad10'],
                 db_info['MeanIOU']['all'], db_info['MeanIOU']['avg99'], db_info['MeanIOU']['avg95'], db_info['MeanIOU']['list'], db_info['MeanIOU']['bad10'],
                 db_info['FrequencyWeightedIOU']['all'], db_info['FrequencyWeightedIOU']['avg99'], db_info['FrequencyWeightedIOU']['avg95'], db_info['FrequencyWeightedIOU']['list'], db_info['FrequencyWeightedIOU']['bad10'],
                 db_info['MeanIOUBoundary']['all'], db_info['MeanIOUBoundary']['avg99'], db_info['MeanIOUBoundary']['avg95'], db_info['MeanIOUBoundary']['list'], db_info['MeanIOUBoundary']['bad10'],
                 db_info['time']['sum'], db_info['time']['sum'], db_info['time']['avg'], db_info['time']['avg99'], db_info['time']['avg95'], db_info['time']['bad10'], db_info['time']['bad10'])
    f = open('/home/zhaoqike/ddd.txt', 'w')
    print >> f,  sql_command
    f.close()
    ttt = cursor.execute(sql_command)

    print ttt
    conn.commit()

    cursor.close()
    conn.close()
    return None


def store_time_to_db(data):
    # for ant_name, ant_info in data.items():

    return None


def store_measures_to_db():
    return None


if __name__ == '__main__':
    conn = MySQLdb.connect(host='localhost',port=3306,user='root',passwd='1',db='bokeh_test',charset='utf8')
    cursor=conn.cursor()

    print(conn)
    print(cursor)
    # cursor.execute("""
    #     create table if not EXISTS user
    #     (
    #         userid int(11) PRIMARY KEY ,
    #         username VARCHAR(20)
    #     )
    #     """)
    # for i in range(1,10):
    #     print cursor.execute("insert into user(userid,username) values('%d','%s')" %(int(i),'name'+str(i)))
    #     conn.commit()
    # ttt = cursor.execute("""
    #     INSERT INTO bokeh_test_tbl
    #         (bokeh_test_id
    #         )
    #         VALUES
    #         (1
    #         );
    #         """)
    # sql_command =
    ttt = cursor.execute("""
        INSERT INTO bokeh_test.bokeh_test_tbl
            (bokeh_test_title,
            model_name,
            pixel_accuracy,
            pixel_accuracy_top_95,
            pixel_accuracy_top_99,
            pixel_accuracy_list,
            pixel_accuracy_bad_10,
            mean_accuracy,
            mean_accuracy_top_95,
            mean_accuracy_top_99,
            mean_accuracy_list,
            mean_accuracy_bad_10,
            mean_iou,
            mean_iou_top_95,
            mean_iou_top_99,
            mean_iou_list,
            mean_iou_bad_10,
            frequency_weighted_iou,
            frequency_weighted_iou_top_95,
            frequency_weighted_iou_top_99,
            frequency_weighted_iou_list,
            frequency_weighted_iou_bad_10,
            mean_iou_boundary,
            mean_iou_boundary_top_95,
            mean_iou_boundary_top_99,
            mean_iou_boundary_list,
            mean_iou_boundary_bad_10,
            cost_time,
            cost_time_sum,
            cost_time_avg,
            cost_time_top_95,
            cost_time_top_99,
            cost_time_list,
            cost_time_bad_10,
            bokeh_test_time)
            VALUES
            ('ttt',
            'ttt',
            0.88,
            0.88,
            0.88,
            '<xml><name>111.jpg</name><score>0.88</score><name>112.jpg</name><score>0.99</score></xml>',
            '<xml><name>111.jpg</name><score>0.88</score><name>112.jpg</name><score>0.99</score></xml>',
            0.88,
            0.88,
            0.88,
            '<xml><name>111.jpg</name><score>0.88</score><name>112.jpg</name><score>0.99</score></xml>',
            '<xml><name>111.jpg</name><score>0.88</score><name>112.jpg</name><score>0.99</score></xml>',
            0.88,
            0.88,
            0.88,
            '<xml><name>111.jpg</name><score>0.88</score><name>112.jpg</name><score>0.99</score></xml>',
            '<xml><name>111.jpg</name><score>0.88</score><name>112.jpg</name><score>0.99</score></xml>',
            0.88,
            0.88,
            0.88,
            '<xml><name>111.jpg</name><score>0.88</score><name>112.jpg</name><score>0.99</score></xml>',
            '<xml><name>111.jpg</name><score>0.88</score><name>112.jpg</name><score>0.99</score></xml>',
            0.88,
            0.88,
            0.88,
            '<xml><name>111.jpg</name><score>0.88</score><name>112.jpg</name><score>0.99</score></xml>',
            '<xml><name>111.jpg</name><score>0.88</score><name>112.jpg</name><score>0.99</score></xml>',
            0.88,
            0.88,
            0.88,
            0.88,
            0.88,
            '<xml><name>111.jpg</name><score>0.88</score><name>112.jpg</name><score>0.99</score></xml>',
            '<xml><name>111.jpg</name><score>0.88</score><name>112.jpg</name><score>0.99</score></xml>',
            now());
            """)

    print ttt
    conn.commit()

    cursor.close()
    conn.close()


