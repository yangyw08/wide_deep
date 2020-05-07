from __future__ import print_function

from os.path import dirname, abspath, join
import sys
import threading

import numpy as np
import tensorflow as tf
import grpc
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import prediction_service_pb2

PACKAGE_DIR = dirname(dirname(abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)
from lib.utils.util import column_to_dtype
from lib.read_conf import Config

tf.app.flags.DEFINE_integer('concurrency', 1, 'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_string('model', 'wide_deep','Model name.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

PRED_FILE = join(dirname(dirname(dirname(abspath(__file__)))), 'data/pred/pred1')


def _read_test_input():
    for line in open(PRED_FILE):
        return line.strip('\n').split('\t')

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    bytesList = tf.train.BytesList(value=[value.encode()])
    return tf.train.Feature(bytes_list=bytesList)

def pred_input_fn(csv_data):
    """Prediction input fn for a single data, used for serving client"""
    conf = Config()
    feature = conf.get_feature_name()
    feature_unused = conf.get_feature_name('unused')
    feature_conf = conf.read_feature_conf()
    csv_default = column_to_dtype(feature, feature_conf)
    csv_default.pop('label')

    feature_dict = {}
    for idx, f in enumerate(csv_default.keys()):
        if f in feature_unused:
            continue
        else:
            print(f,csv_data[idx])
            if csv_default[f] == tf.string:
                feature_dict[f] = _bytes_feature(str(csv_data[idx]))
            else:
                feature_dict[f] = _float_feature(float(csv_data[idx]))
    return feature_dict

def main(_):
    hostport = FLAGS.server
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    #
    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model
    request.model_spec.signature_name = 'serving_default'
    #
    data = _read_test_input()
    feature_dict = pred_input_fn(data)
    #
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    serialized = example.SerializeToString()
    #
    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(serialized, shape=[1]))
    result_future = stub.Predict.future(request, 5.0)
    prediction = result_future.result().outputs['scores']
    # print('True label: ' + str(label))
    print('Prediction: ' + str(np.argmax(prediction.float_val)))

if __name__ == '__main__':
    tf.app.run()