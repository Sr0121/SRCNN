import tensorflow as tf
import os
from SRCNN import *

flags = tf.app.flags
flags.DEFINE_float('lr', 0.0001, 'learning rate')
flags.DEFINE_float('beta1', 0.9, 'beta1')
flags.DEFINE_float('beta2', 0.999, 'beta2')
flags.DEFINE_float('lambd', 0.001, 'coeff for adversarial loss')
flags.DEFINE_string('dataset_dir', 'data', 'dataset directory')
flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'checkpoint directory')
flags.DEFINE_string('sample_dir', 'sample', 'sample directory')
flags.DEFINE_string('test_dir', 'test', 'test directory')
flags.DEFINE_string('model_dir', 'ImageNet', 'using imagenet dataset')
flags.DEFINE_string('logs_dir', 'logs', 'log directory')
flags.DEFINE_bool('is_crop', True, 'crop images')
flags.DEFINE_integer('epoches',1500, 'training epoches')
flags.DEFINE_integer('fine_size', 33, 'fine size')
flags.DEFINE_string('train_set', 'ImageNet', 'train phase')
flags.DEFINE_string('val_set', 'Set5', 'val phase')
flags.DEFINE_string('test_set', 'Set14', 'test phase')
flags.DEFINE_bool('is_testing', False, 'training or testing')
flags.DEFINE_bool('is_training', False, 'training or testing')
flags.DEFINE_integer('scale',3,'the scale of bicubic')
flags.DEFINE_string('file_name', None, 'file name')
flags.DEFINE_string('load_model', 'SRCNN.model-1', 'load model step')
flags.DEFINE_bool('has_model', False, 'has model to train')
flags.DEFINE_integer('load_model_counter',0,'load model counter')
FLAGS = flags.FLAGS

def check_dir():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.mkdir(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.logs_dir):
        os.mkdir(FLAGS.logs_dir)
    if not os.path.exists(FLAGS.test_dir):
        os.mkdir(FLAGS.test_dir)


def main(_):
    check_dir()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        srcnn = SRCNN(FLAGS, batch_size=8, input_size=33, output_size=33, input_channels=3, sess=sess)
        srcnn.build_model()
        # srcnn.sample(1, 1)
        if FLAGS.is_training:
            srcnn.train()
        if FLAGS.is_testing:
            srcnn.test()


if __name__=='__main__':
    with tf.device('/gpu:0'):
        tf.app.run()
