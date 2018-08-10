import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import numpy as np
import os
from glob import glob
from ops import *
from utils import *
import h5py


class SRCNN:
    model_name = 'SRCNN'
    def __init__(self, config, batch_size=1, input_size=256, output_size=256, input_channels=3, sess=None):
        self.input_size = input_size
        self.output_size = output_size
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.images_norm = True
        self.config = config
        self.sess = sess
        
    def generator(self, input_x, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            with slim.arg_scope([slim.conv2d],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                weights_regularizer=None,
                                activation_fn=None,
                                normalizer_fn=None,
                                padding='VALID'):
                input_x_padding = add_padding(input_x, 33, 7, 1)
                conv1 = tf.nn.relu(slim.conv2d(input_x_padding, 64, 7, 1, scope='g_conv1'))
                print(conv1)
                shortcut = conv1
                # res_block(input_x, out_channels=64, k=3, s=1, scope='res_block'):
                res1 = res_block(conv1, 64, 7, 1, scope='g_res1')
                res2 = res_block(res1, 64, 7, 1, scope='g_res2')
                res3 = res_block(res2, 64, 7, 1, scope='g_res3')
                res4 = res_block(res3, 64, 7, 1, scope='g_res4')
                res5 = res_block(res4, 64, 7, 1, scope='g_res5')

                res5_padding = add_padding(res5, 33, 7, 1)
                conv2 = slim.batch_norm(slim.conv2d(res5_padding, 64, 7, 1, scope='g_conv2', padding='VALID')
                                        , scope='g_bn_conv2')
                print(conv2)
                conv2_out = shortcut+conv2
                print(conv2_out)

                conv2_out_padding = add_padding(conv2_out, 33, 7, 1)
                conv3 = tf.nn.relu(slim.conv2d(conv2_out_padding, 256, 7, 1, scope='g_conv3', padding='VALID'))

                conv3_padding = add_padding(conv3, 33, 7, 1)
                conv4 = tf.nn.relu(slim.conv2d(conv3_padding, 256, 7, 1, scope='g_conv4'))

                conv4_padding = add_padding(conv4, 33, 3, 1)
                conv5 = slim.conv2d(conv4_padding, 3, 3, 1, scope='g_conv5')

                self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
                return tf.nn.tanh(conv5)


    def build_model(self):
        self.input_target = tf.placeholder(tf.float32, [None, self.output_size, self.output_size, self.input_channels], name='input_target')
        self.input_source = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, self.input_channels], name='input_source')

        self.real = self.input_target

        self.fake = self.generator(self.input_source, reuse=False)
        self.psnr = PSNR(self.real, self.fake)
        self.g_loss = self.inference_loss(self.real, self.fake)
        print('d, g_loss')
        self.g_optim = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=self.config.beta1, beta2=self.config.beta2).minimize(self.g_loss, var_list=self.g_vars)
        print('d_optim')

        self.saver = tf.train.Saver()
        print('builded model...')

    def inference_loss(self, real, fake):
        loss = tf.sqrt(tf.reduce_mean(tf.square(real - fake)))
        return loss

    def train(self):
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        times = []
        epochs = []
        psnrs = []
        # data/train/*.*
        data = glob(os.path.join(self.config.dataset_dir, 'train', self.config.train_set, '*.*'))

        bool_check, counter = self.load_model(self.config.checkpoint_dir)
        if bool_check:
            print('[!!!] load model successfully')
            counter = counter + 1
        else:
            print('[***] fail to load model')
            counter = 1


        # last_ave_loss = None
        # loss_dic = None

        for epoch in range(self.config.epoches):
            start_time = time.time()
            end_time = time.time()
            np.random.shuffle(data)

            for file_name in data:
                images, images_blur = get_images(file_name, self.config.fine_size, True, self.config.scale)
                images = np.asarray(images)
                images_blur = np.asarray(images_blur)
                images = images[:images.shape[0] // 8 * 8, :, :, :]
                images_blur = images_blur[:images_blur.shape[0] // 8 * 8, :, :, :]

                batches_ori = images.reshape([-1, self.batch_size, self.input_size, self.input_size, 3])
                batches_blur = images_blur.reshape([-1, self.batch_size, self.input_size, self.input_size, 3])

                # if last_ave_loss is not None and loss_dic[str(file_name)] > last_ave_loss:
                #     batches = batches[
                #         np.random.choice(range(batches.shape[0]),
                #                          np.random.randint(batches.shape[0]//8, batches.shape[0]//4))]
                # else:
                #     batches = batches[
                #         np.random.choice(range(batches.shape[0]),
                #                          np.random.randint(batches.shape[0]//8))]

                # batches = batches[np.random.choice(range(batches.shape[0]),np.random.randint(batches.shape[0]//8))]

                rand_index = np.random.choice(range(batches_ori.shape[0]),np.random.randint(batches_ori.shape[0]//8))

                for index in rand_index:
                    # batch_x = [blur_images(imgs, self.images_norm, self.output_size,self.config.scale) for imgs in batch_x]
                    input_batch = batches_blur[index]
                    target_batch = batches_ori[index]
                    input_batch = np.array(input_batch).astype(np.float32)
                    target_batch = np.array(target_batch).astype(np.float32)

                    _, content_loss, psnr = self.sess.run([self.g_optim, self.g_loss, self.psnr],
                                                          feed_dict={self.input_target: target_batch,
                                                                     self.input_source: input_batch})

                   # self.save_model(self.config.checkpoint_dir, counter)

                    if np.mod(counter, 500) == 0:
                        self.save_model(self.config.checkpoint_dir, counter)

                    counter = counter + 1

                # loss_dic, last_ave_loss = self.calculate_loss(epoch)

            print('---------------------------------------')

            print('epoch{}:total_time:{:.4f},content_loss:{:4f},psnr:{:.4f}'.format(epoch,
                                                                                    end_time - start_time,
                                                                                    content_loss, psnr))
            self.sample(epoch)
            end_time = time.time()
            print(end_time - start_time)
            times.append(end_time-start_time)
            epochs.append(epoch)
            psnrs.append(psnr)
            # loss_dic, last_ave_loss = self.calculate_loss(epoch)

            if epoch % 20 == 0 and epoch != 0:
                self.write_data(times,epochs,psnrs,epoch)

            print('---------------------------------------')

    def write_data(self,times,epochs,psnrs,epoch):
        savepath = os.path.join(os.getcwd(), 'checkpoint','train{}.h5'.format(epoch))
        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset('times', data=times)
            hf.create_dataset('epochs', data=epochs)
            hf.create_dataset('psnrs', data=psnrs)
        print("finish write")


    def calculate_loss(self, epoch):
        files = glob(os.path.join(self.config.dataset_dir, 'train', self.config.train_set, '*.*'))
        loss_dic = {}

        for file in files:
            h_, w_, input_, sample_ = get_sample_image(file, self.input_size, self.output_size, self.images_norm)

            sample = sample_[np.random.randint(len(input_))].reshape([-1, 33, 33, 3])
            input = input_[np.random.randint(len(input_))].reshape([-1, 33, 33, 3])
            loss = self.sess.run([self.g_loss], feed_dict={self.input_target: sample,
                                                           self.input_source: input})

            loss_dic[str(file)] = loss

        average_loss = np.average([value for value in loss_dic.values()])

        print('epoch{}:average_loss:{:.4f}'.format(epoch, average_loss))

        return loss_dic, average_loss


    def sample(self,epoch):
        files = glob(os.path.join(self.config.dataset_dir, 'val', self.config.val_set, '*.*'))
        # for file in files:
        file = files[0]
        h_, w_, input_, sample_ = get_sample_image(file, self.input_size, self.output_size, self.images_norm,self.config.scale)
        origin_ = sample_
        sample_images, psnr, input_source = self.sess.run([self.fake, self.psnr, self.input_source],
                                                          feed_dict={self.input_target:sample_, self.input_source:input_})

        sample_ = save_images(sample_images, [h_,w_], './{}/{}_sample_{}.png'.format(self.config.sample_dir, self.config.val_set,epoch),
                    self.images_norm)
        source_ = save_images(input_source, [h_,w_], './{}/{}_input_{}.png'.format(self.config.sample_dir, self.config.val_set,epoch),
                    self.images_norm)
        origin_ = save_images(origin_, [h_,w_], './{}/{}_origin_{}.png'.format(self.config.sample_dir, self.config.val_set,epoch),
                    self.images_norm)
        sample_ = (sample_ - 127.5 )/ 127.5
        source_ = (source_ - 127.5) / 127.5
        origin_ = (origin_ - 127.5 )/ 127.5
        psnr1 = self.PSNR_whole(sample_, origin_)
        psnr2 = self.PSNR_whole(source_, origin_)
        print("the source psnr is {}, the sample psnr is {}".format(psnr2, psnr1))

        print('epoch{}:the whole psnr is :{:.4f}'.format(epoch, psnr1 - psnr2))

    def PSNR_whole(self, real,fake):
        mse = np.mean(np.square(127.5 * (real - fake)), axis=(-3, -2, -1))
        psnr = np.mean(10 * (np.log(255 * 255 / np.sqrt(mse)) / np.log(10)))
        return psnr

    def test(self):
        print('testing')
        bool_check, _ = self.load_model(self.config.checkpoint_dir, is_test=True)
        if bool_check:
            print('[!!!] load model successfully')
        else:
            print('[***] fail to load model')

        file_name = self.config.file_name
        file = glob(os.path.join(self.config.dataset_dir, 'test', self.config.test_set, file_name))
        file = file[0]
        # for file in files:
        h_, w_, input_, sample_ = get_sample_image(file, self.input_size, self.output_size, self.images_norm, self.config.scale)

        sample_images, psnr, input_source = self.sess.run([self.fake, self.psnr, self.input_source],
                                                          feed_dict={self.input_target:sample_, self.input_source:input_})

        save_images(sample_images, [h_,w_], './{}/{}_sample.png'.format(self.config.test_dir, file_name),
                    self.images_norm)
        save_images(input_source, [h_,w_], './{}/{}_input.png'.format(self.config.test_dir, file_name),
                    self.images_norm)

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.config.dataset_name,
            self.batch_size)

    def save_model(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.config.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load_model(self, checkpoint_dir, is_test = False):
        import re
        print(" [*] Reading checkpoints...")

        checkpoint_dir = os.path.join(checkpoint_dir, self.config.model_dir, self.model_name)

        if is_test is False:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
                print(" [*] Success to read {}".format(ckpt_name))
                return True, counter
            else:
                print(" [*] Failed to find a checkpoint")
            return False, 0

        else:
            ckpt_name = self.config.load_model
            try:
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                print(" [*] Success to read {}".format(ckpt_name))
            except:
                print(" [*] Failed to find a checkpoint")
                assert 0
            return True, 0


if __name__=='__main__':
    srcnn = SRCNN(None)
    # a = tf.random_normal([8,64,64,3])
    # out = srcnn.generator(a)
    # out,_ = srcnn.discriminator(a)
    # print(out)