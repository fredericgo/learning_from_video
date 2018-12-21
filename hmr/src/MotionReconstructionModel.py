""" Evaluates a trained model using placeholders. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from os.path import exists

from .tf_smpl import projection as proj_util
from .tf_smpl.batch_smpl import SMPL
from .models import get_encoder_fn_separate


class MotionReconstructionModel(object):
    def __init__(self, config, sess=None):
        """
        Args:
          config
        """
        self.config = config
        self.load_path = config.load_path

        # Config + path
        if not config.load_path:
            raise Exception(
                "[!] You need to specify `load_path` to load a pretrained model"
            )
        if not exists(config.load_path + '.index'):
            print('%s doesnt exist..' % config.load_path)
            import ipdb
            ipdb.set_trace()

        # Data
        self.batch_size = config.batch_size
        self.img_size = config.img_size

        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path

        input_size = (self.batch_size, self.img_size, self.img_size, 3)
        self.images_pl = tf.placeholder(tf.float32, shape=input_size)

        # Model Settings
        self.num_stage = config.num_stage
        self.model_type = config.model_type
        self.joint_type = config.joint_type
        # Camera
        self.num_cam = 3
        self.proj_fn = proj_util.batch_orth_proj_idrot

        self.num_theta = 72
        # Theta size: camera (3) + pose (24*3) + shape (10)
        self.total_params = self.num_cam + self.num_theta + 10

        self.smpl = SMPL(self.smpl_model_path, joint_type=self.joint_type)

        self.img_enc_fn, self.threed_enc_fn = \
            get_encoder_fn_separate(self.model_type)

        self.build_test_model_ief()

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        # Load data.
        self.saver = tf.train.Saver()
        self.prepare()

        # motion reconstruction model setting
        self.num_hidden = 2048

    def build_test_model_ief(self):
        # Load mean value
        self.mean_var = tf.Variable(tf.zeros((1, self.total_params)),
                                    name="mean_param", dtype=tf.float32)

        # Extract image features.
        self.img_feat, self.E_var = self.img_enc_fn(self.images_pl,
                                                    is_training=False,
                                                    reuse=False)
        # Start loop
        self.all_verts = []
        self.all_kps = []
        self.all_cams = []
        self.all_Js = []
        self.final_thetas = []
        theta_prev = tf.tile(self.mean_var, [self.batch_size, 1])
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            state = tf.concat([self.img_feat, theta_prev], 1)

            if i == 0:
                delta_theta, _ = self.threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=False)
            else:
                delta_theta, _ = self.threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=True)

            # Compute new theta
            theta_here = theta_prev + delta_theta
            # cam = N x 3, pose N x self.num_theta, shape: N x 10
            cams = theta_here[:, :self.num_cam]
            poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            shapes = theta_here[:, (self.num_cam + self.num_theta):]

            verts, Js, _, Jtrans = self.smpl(shapes, poses, get_skin=True)

            # Project to 2D!
            pred_kp = self.proj_fn(Js, cams, name='proj_2d_stage%d' % i)
            self.all_verts.append(verts)
            self.all_kps.append(pred_kp)
            self.all_cams.append(cams)
            self.all_Js.append(Js)
            # save each theta.
            self.final_thetas.append(theta_here)
            # Finally)update to end iteration.
            theta_prev = theta_here

    def morec_model(self, z0, x2d0):
        num_steps = z0.shape[0]
        z = tf.get_variable("Z",
                            shape=(num_steps, 2048),
                            initializer=tf.zeros_initializer())
        theta_prev = tf.tile(self.mean_var, [num_steps, 1])
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            state = tf.concat([z, theta_prev], 1)

            if i == 0:
                delta_theta, _ = self.threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=True)
            else:
                delta_theta, _ = self.threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=True)

            # Compute new theta
            theta_here = theta_prev + delta_theta
            # Finally)update to end iteration.
            theta_prev = theta_here

        # cam = N x 3, pose N x self.num_theta, shape: N x 10
        cams = theta_here[:, :self.num_cam]
        poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
        shapes = theta_here[:, (self.num_cam + self.num_theta):]
        verts, Js, _, _ = self.smpl(shapes, poses, get_skin=True)
        # Project to 2D!
        pred_kp = self.proj_fn(Js, cams, name='proj_2d_stage%d' % i)
        return verts, pred_kp, poses, cams, Js

    def prepare(self):
        print('Restoring checkpoint %s..' % self.load_path)
        self.saver.restore(self.sess, self.load_path)
        self.mean_value = self.sess.run(self.mean_var)

    def predict(self, images):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        """
        images = images
        img_size = images[0].shape[:2]
        # num_steps = len(images)
        results = self.initial_predict(images)
        x2d0 = results['joints']
        q3d0 = results['theta'][:, self.num_cam:(self.num_cam + self.num_theta)]
        z0 = results['hidden']
        verts, x2d, q3d, Rs, J3d = self.morec_model(z0, x2d0)

        l_2d = tf.reduce_mean(tf.abs(x2d - x2d0))
        l_3d = tf.reduce_mean(tf.abs(q3d - q3d0))
        l_sm = tf.reduce_mean(tf.squared_difference(J3d[1:], J3d[:-1]))
        l_cam = tf.reduce_mean(tf.squared_difference(Rs[1:], Rs[:-1]))
        loss = 10 * l_2d + 100 * l_3d + 25 * l_sm + 25 * l_cam

        optimizer = tf.train.AdamOptimizer()
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Z")
        train = optimizer.minimize(loss, var_list=train_vars)
        uninit_names = set(self.sess.run(tf.report_uninitialized_variables()))
        uninit_vars = [v for v in tf.global_variables()
                       if v.name.split(":")[0] in uninit_names]
        # train_writer = tf.summary.FileWriter( 'summary', self.sess.graph)
        init_op = tf.variables_initializer(uninit_vars)
        self.sess.run(init_op)

        # self.sess.run(tf.global_variables_initializer())

        for i in range(300):
            _, x_val, loss_value = self.sess.run((train, x2d, loss))
            print("step {}, loss = {}".format(i, loss_value))

        verts_p, j2d, q3d_pred, j3d_pred, cams = self.sess.run(
            [verts, x2d, q3d, J3d, Rs])
        # cams = results['cams']
        j2d = ((j2d + 1) * 0.5) * img_size
        return verts_p, j2d, q3d_pred, j3d_pred, cams  # results['joints3d']

    def initial_predict(self, images):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        """
        n = images.shape[0]

        res = []
        b = np.zeros((self.batch_size, self.img_size, self.img_size, 3))
        for i in range(0, n, self.batch_size):
            nb = images[i:i + self.batch_size].shape[0]
            b[:nb] = images[i:i + self.batch_size]
            out = self.predict_dict(b)
            out = {k: v[:nb] for k, v in out.items()}
            res.append(out)

        results = {}
        results['joints'] = np.concatenate([v['joints'] for v in res])
        results['verts'] = np.concatenate([v['verts'] for v in res])
        results['cams'] = np.concatenate([v['cams'] for v in res])
        results['joints3d'] = np.concatenate([v['joints3d'] for v in res])
        results['theta'] = np.concatenate([v['theta'] for v in res])
        results['hidden'] = np.concatenate([v['hidden'] for v in res])

        return results

    def predict_dict(self, images):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        Runs the model with images.
        """
        feed_dict = {
            self.images_pl: images,
            # self.theta0_pl: self.mean_var,
        }
        fetch_dict = {
            'joints': self.all_kps[-1],
            'verts': self.all_verts[-1],
            'cams': self.all_cams[-1],
            'joints3d': self.all_Js[-1],
            'theta': self.final_thetas[-1],
            'hidden': self.img_feat,
        }

        results = self.sess.run(fetch_dict, feed_dict)

        # Return joints in original image space.
        joints = results['joints']
        results['joints'] = ((joints + 1) * 0.5) * self.img_size

        return results
