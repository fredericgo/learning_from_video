from __future__ import print_function
import numpy as np
import math
import cv2
import os, sys
root_path = os.path.realpath(__file__).split('/evaluate/multi_pose_net_estimator.py')[0]
sys.path.append(root_path)
import torch
import torch.nn as nn
from lib.utils.log import logger
import network.net_utils as net_utils
from network.posenet import poseNet
from network.joint_utils import get_joint_list, plot_result

from datasets.coco_data.preprocessing import resnet_preprocess
from datasets.coco_data.prn_gaussian import gaussian, crop

class Params(object):
    trunk = 'resnet101'  # select the model
    coeff = 2
    in_thres = 0.21

    testdata_dir = './demo/test_images/'
    testresult_dir = './demo/output/'
    testresult_write_image = False  # write image results or not
    testresult_write_json = False  # write json results or not
    gpus = [0]
    ckpt = './demo/models/ckpt_baseline_resnet101.h5'  # checkpoint file to load, no need to change this
    coco_root = 'coco_root/'
    coco_result_filename = './extra/multipose_coco2017_results.json'

    # # required params
    inp_size = 480  # input size 480*480
    exp_name = 'multipose101'
    subnet_name = 'keypoint_subnet'
    batch_size = 32
    print_freq = 20


class MultiPoseNetEstimator(object):
    def __init__(self, params):
        assert isinstance(params, Params)
        self.params = params
        if self.params.backbone == 'resnet101':
            model = poseNet(101)
        elif self.params.backbone == 'resnet50':
            model = poseNet(50)

        for name, module in model.named_children():
            for para in module.parameters():
                para.requires_grad = False

        # load model
        self.model = model
        ckpt = self.params.ckpt

        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        self.model = nn.DataParallel(self.model, device_ids=self.params.gpus)
        self.model = self.model.cuda(device=self.params.gpus[0])
        self.model.eval()
        self.model.module.freeze_bn()

    def infer(self, img):
        shape_dst = np.max(img.shape)
        scale = float(shape_dst) / self.params.inp_size
        pad_size = np.abs(img.shape[1] - img.shape[0])

        img_resized = np.pad(img, ([0, pad_size], [0, pad_size], [0, 0]), 'constant')[:shape_dst, :shape_dst]
        img_resized = cv2.resize(img_resized, (self.params.inp_size, self.params.inp_size))
        img_input = resnet_preprocess(img_resized)
        img_input = torch.from_numpy(np.expand_dims(img_input, 0))

        with torch.no_grad():
            img_input = img_input.cuda(device=self.params.gpus[0])

        heatmaps, [scores, classification, transformed_anchors] = self.model([img_input, self.params.subnet_name])
        heatmaps = heatmaps.cpu().detach().numpy()
        heatmaps = np.squeeze(heatmaps, 0)
        heatmaps = np.transpose(heatmaps, (1, 2, 0))
        heatmap_max = np.max(heatmaps[:, :, :18], 2)

        # segment_map = heatmaps[:, :, 17]
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        joint_list = get_joint_list(img_resized, param, heatmaps[:, :, :18], scale)
        joint_list = joint_list.tolist()
        del img_resized

        joints = []
        for joint in joint_list:
            if int(joint[-1]) != 1:
                joint[-1] = max(0, int(joint[-1]) - 1)
                joints.append(joint)
        joint_list = joints

        # bounding box from retinanet
        scores = scores.cpu().detach().numpy()
        classification = classification.cpu().detach().numpy()
        transformed_anchors = transformed_anchors.cpu().detach().numpy()
        idxs = np.where(scores > 0.5)
        bboxs=[]
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]*scale
            if int(classification[idxs[0][j]]) == 0:  # class0=people
                bboxs.append(bbox.tolist())

        prn_result = self.prn_process(joint_list, bboxs, None)
        results = []
        for prn in prn_result:
            results.append(np.array(prn['keypoints']).reshape(-1, 3))
        return results

    def _load_ckpt(self, ckpt):
        _, _ = net_utils.load_net(ckpt, self.model, load_state_dict=True)

    def prn_process(self, kps, bbox_list, file_name, image_id=0):

        prn_result = []

        idx = 0
        ks = []
        for j in range(17):  # joint type
            t = []
            for k in kps:
                if k[-1] == j:  # joint type
                    x = k[0]
                    y = k[1]
                    v = 1  # k[2]
                    if v > 0:
                        t.append([x, y, 1, idx])
                        idx += 1
            ks.append(t)
        peaks = ks

        w = int(18 * self.params.coeff)
        h = int(28 * self.params.coeff)

        bboxes = []
        for bbox_item in bbox_list:
            bboxes.append([bbox_item[0], bbox_item[1], bbox_item[2]-bbox_item[0], bbox_item[3]-bbox_item[1]])

        if len(bboxes) == 0 or len(peaks) == 0:
            return prn_result

        weights_bbox = np.zeros((len(bboxes), h, w, 4, 17))

        for joint_id, peak in enumerate(peaks):  # joint_id: which joint
            for instance_id, instance in enumerate(peak):  # instance_id: which people
                p_x = instance[0]
                p_y = instance[1]
                for bbox_id, b in enumerate(bboxes):
                    is_inside = p_x > b[0] - b[2] * self.params.in_thres and \
                                p_y > b[1] - b[3] * self.params.in_thres and \
                                p_x < b[0] + b[2] * (1.0 + self.params.in_thres) and \
                                p_y < b[1] + b[3] * (1.0 + self.params.in_thres)
                    if is_inside:
                        x_scale = float(w) / math.ceil(b[2])
                        y_scale = float(h) / math.ceil(b[3])
                        x0 = int((p_x - b[0]) * x_scale)
                        y0 = int((p_y - b[1]) * y_scale)
                        if x0 >= w and y0 >= h:
                            x0 = w - 1
                            y0 = h - 1
                        elif x0 >= w:
                            x0 = w - 1
                        elif y0 >= h:
                            y0 = h - 1
                        elif x0 < 0 and y0 < 0:
                            x0 = 0
                            y0 = 0
                        elif x0 < 0:
                            x0 = 0
                        elif y0 < 0:
                            y0 = 0
                        p = 1e-9
                        weights_bbox[bbox_id, y0, x0, :, joint_id] = [1, instance[2], instance[3], p]
        old_weights_bbox = np.copy(weights_bbox)

        for j in range(weights_bbox.shape[0]):
            for t in range(17):
                weights_bbox[j, :, :, 0, t] = gaussian(weights_bbox[j, :, :, 0, t])

        output_bbox = []
        for j in range(weights_bbox.shape[0]):
            inp = weights_bbox[j, :, :, 0, :]
            input = torch.from_numpy(np.expand_dims(inp, axis=0)).cuda().float()
            output, _ = self.model([input, 'prn_subnet'])
            temp = np.reshape(output.data.cpu().numpy(), (56, 36, 17))
            output_bbox.append(temp)

        output_bbox = np.array(output_bbox)

        keypoints_score = []

        for t in range(17):
            indexes = np.argwhere(old_weights_bbox[:, :, :, 0, t] == 1)
            keypoint = []
            for i in indexes:
                cr = crop(output_bbox[i[0], :, :, t], (i[1], i[2]), N=15)
                score = np.sum(cr)

                kp_id = old_weights_bbox[i[0], i[1], i[2], 2, t]
                kp_score = old_weights_bbox[i[0], i[1], i[2], 1, t]
                p_score = old_weights_bbox[i[0], i[1], i[2], 3, t]  ## ??
                bbox_id = i[0]

                score = kp_score * score

                s = [kp_id, bbox_id, kp_score, score]

                keypoint.append(s)
            keypoints_score.append(keypoint)

        bbox_keypoints = np.zeros((weights_bbox.shape[0], 17, 3))
        bbox_ids = np.arange(len(bboxes)).tolist()

        # kp_id, bbox_id, kp_score, my_score
        for i in range(17):
            joint_keypoints = keypoints_score[i]
            if len(joint_keypoints) > 0:  # if have output result in one type keypoint

                kp_ids = list(set([x[0] for x in joint_keypoints]))

                table = np.zeros((len(bbox_ids), len(kp_ids), 4))

                for b_id, bbox in enumerate(bbox_ids):
                    for k_id, kp in enumerate(kp_ids):
                        own = [x for x in joint_keypoints if x[0] == kp and x[1] == bbox]

                        if len(own) > 0:
                            table[bbox, k_id] = own[0]
                        else:
                            table[bbox, k_id] = [0] * 4

                for b_id, bbox in enumerate(bbox_ids):  # all bbx, from 0 to ...

                    row = np.argsort(-table[bbox, :, 3])  # in bbx(bbox), sort from big to small, keypoint score

                    if table[bbox, row[0], 3] > 0:  # score
                        for r in row:  # all keypoints
                            if table[bbox, r, 3] > 0:
                                column = np.argsort(
                                    -table[:, r, 3])  # sort all keypoints r, from big to small, bbx score

                                if bbox == column[0]:  # best bbx. best keypoint
                                    bbox_keypoints[bbox, i, :] = [x[:3] for x in peaks[i] if x[3] == table[bbox, r, 0]][
                                        0]
                                    break
                                else:  # for bbx column[0], the worst keypoint is row2[0],
                                    row2 = np.argsort(table[column[0], :, 3])
                                    if row2[0] == r:
                                        bbox_keypoints[bbox, i, :] = \
                                            [x[:3] for x in peaks[i] if x[3] == table[bbox, r, 0]][0]
                                        break
            else:  # len(joint_keypoints) == 0:
                for j in range(weights_bbox.shape[0]):
                    b = bboxes[j]
                    x_scale = float(w) / math.ceil(b[2])
                    y_scale = float(h) / math.ceil(b[3])

                    for t in range(17):
                        indexes = np.argwhere(old_weights_bbox[j, :, :, 0, t] == 1)
                        if len(indexes) == 0:
                            max_index = np.argwhere(output_bbox[j, :, :, t] == np.max(output_bbox[j, :, :, t]))
                            bbox_keypoints[j, t, :] = [max_index[0][1] / x_scale + b[0],
                                                       max_index[0][0] / y_scale + b[1], 0]

        my_keypoints = []

        for i in range(bbox_keypoints.shape[0]):
            k = np.zeros(51)
            k[0::3] = bbox_keypoints[i, :, 0]
            k[1::3] = bbox_keypoints[i, :, 1]
            k[2::3] = bbox_keypoints[i, :, 2]

            pose_score = 0
            count = 0
            for f in range(17):
                if bbox_keypoints[i, f, 0] != 0 and bbox_keypoints[i, f, 1] != 0:
                    count += 1
                pose_score += bbox_keypoints[i, f, 2]
            pose_score /= 17.0

            my_keypoints.append(k)

            image_data = {
                'image_id': image_id,
                'file_name': file_name,
                'category_id': 1,
                'bbox': bboxes[i],
                'score': pose_score,
                'keypoints': k.tolist()
            }
            prn_result.append(image_data)

        return prn_result
