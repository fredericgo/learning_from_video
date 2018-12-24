"""
Script to convert openpose output into bbox
"""
import json
import numpy as np


def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints']).reshape(-1, 3)
        kps.append(kp)
    return kps


def get_bbox_info(kps, vis_thr=0.2):
    """
    joints is 3 x 19. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    """
    coco_order = [0, 14, 13, 16, 15, 4, 1, 5, 2, 6, 3, 10, 7, 11, 8, 12, 9]
    # coco               posenet
    # 0 head                  0
    # 1 neck                 14
    # 2 right shoulder       13
    # 3 right elbow          16
    # 4 right hand           15
    # 5 left shoulder         4
    # 6 left elbow            1
    # 7 left hand             5
    # 8 Right hip             2
    # 9 Right knee            6
    # 10 Right foot           3
    # 11 Left hip            10
    # 12 Left knee            7
    # 13 Left foot           11
    # 14 Right eye            8
    # 15 Left eye            12
    # 16 Right Ear            9
    # 17 Left ear

    if isinstance(kps, basestring):
        kps = read_json(kps)
    # Pick the most confident detection.
    scores = [np.mean(kp[kp[:, 2] > vis_thr, 2]) for kp in kps]
    kp = kps[np.argmax(scores)]
    vis = kp[:, 2] > vis_thr
    vis_kp = kp[vis, :2]
    min_pt = np.min(vis_kp, axis=0)
    max_pt = np.max(vis_kp, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        print('bad!')
        import ipdb
        ipdb.set_trace()
    center = (min_pt + max_pt) / 2.
    scale = 150. / person_height

    return scale, center, min_pt, max_pt


def get_openpose_best_score(kps, vis_thr):
    scores = [np.mean(kp[kp[:, 2] > vis_thr, 2]) for kp in kps]
    kp = kps[np.argmax(scores)]
    return kp[:, 2]
