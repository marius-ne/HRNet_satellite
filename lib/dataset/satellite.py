# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
#   Modified for ESA Satellite Pose Estimation Tasks by Marius Neuhalfen, 28/11/2024
#   Modifications highlighted by comments beginning with ALL CAPS.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


class SatelliteDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = cfg.MODEL.NUM_JOINTS      # MODIFIED: Taken directly from cfg file instead of declaring
                                                    # seperately.
        self.flip_pairs = cfg.DATASET.FLIP_PAIRS    # MODIFIED: We take it from the cfg (to enable multiple datasets with this same
                                                    # python file.)
                                                    # This is apparently for semantic consistency of the chirality during the flip.
                                                    # So for example after flipping the image (for data augmentation) the left hand
                                                    # ends up in the position of the right hand. Then you change the labels so that
                                                    # the network doesn't mix up the sides. For non-symmetric parts like the protruding
                                                    # elements on ENVISAT, you don't flip the labels so you don't add them to the list.
                                                    # Thus we just add here the symmetric parts that correspond along the z axis.
                                                    # Be careful, here the keypoints are 0-indexes, so you have to subtract one for each
                                                    # of the Matlab indeces.
                                                                                
        self.parent_ids = None          # MODIFIED:
                                        # We don't need kinematic joint relationhips, the body is assumed rigid.
                                        # If there is a protruding part that is not rigid, for example a solar
                                        # panel, adjust the parent_ids accordingly. Each keypoint index has its 
                                        # corresponding parent that it is attached to at the same index in the 
                                        # parent_ids. In any case it is unclear for what this variable is used 
                                        # because we could only find the definition here, not a call by another
                                        # function. Maybe just for visualization?
                                        # COCO dataset also uses None here.

        self.upper_body_ids = (2, 3, 12, 10, 11, 15, 6, 7, 13) # NOTE: We disabled the half_body_transform in the cfg
                                                                # so we don't need to modify these. NOW ENABLED!
        self.lower_body_ids = (0, 1, 4, 5, 14, 8, 9)

        self.db = self._get_db()
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(
            self.root, 'annot', self.image_set+'.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=float)
            s = np.array([a['scale'], a['scale']], dtype=float)

            # Adjust center/scale slightly to avoid cropping limbs
            # MODIFIED: our bounding boxes are already quite spacious and
            # they do not intersect the satellite body. So we remove this.
            #if c[0] != -1:
            #    c[1] = c[1] + 15 * s[1]
            #    s = s * 1.25

            # mpii uses matlab format, index is based 1,
            # we should first convert to 0-based index
            # NOTE: We use matlab image indeces too.
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        # Values for computing the PCKh from https://arxiv.org/pdf/1902.09212
        # threshold is the closeness required for a keypoint to be considered 
        # correct. The SC_BIAS is the empirical observation that the real headsize
        # corresponds to 60% of the bounding box headsize. We turn this bias to 0.03.
        # See below for more explanation.
        threshold = 0.03

        # ISSUE, TO BE RESOLVED
        # We don't yet have the gt_mat file.
        # And we also don't have the individual bodyparts.
        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'annot',
                               'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
        gt_dict = loadmat(gt_file)
        dataset_joints = cfg.DATASET.DATASET_JOINTS  # Names of keypoints in char
        jnt_visible = gt_dict['joint_visible']        # We modified this from joint_missing to joint_visible
        pos_gt_src = gt_dict['pos_gt_src']
        bbox_src = gt_dict['boundingBoxWidths_src']      # Changed from headbox to width of complete bounding box

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        # Ensure dataset_joints is a clean list of keypoint names
        if isinstance(dataset_joints, np.ndarray):
            if dataset_joints.ndim > 1:
                dataset_joints = dataset_joints.squeeze()  # Remove extra dimensions
            dataset_joints = dataset_joints.tolist()  # Convert to Python list
        
        dataset_joints = [str(name[0]) if isinstance(name, np.ndarray) else str(name) for name in dataset_joints]
            
        #print("dataset_joints:", dataset_joints)  # Should list keypoint names
        #print("len(dataset_joints):", len(dataset_joints))  # Should match the number of keypoints

        # Create a dictionary mapping keypoint names to their indices, dataset-agnostic!
        keypoint_indices = {
            name: dataset_joints.index(name) if name in dataset_joints else None
            for name in dataset_joints
        }
        #print("keypoint_indices:", keypoint_indices)

        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        
        #print("uv_err:", uv_err)
        
        # We modify the headsize metric. The size of one of the protruding elements when the bounding box is 600
        # pixels wide is around 15 pixels. That makes a ratio of 0.025, we are liberal and say 0.03. This means
        # that we consider a keypoint correct when it falls into 3% of the location of the ground truth.
        
        #headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        #headsizes = np.linalg.norm(headsizes, axis=0)
        #headsizes *= SC_BIAS
        #scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        #scaled_uv_err = np.divide(uv_err, scale)
        #scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        #jnt_count = np.sum(jnt_visible, axis=1)
        #less_than_threshold = np.multiply((scaled_uv_err <= threshold),
        #                                  jnt_visible)
        
        scale = bbox_src * 0.03   # 0.03 times the width of each bounding box
        #scale = np.expand_dims(scale, axis=0)  # Ensure scale has the correct shape
        
        # Normalize UV error by the scale
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)  # Apply visibility mask
        
        # Compute the number of valid joints and threshold checks
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= 1.0), jnt_visible)  # Error normalized by scale <= 1.0
        
        #print("jnt_count:", jnt_count)
        #print("less_than_threshold:", less_than_threshold)
        
        #print("uv_error shape:", uv_error.shape)  # Should be (16, 2, 2400)
        #print("uv_err shape:", uv_err.shape)  # Should be (16, 2400)
        #print("scale shape:", scale.shape)  # Should be (1, 2400)
        #print("scaled_uv_err shape:", scaled_uv_err.shape)  # Should be (16, 2400)
        #print("less_than_threshold shape:", less_than_threshold.shape)  # Should be (16, 2400)
        #print("jnt_count shape:", jnt_count.shape)  # Should be (16, 1)

        PCKh = np.divide(100.0 * np.sum(less_than_threshold, axis=1), jnt_count)
        
        #print("PCKh shape:", PCKh.shape)  # Should be (16,)

        # Testing sensibility of PCKh on the choice of threshold
        # We reduce max value from 0.5 to 0.1 because our threshold is very small
        rng = np.arange(0, 0.03+0.01, 0.01)
        pckAll = np.zeros((len(rng), len(dataset_joints)))

        #print("rng:", rng)
        #print("pckAll:", pckAll)
        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply((scaled_uv_err <= threshold), jnt_visible)
            #print("less_than_threshold:", less_than_threshold)
            pckAll[r, :] = np.divide(100.0 * np.sum(less_than_threshold, axis=1), jnt_count)


        # We don't need the masks and the reason behind them is unclear.
        # Apparently the shoulders are not good for evaluation.
        #PCKh = np.ma.array(PCKh, mask=False)
        #PCKh.mask[6:8] = True

        #jnt_count = np.ma.array(jnt_count, mask=False)
        #jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(float)

        #name_value = [
        #    ('Head', PCKh[head]),
        #    ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
        #    ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
        #    ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
        #    ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
        #    ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
        #    ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
        #    ('Mean', np.sum(PCKh * jnt_ratio)),
        #    ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        #]

        # Calculate metrics dynamically using the keypoint_indices dictionary
        # Individual keypoint PCKh without summing
        name_value = [(name, PCKh[index]) for name, index in keypoint_indices.items()]

        name_value = OrderedDict(name_value)
        
        # Add Mean PCKh
        name_value['Mean'] = np.sum(PCKh * jnt_ratio)
        name_value['Mean@0.01'] = np.sum(pckAll[1, :] * jnt_ratio) #pckAll [0.00,0.01,0.02,0.03(end exclusive))] -> 0.01 is second element, so index 1

        return name_value, name_value['Mean']
