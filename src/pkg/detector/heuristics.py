import os
import sys
import numpy as np
import copy
import open3d as o3d
from enum import Enum
from ..utils.rotation_utils import *
from abc import abstractmethod

# Class dictionary for object detection & segmentation
class_dict = {'person':0, 'bicycle':1, 'car':2, 'motorcycle':3, 'airplane':4, 'bus':5, 'train':6,
              'truck':7, 'boat':8, 'traffic light':9, 'fire hydrant':10, 'stop sign':11, 'parking meter':12,
              'bench':13, 'bird':14, 'cat':15, 'dog':16, 'horse':17, 'sheep':18, 'cow':19, 'elephant':20,
              'bear':21, 'zebra':22, 'giraffe':23, 'backpack':24, 'umbrella':25, 'handbag':26, 'tie':27,
              'suitcase':28, 'frisbee':29, 'skis':30, 'snowboard':31, 'sports ball':32, 'kinte':33,
              'baseball bat':34, 'baseball glove':35, 'skateboard':36, 'surfboard':37, 'tennis racket':38,
              'bottle':39, 'wine glass':40, 'cup':41, 'fork':42, 'knife':43, 'spoon':44, 'bowl':45,
              'banana':46, 'apple':47, 'sandwich':48, 'orange':49, 'broccoli':50, 'carrot':51, 'hot dog':52,
              'pizza':53, 'donut':54, 'cake':55, 'chair':56, 'couch':57, 'potted plant':58, 'bed':59,
              'dining table':60, 'toilet':61, 'tv':62, 'laptop':63, 'mouse':64, 'remote':65, 'keyboard':66,
              'cell phone':67, 'microwave':68, 'oven':69, 'toaster':70, 'sink':71, 'refrigerator':72, 'book':73,
              'clock':74, 'vase':75, 'scissors':76, 'teddy bear':77, 'hair drier':78, 'toothbrush':79}


##
#
def cut_pcd(name, pcd_np):
    if name == "couch":
        # x, y cutting
        pcd_cut = pcd_np[np.where(np.abs(pcd_np)[:,0]<0.35)[0]]
        pcd_cut = pcd_cut[np.where(np.abs(pcd_cut)[:,1] < 0.25)[0]]
    return pcd_cut


def load_model(name):
    if name == "couch":
        plane = o3d.geometry.TriangleMesh.create_box(width=0.7, height=0.5, depth=0.001)
        plane.translate((-0.3, -0.15, 0))
    return plane


class ObjectInfo:
    ##
    # @param name   name of object
    # @param dims   geometry dimensions
    # @param color  geometry visualization color
    # @param Toff   offset of coordinate frame
    def __init__(self, name, dims=None, color=None, Toff=None,
                 scale=[1e-3,1e-3,1e-3], url=None):
        if color is None:
            color = (0.6,0.6,0.6,1)
        self.name = name
        self.dims, self.color = dims, color
        self.Toff = Toff
        self.scale = scale
        self.url = url
        self.detection_network = True

        if name in class_dict.keys():
            # object which is able to detect through Mask RCNN
            self.detection_network = True
        else:
            # object which is not able to detect through Mask RCNN
            self.detection_network = False


##
# @class InitializeRule
# @brief initialization rule when point cloud is given
class InitializeRule:
    @abstractmethod
    def get_initial(self, pcd, R=None, offset=None):
        raise(NotImplementedError("get_initial not implemented"))

##
# @class OffsetRule
# @brief give offset from center of point cloud in point cloud coordinates (orientation)
class OffsetOnModelCoord(InitializeRule):
    def __init__(self, name, R=None, offset=None, use_median=False):
        self.name = name
        self.R = np.identity(3) if R is None else R
        self.offset = np.zeros(3) if offset is None else offset
        self.use_median = use_median

    def get_initial(self, pcd, R=None, offset=None):
        if R is None:
            R = self.R
        if offset is None:
            offset = self.offset
            if self.use_median:
                center_p = np.median(pcd.points, axis=0)
            else:
                center_p = pcd.get_center()
        return SE3(R, center_p + offset)


class MaskBox:
    def __init__(self, Toff, dims, include):
        self.Toff, self.dims, self.include = Toff, dims, include

    def get_tf(self, Tparent):
        return np.matmul(Tparent, self.Toff)

##
# @class UpdateRuleFun
# @brief base class for mask box rule updater function
class UpdateRuleFun:
    def __init__(self, mbr):
        self.mbr = mbr

    ##
    # @param micp_target on camera coords
    # @param micp_parent on camera coords
    def __call__(self, micp_target, micp_parent, Tc=None):
        pass

##
# @class MaskRule
# @brief class interface for point cloud masking
class MaskRule:
    def __init__(self):
        # @brief UpdateRuleFun
        self.update_rule = UpdateRuleFun(self)

    @abstractmethod
    def apply_rule(self, pcd_in, objectPose_dict):
        raise(NotImplementedError("apply_rule should be implemented"))


##
# @class MaskBoxRule
# @brief Box masking rule class
class MaskBoxRule(MaskRule):
    def __init__(self, target, parent, merge_rule=np.all):
        self.target = target
        self.parent = parent
        self.box_list = []
        self.merge_rule = merge_rule
        # @brief UpdateRuleFun
        self.update_rule = UpdateRuleFun(self)

    def add_box(self, mbox):
        self.box_list.append(mbox)
        return self

    def box_clear(self):
        self.box_list = []

    ##
    # @param objectPose_dict in cam coords
    def apply_rule(self, pcd_in, objectPose_dict):
        pcd_dict = {}
        for oname, To in sorted(objectPose_dict.items()):
            pcd = copy.deepcopy(pcd_in)
            points = np.asarray(pcd.points) # in cam coords
            points4d = np.pad(points, ((0, 0), (0, 1)), 'constant', constant_values=1)
            mask_list = []
            for mbox in self.box_list:
                if self.parent == oname:
                    T_bx = mbox.get_tf(To)  # box tf in cam coords
                    T_xb = SE3_inv(T_bx)    # cam tf in box coords
                    abs_cuts = np.divide(mbox.dims, 2)
                    points_x = np.matmul(points4d, T_xb.transpose())[:, :3] # points in box coords
                    if mbox.include:
                        mask = np.all(np.abs(points_x) < abs_cuts, axis=-1) # check inside box
                    else:
                        mask = np.any(np.abs(points_x) > abs_cuts, axis=-1) # check outside box
                    mask_list.append(mask)
            idc = np.where(self.merge_rule(mask_list, axis=0))[0]
            pcd.points = o3d.utility.Vector3dVector(points[idc])
            pcd_dict[oname.replace(self.parent, self.target)] = pcd
        return pcd_dict