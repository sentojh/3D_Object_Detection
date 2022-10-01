import os
import pickle
import dill

dill._dill._reverse_typemap['ObjectType'] = object
dill._dill._reverse_typemap[b'ListType'] = list

import time
import numpy as np
import collections

import datetime
def get_now():
    return str(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))


def try_mkdir(path):
    try: os.mkdir(path)
    except: pass

from enum import Enum

##
# @class TextColors
# @brief color codes for terminal. use println to simply print colored message
class TextColors(Enum):
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def println(self, msg):
        print(self.value + str(msg) + self.ENDC.value)

##
# @class    Singleton
# @brief    Template to make a singleton class.
# @remark   Inherit this class to make a class a singleton.
#           Do not call the class constructor directly, but call <class name>.instance() to get singleton instance.
class Singleton:
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls.__instance

##
# @class    GlobalTimer
# @brief    A singleton timer to record timings anywhere in the code.
# @remark   Call GlobalTimer.instance() to get the singleton timer.
#           To see the recorded times, just print the timer: print(global_timer)
# @param    scale       scale of the timer compared to a second. For ms timer, 1000
# @param    timeunit    name of time unit for printing the log
# @param    stack   default value for "stack" in toc
class GlobalTimer(Singleton):
    def __init__(self, scale=1000, timeunit='ms', stack=False):
        self.reset(scale, timeunit, stack)

    ##
    # @brief    reset the timer.
    # @param    scale       scale of the timer compared to a second. For ms timer, 1000
    # @param    timeunit    name of time unit for printing the log
    # @param    stack   default value for "stack" in toc
    def reset(self, scale=1000, timeunit='ms', stack=False):
        self.stack = stack
        self.scale = scale
        self.timeunit = timeunit
        self.name_list = []
        self.ts_dict = {}
        self.time_dict = collections.defaultdict(lambda: 0)
        self.min_time_dict = collections.defaultdict(lambda: 1e10)
        self.max_time_dict = collections.defaultdict(lambda: 0)
        self.count_dict = collections.defaultdict(lambda: 0)
        self.timelist_dict = collections.defaultdict(list)
        self.switch(True)

    ##
    # @brief    switch for recording time. switch-off to prevent time recording for optimal performance
    def switch(self, onoff):
        self.__on = onoff

    ##
    # @brief    mark starting point of time record
    # @param    name    name of the section to record time.
    def tic(self, name):
        if self.__on:
            if name not in self.name_list:
                self.name_list.append(name)
            self.ts_dict[name] = time.time()

    ##
    # @brief    record the time passed from last call of tic with same name
    # @param    name    name of the section to record time
    # @param    stack   to stack each time duration to timelist_dict, set this value to True,
    #                   don't set this value to use default setting
    def toc(self, name, stack=None):
        if self.__on:
            dt = (time.time() - self.ts_dict[name]) * self.scale
            self.time_dict[name] = self.time_dict[name] + dt
            self.min_time_dict[name] = min(self.min_time_dict[name], dt)
            self.max_time_dict[name] = max(self.max_time_dict[name], dt)
            self.count_dict[name] = self.count_dict[name] + 1
            if stack or (stack is None and self.stack):
                self.timelist_dict[name].append(dt)
            return dt

    ##
    # @brief    get current time and estimated time arrival
    # @param    name    name of the section to record time
    # @param    current current index recommanded to start from 1
    # @param    end     last index
    # @return   (current time, eta)
    def eta(self, name, current, end):
        dt = self.toc(name, stack=False)
        return dt, (dt / current * end if current != 0 else 0)

    ##
    # @brief    record and start next timer in a line.
    def toctic(self, name_toc, name_tic, stack=None):
        dt = self.toc(name_toc, stack=stack)
        self.tic(name_tic)
        return dt

    ##
    # @brief you can just print the timer instance to see the record
    def __str__(self):
        strout = ""
        names = self.name_list
        for name in names:
            strout += "{name}: \t{tot_T} {timeunit}/{tot_C} = {per_T} {timeunit} ({minT}/{maxT})\n".format(
                name=name, tot_T=np.round(np.sum(self.time_dict[name])), tot_C=self.count_dict[name],
                per_T=np.round(np.sum(self.time_dict[name]) / self.count_dict[name], 3),
                timeunit=self.timeunit, minT=round(self.min_time_dict[name], 3), maxT=round(self.max_time_dict[name], 3)
            )
        return strout

    ##
    # @brief use "with timer:" to easily record duration of a code block
    def block(self, key, stack=None):
        return BlockTimer(self, key,stack=stack)

    def __enter__(self):
        self.tic("block")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc("block")

##
# @class    BlockTimer
# @brief    Wrapper class to record timing of a code block.
class BlockTimer:
    def __init__(self, gtimer, key, stack=None):
        self.gtimer, self.key, self.stack = gtimer, key, stack

    def __enter__(self):
        self.gtimer.tic(self.key)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gtimer.toc(self.key, stack=self.stack)


import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(filename, data):
    with open(filename, "w") as json_file:
        json.dump(data, json_file, cls=NumpyEncoder,indent=2)
    
def load_json(filename):
    with open(filename, "r") as st_json:
        st_python = json.load(st_json)
    return st_python

def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data


def load_scene_data(CONVERTED_PATH, DATASET, WORLD, SCENE, ACTION, joint_num, get_deviation=False):
    N_vtx_box = 3 * 8
    N_mask_box = 1
    N_joint_box = joint_num
    N_label_box = N_vtx_box + N_mask_box + N_joint_box
    N_vtx_cyl = 3 * 2 + 1
    N_mask_cyl = 1
    N_joint_cyl = joint_num
    N_label_cyl = N_vtx_cyl + N_mask_cyl + N_joint_cyl
    N_vtx_init = 3 * 8
    N_mask_init = 1
    N_joint_init = joint_num
    N_label_init = N_vtx_init + N_mask_init + N_joint_init
    N_vtx_goal = 3 * 8
    N_mask_goal = 1
    N_joint_goal = joint_num
    N_label_goal = N_vtx_goal + N_mask_goal + N_joint_goal
    N_joint_label = 6 * joint_num
    N_cell_label = N_label_box + N_label_cyl + N_label_init + N_label_goal + N_joint_label
    N_BEGIN_CYL = N_vtx_box + N_mask_box + N_joint_box
    N_BEGIN_INIT = N_BEGIN_CYL + N_vtx_cyl + N_mask_cyl + N_joint_cyl
    N_BEGIN_GOAL = N_BEGIN_INIT + N_vtx_init + N_mask_init + N_joint_init

    # print("load: {}".format((CONVERTED_PATH, DATASET, WORLD, SCENE)))
    scene_pickle = load_pickle(os.path.join(CONVERTED_PATH, DATASET, WORLD, SCENE, "scene.pkl"))
    scene_data = scene_pickle[b'scene_data']
    ctem_names = scene_pickle[b'ctem_names']
    ctem_cells = scene_pickle[b'ctem_cells']

    act_dat = load_pickle(os.path.join(CONVERTED_PATH, DATASET, WORLD, SCENE, ACTION))
    init_box_dat = act_dat[b'init_box_dat']
    goal_box_dat = act_dat[b'goal_box_dat']
    ctem_dat_list = act_dat[b'ctem_dat_list']
    skey = int(act_dat[b'skey'])
    success = act_dat[b'success']
    ### put init, goal item data
    cell, verts, chain = init_box_dat
    scene_data[cell[0], cell[1], cell[2], N_BEGIN_INIT:N_BEGIN_INIT + N_vtx_init] = verts
    scene_data[cell[0], cell[1], cell[2], N_BEGIN_INIT + N_vtx_init:N_BEGIN_INIT + N_vtx_init + N_mask_init] = 1
    scene_data[cell[0], cell[1], cell[2],
    N_BEGIN_INIT + N_vtx_init + N_mask_init:N_BEGIN_INIT + N_vtx_init + N_mask_init + N_joint_init] = chain
    cell_init = cell

    cell, verts, chain = goal_box_dat
    scene_data[cell[0], cell[1], cell[2], N_BEGIN_GOAL:N_BEGIN_GOAL + N_vtx_goal] = verts
    scene_data[cell[0], cell[1], cell[2], N_BEGIN_GOAL + N_vtx_goal:N_BEGIN_GOAL + N_vtx_goal + N_mask_goal] = 1
    scene_data[cell[0], cell[1], cell[2],
    N_BEGIN_GOAL + N_vtx_goal + N_mask_goal:N_BEGIN_GOAL + N_vtx_goal + N_mask_goal + N_joint_goal] = chain
    cell_goal = cell

    ### add/replace collilsion object
    for cname, ctype, cell, verts, chain in ctem_dat_list:
        if ctype == b'BOX':
            N_BEGIN_REP, N_vtx, N_mask, N_joint = 0, N_vtx_box, N_mask_box, N_joint_box
        elif ctype == b'CYLINDER':
            N_BEGIN_REP, N_vtx, N_mask, N_joint = N_BEGIN_CYL, N_vtx_cyl, N_mask_cyl, N_joint_cyl
        else:
            raise (RuntimeError("Non considered shape key"))
        scene_data[cell[0], cell[1], cell[2], N_BEGIN_REP:N_BEGIN_REP + N_vtx] = verts
        scene_data[cell[0], cell[1], cell[2], N_BEGIN_REP + N_vtx:N_BEGIN_REP + N_vtx + N_mask] = 1
        scene_data[cell[0], cell[1], cell[2],
        N_BEGIN_REP + N_vtx + N_mask:N_BEGIN_REP + N_vtx + N_mask + N_joint] = chain
    if get_deviation:
        return scene_data, success, skey, cell_init, cell_goal
    else:
        return scene_data, success, skey

from math import *

def rad2deg(rads):
    return rads/np.pi*180
        
def deg2rad(degs):
    return degs/180*np.pi

def Rot_axis( axis, q ):
    '''
    make rotation matrix along axis
    '''
    if axis==1:
        R = np.asarray([[1,0,0],
                        [0,cos(q),-sin(q)],
                        [0,sin(q),cos(q)]])
    if axis==2:
        R = np.asarray([[cos(q),0,sin(q)],
                        [0,1,0],
                        [-sin(q),0,cos(q)]])
    if axis==3:
        R = np.asarray([[cos(q),-sin(q),0],
                        [sin(q),cos(q),0],
                        [0,0,1]])
    return R

def Rot_axis_series(axis_list, rad_list):
    '''
    zyx rotation matrix - caution: axis order: z,y,x
    '''
    R = Rot_axis(axis_list[0], rad_list[0])
    for ax_i, rad_i in zip(axis_list[1:], rad_list[1:]):
        R = np.matmul(R, Rot_axis(ax_i,rad_i))
    return R

def Rot_zyx(zr,yr,xr):
    '''
    zyx rotation matrix - caution: axis order: z,y,x
    '''
    return Rot_axis_series([3,2,1], [zr,yr,xr])

def Rot_zxz(zr1,xr2,zr3):
    '''
    zxz rotation matrix - caution: axis order: z,x,z
    '''
    return Rot_axis_series([3,1,3], [zr1,xr2,zr3])

def Rot2zyx(R):
    '''
    rotation matrix to zyx angles - caution: axis order: z,y,x
    '''
    sy = sqrt(R[0,0]**2 + R[1,0]**2)

    if sy > 0.000001:
        x = atan2(R[2,1] , R[2,2])
        y = atan2(-R[2,0], sy)
        z = atan2(R[1,0], R[0,0])
    else:
        x = atan2(-R[1,2], R[1,1])
        y = atan2(-R[2,0], sy)
        z = 0
    return np.asarray([z,y,x])

def Rot2zxz(R):
    '''
    rotatio matrix to zyx angles - caution: axis order: z,y,x
    '''
    sy = sqrt(R[0,2]**2 + R[1,2]**2)

    if sy > 0.000001:
        z1 = atan2(R[0,2] , -R[1,2])
        x2 = atan2(sy,R[2,2])
        z3 = atan2(R[2,0], R[2,1])
    else:
        z1 = 0
        x2 = atan2(sy,R[2,2])
        z3 = atan2(-R[0,1], R[0,0])
    return np.asarray([z1,x2,z3])

def SE3(R,P):
    T = np.identity(4,dtype='float32')
    T[0:3,0:3]=R
    T[0:3,3]=P
    return T

def SE3_inv(T):
    R=T[0:3,0:3].transpose()
    P=-np.matmul(R,T[0:3,3])
    return (SE3(R,P))

##
# @brief convert cylindrical coordinate to cartesian coordinate
# @param radius x-y plane radius
# @param theta angle from x-axis, along z-axis
# @param height height from x-y plane
def cyl2cart(radius, theta, height):
    return np.cos(theta)*radius, np.sin(theta)*radius, height

##
# @brief convert cartesian coordinate to cylindrical coordinate
# @return radius x-y plane radius
# @return theta angle from x-axis, along z-axis
# @return height height from x-y plane
def cart2cyl(x, y, z):
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    height = z
    return r, theta, height


def sigmoid(x):
    return 1 / (1 +np.exp(-x))

##
# @brief    print confusion matrix
# @remark   rows: ground truth, cols: prediction
def print_confusion_mat(GT, Res):
    TP = np.sum(np.logical_and(GT, Res))
    FN = np.sum(np.logical_and(GT, np.logical_not(Res)))
    FP = np.sum(np.logical_and(np.logical_not(GT), Res))
    TN = np.sum(np.logical_and(np.logical_not(GT), np.logical_not(Res)))
    N = TP + FN + FP + TN
    print("\t PP \t \t PN \t \t {}".format(N))
    print("GP \t {} \t \t {} \t \t {:.2%}".format(TP, FN, float(TP) / (TP + FN)))
    print("GN \t {} \t \t {} \t {:.2%}".format(FP, TN, float(TN) / (FP + TN)))
    print(
        "AL \t {:.2%} \t {:.2%} \t {:.2%}".format(float(TP) / (TP + FP), float(TN) / (TN + FN), float(TP + TN) / N))