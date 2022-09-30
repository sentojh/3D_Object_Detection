'''
Created on 2019. 3. 15.

@author: JSK
'''
from scipy.spatial.transform import Rotation
import numpy as np
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

    if sy > 1e-10:
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

    if sy > 1e-10:
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

def SE3_R(T):
    return T[0:3,0:3]

def SE3_P(T):
    return T[0:3,3]

def SE3_mul_vec3(T,v):
    r=np.matmul(SE3_R(T),v)
    return np.add(r,SE3_P(T))

def average_SE3(Ts):
    nT = Ts.shape[0]
    Rref = Ts[0,:3,:3]
    dRlie_list = []
    for i in range(nT):
        Ri = Ts[i,:3,:3]
        dRi = np.matmul(Rref.transpose(),Ri)
        dRlie = np.real(scipy.linalg.logm(dRi,disp=False)[0])
        dRlie_list += [dRlie]
    dRlie_list = np.array(dRlie_list)
    dRlie_m = np.mean(dRlie_list,axis=0)
    R_m = np.matmul(Rref,scipy.linalg.expm(dRlie_m))
    P_m = np.mean(Ts[:,:3,3],axis=0)
    T_m=SE3(R_m,P_m)
    return T_m

def align_z(Two):
    Rwo = Two[0:3,0:3]
    Zwo=np.matmul(Rwo,[[0],[0],[1]])
    azim=np.arctan2(Zwo[1],Zwo[0])-np.deg2rad(90)
    altit=np.arctan2(np.linalg.norm(Zwo[:2]),Zwo[2])
    Rwo_=np.matmul(Rot_zxz(azim,altit,-azim),Rwo)
    Two_out = Two.copy()
    Two_out[0:3,0:3] = Rwo_
    return Two_out

##
# @brief fit camera detection to workplane considering viewpoint
# @param    Tco     object pose from camera
# @param    minz    local z-axis value of object's bottom point
# @param    Tcw     workplace cooridnate from camera view
def fit_floor(Tco, minz=0, Tcw=None ):
    if Tcw is None:
        Tbo = np.copy(Tco)
        Tbo[2,3] = -minz
        return Tbo
    Pco = Tco[0:3,3]
    Twc = np.linalg.inv(Tcw)
    Pco_wz = np.dot(Twc[2,0:3],Pco)
    if abs(Pco_wz)<0.000001:
        Pco_wz = 0.000001
    alpha = abs((-minz - Twc[2,3])/Pco_wz)
    Pco_ = Pco*alpha
    Tco_out = Tco.copy()
    Tco_out[0:3,3]=Pco_
    return Tco_out

def project_px(Tco, cam_K, points):
    vtx_cco = np.matmul(cam_K, np.matmul(Tco[:3, :3], points.transpose()) + Tco[:3, 3:4])
    vtx_px = (vtx_cco[:2, :] / vtx_cco[2:3, :])
    return vtx_px, vtx_cco[2, :]

def Rot_rpy(rpy):
    return np.transpose(Rot_axis_series([1,2,3],np.negative(rpy)))

def Rot2rpy(R):
    return np.asarray(list(reversed(Rot2zyx(R))))

def Rot2axis(R, axis):
    shift = 1-axis
    Rshift = np.zeros((3,3))
    for ax in range(3):
        Rshift[ax+shift,ax] = 1
    Rx = matmul_series(Rshift, R, Rshift.transpose())
    rot_x = Rot2zyx(Rx)[-1]
    return rot_x

##
# @return tuple(xyz, rpy(rad))
def T2xyzrpy(T):
    return T[:3,3].tolist(), Rot2rpy(T[:3,:3]).tolist()

##
# @return tuple(xyz, rotvec)
def T2xyzrvec(T, decimals=None):
    if decimals is None:
        return T[:3,3].tolist(), Rotation.from_dcm(T[:3,:3]).as_rotvec().tolist()
    else:
        return tuple(np.round(T[:3,3], decimals)), tuple(Rotation.from_dcm(T[:3,:3]).as_rotvec())

##
# @return tuple(xyz, quaternion)
def T2xyzquat(T, decimals=None):
    if decimals is None:
        return T[:3,3].tolist(), Rotation.from_dcm(T[:3,:3]).as_quat().tolist()
    else:
        return tuple(np.round(T[:3,3], decimals)), tuple(np.round(Rotation.from_dcm(T[:3,:3]).as_quat(), decimals))

##
# @param xyzrpy tuple(xyz, rpy(rad))
def T_xyzrpy(xyzrpy):
    return SE3(Rot_rpy(xyzrpy[1]), xyzrpy[0])

##
# @param xyzrpy tuple(xyz, quaternion)
def T_xyzquat(xyzquat):
    return SE3(Rotation.from_quat(xyzquat[1]).as_dcm(), xyzquat[0])

##
# @param xyzrpy tuple(xyz, quaternion)
def T_xyzrvec(xyzrvec):
    return SE3(Rotation.from_rotvec(xyzrvec[1]).as_dcm(), xyzrvec[0])

def matmul_series(*Tlist):
    T = Tlist[0]
    for T_i in Tlist[1:]:
        T = np.matmul(T, T_i)
    return T

Tx180 = np.identity(4, 'float32')
Tx180[1,1]=-1
Tx180[2,2]=-1

Ty180 = np.identity(4, 'float32')
Ty180[0,0]=-1
Ty180[2,2]=-1

##
# @brief rotation vector to rotate from vec1 to vec2
def calc_rotvec_vecs(vec1, vec2):
    cross_vec = np.cross(vec1, vec2)
    dot_val = np.dot(vec1, vec2)
    cross_abs = np.linalg.norm(cross_vec)
    if np.linalg.norm(cross_abs) < 1e-8:
        if len(vec1)==2:
            if dot_val>=0:
                rotvec = 0
            else:
                rotvec = np.pi
        elif len(vec1)==3:
            rotvec = np.zeros(3)
            if dot_val<0:
                rotvec[2] = np.pi
    else:
        cross_nm = cross_vec/cross_abs
        rotvec = cross_nm * np.arctan2(cross_abs, dot_val)
    return rotvec

def calc_zvec_R(zvec):
    v = np.arctan2(np.linalg.norm(zvec[:2]), zvec[2])
    w = np.arctan2(zvec[1], zvec[0])
    R = Rot_zyx(w,v,0)
    return R


##
# @brief convert cylindrical coordinate to cartesian coordinate
# @param radius x-y plane radius
# @param theta angle from x-axis, along z-axis
# @param height height from x-y plane
def cyl2cart(radius, theta, height):
    return np.cos(theta)*radius, np.sin(theta)*radius, height


##
# @brief convert horizontal coordinate to orientation matrix
# @param theta position vector angle from x-axis, along z-axis
# @param azimuth_loc angle from radial axis along z axis
# @param zenith angle from bottom zenith
def hori2mat(theta, azimuth_loc, zenith):
    return Rot_axis_series([3,2], [theta+azimuth_loc, np.pi-zenith])


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


##
# @brief convert orientation matrix to horizontal coordinate
# @param theta position vector angle from x-axis, along z-axis
# @param orientaion_mat orentation
# @return azimuth_loc angle from radial axis along z axis
# @return zenith angle from bottom zenith
def mat2hori(orientation_mat, theta=0):
    x,y,z = orientation_mat[:,2]
    azimuth = np.arctan2(y,x)
    azimuth_loc = (azimuth-theta + np.pi)%(2*np.pi)-np.pi
    zenith = np.arctan2(np.sqrt(x**2+y**2), -z)%np.pi
    return azimuth_loc, zenith

##
# @brief convert cartesian coordinate to spherical coordinate
# @return radius distance from origin
# @return psi   angle from z-axis
# @return theta angle from x-axis, along z-axis
def cart2spher(x, y, z):
    radius = np.sqrt(x**2+y**2+z**2)
    psi = np.arctan2(y,x)
    theta = np.arccos(z/radius)
    return radius, psi, theta


##
# @brief convert spherical coordinate to cartesian coordinate
# @param radius distance from origin
# @param psi   angle from z-axis
# @param theta angle from x-axis, along z-axis
def spher2cart(radius, psi, theta):
    sin_theta = np.sin(theta)
    return radius*sin_theta*np.cos(psi), radius*sin_theta*np.sin(psi), radius*np.cos(theta)

def intrins2cammat(intrins):
    cameraMatrix = np.array([[intrins[2], 0, intrins[4]],
                             [0, intrins[3], intrins[5]],
                             [0,0,1]])
    return cameraMatrix, intrins[:2]

def cammat2intrins(cameraMatrix, img_dim):
    intrins = [img_dim[0], img_dim[1],
               cameraMatrix[0,0], cameraMatrix[1,1],
               cameraMatrix[0,2], cameraMatrix[1,2]
              ]
    return intrins

##
# @brief interpolate between 2 transformation matrices(4x4)
# @remark Either POS_STEP and ROT_STEP or N_STEP should be given
def interpolate_T(T1, T2, POS_STEP=None, ROT_STEP=None, N_STEP=None):
    R_cur = T1[:3, :3]
    R_new = T2[:3, :3]

    P_cur = T1[:3, 3]
    P_new = T2[:3, 3]

    dP = P_new - P_cur
    dPnm = np.linalg.norm(dP)

    dR = np.matmul(R_cur.transpose(), R_new)
    dRvec = Rotation.from_dcm(dR).as_rotvec()
    dRnm = np.linalg.norm(dRvec)

    if N_STEP is None:
        assert None not in [POS_STEP, ROT_STEP], "Either POS_STEP and ROT_STEP or N_STEP should be given"
        N_STEP = max(int(ceil(dPnm / POS_STEP)), int(ceil(dRnm / ROT_STEP)))

    dPstep = dP / N_STEP
    Parr = np.array([np.arange(P_cur[i], P_new[i] + dPstep[i]/2, dPstep[i]) for i in range(3)]).transpose()

    dRstep = dRvec / N_STEP
    dRarr = np.array([np.arange(0, dRvec[i] + dRstep[i]/2, dRstep[i]) for i in range(3)])
    Rarr = np.array([np.matmul(R_cur, Rotation.from_rotvec(dRvec_tmp).as_dcm()) for dRvec_tmp in dRarr.transpose()])

    return [SE3(R, P) for R, P in zip(Rarr, Parr)]

def norm_SE3(T, rot_scale=0.1):
    return np.linalg.norm(T[:3,3]) + rot_scale*np.linalg.norm(Rotation.from_dcm(T[:3,:3]).as_rotvec())
