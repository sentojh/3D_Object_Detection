import os
import sys
import cv2
import copy
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
from collections import namedtuple
from .heuristics import *
from ..camera.kinect import Kinect
from ..camera.realsense import RealSense
from ..camera.camera_interface import CameraInterface
from ..detector_interface import DetectorInterface
from ...geometry.geotype import GEOTYPE
from ...utils.rotation_utils import *
from ...utils.utils import TextColors

from concurrent import futures
import logging
import math
import time
import cv2
import numpy as np
import grpc
from .grpc_cam import RemoteCam_pb2
from .grpc_cam import RemoteCam_pb2_grpc

MAX_MESSAGE_LENGTH = 100000000
GRPC_PORT = 10509
HOST_IP = "192.168.17.2"

##
# @class ColorDepthMap
# @param color numpy 8 bit array
# @param depth numpy 16 bit array
# @param intrins CamIntrins
# @param depth_scale multiplier for depthymap
ColorDepthMap = namedtuple('ColorDepthMap', ['color', 'depth', 'intrins', 'depth_scale'])


##
# @brief convert cdp to pcd
# @param cdp ColorDepthMap
# @param Tc camera coord w.r.t base coord
def cdp2pcd(cdp, Tc=None, depth_trunc=10.0):
    if Tc is None:
        Tc = np.identity(4)
    color = o3d.geometry.Image(cdp.color)
    depth = o3d.geometry.Image(cdp.depth)
    d_scale = cdp.depth_scale
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1 / d_scale,
                                                                    depth_trunc=depth_trunc,
                                                                    convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                         o3d.camera.PinholeCameraIntrinsic(
                                                             *cdp.intrins), SE3_inv(Tc))
    return pcd

##
# @param pcd point cloud
# @param voxel_size voxel size of downsampling
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 6
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200))
    return pcd_down, pcd_fpfh

##
# @param source source point cloud
# @param target target point cloud
# @param transformation estimated transformation from ICP to align source and target
def draw_registration_result(source, target, transformation, option_geos=[]):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    FOR_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    FOR_target = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=target.get_center())

    o3d.visualization.draw_geometries([source_temp, target_temp, FOR_origin, FOR_target]+option_geos)


##
# @class    MultiICP
# @brief    camera module with multi ICP for 3D object detection
#           Must call initialize/disconnect before/after usage.
#           If "Couldn't resolve requests" error is raised, check if it is connected with USB3.
#           You can check by running realsense-viewer from the terminal.
class MultiICP:
    ##
    # @param camera     subclass instances of CameraInterface (realsense or kinect)
    def __init__(self, camera):
        self.camera = camera
        self.img_dim = (720, 1280)
        # self.img_dim = (1080, 1920)
        self.ratio = 0.3
        self.thres_ICP = 0.15
        self.thres_front_ICP = 0.10
        self.config_list = []
        self.micp_dict = {}
        self.objectPose_dict = {}
        self.gscene = None
        self.crob = None
        self.sd = None
        self.visualize = False
        self.cache = None
        self.pcd_total = None

        self.multiobject_num = 1
        self.merge_mask = False
        self.remote_cam = False
        self.outlier_removal = None
        self.rmse_thres = 0.1



    ##
    # @brief initialize camera and set camera configuration
    def initialize(self, config_list=None, img_dim=None, remote_cam=False):
        if self.camera is None:
            if remote_cam:
               self.remote_cam = remote_cam
               print("Camera is not set - skip initialization, use remote camera")

               # get camera config from remote camera
               with grpc.insecure_channel('{}:{}'.format(HOST_IP, GRPC_PORT)) as channel:
                   stub = RemoteCam_pb2_grpc.RemoteCamProtoStub(channel)
                   request_id = 0
                   resp = stub.GetConfig(RemoteCam_pb2.GetConfigRequest(request_id=request_id))
                   print("request {} -> response {}".format(request_id, resp.response_id))
                   cam_mtx = np.array(resp.camera_matrix).reshape((3, 3))
                   dist_coeffs = np.array(resp.dist_coeffs).reshape((5,))
                   depth_scale = resp.depth_scale
                   width = resp.width
                   height = resp.height
                   print("==== Received camera config from remote camera ====")
                   self.config_list = [cam_mtx, dist_coeffs, depth_scale]
                   self.img_dim = (height, width)
                   self.dsize = tuple(reversed(self.img_dim))
                   self.h_fov_hf = np.arctan2(self.img_dim[1], 2 * self.config_list[0][0, 0])
                   self.v_fov_hf = np.arctan2(self.img_dim[0], 2 * self.config_list[0][1, 1])
            else:
                print("Camera is not set - skip initialization, use manually given camera configs")
                assert config_list is not None and img_dim is not None, "config_list and img_dim must be given for no-cam mode"
                self.config_list = config_list
                self.img_dim = img_dim
                self.dsize = tuple(reversed(img_dim))
                self.h_fov_hf = np.arctan2(self.img_dim[1], 2 * config_list[0][0, 0])
                self.v_fov_hf = np.arctan2(self.img_dim[0], 2 * config_list[0][1, 1])
                return
        else:
            self.camera.initialize()
            cameraMatrix, distCoeffs, depth_scale = self.camera.get_config()
            self.config_list = [cameraMatrix, distCoeffs, depth_scale]
            self.img_dim = self.camera.get_image().shape[:2]
            res_scale = np.max(np.ceil(np.divide(np.array(self.img_dim, dtype=float), 2000) / 2).astype(int) * 2)
            self.dsize = tuple(reversed(np.divide(self.img_dim, res_scale).astype(int)))
            self.h_fov_hf = np.arctan2(self.img_dim[1], 2*cameraMatrix[0,0])
            self.v_fov_hf = np.arctan2(self.img_dim[0], 2*cameraMatrix[1,1])
            print("Initialize Done")

    ##
    # @brief disconnect camera
    def disconnect(self):
        if self.camera is not None:
            self.camera.disconnect()

    ##
    # @brief   get camera configuration
    # @return  cameraMatrix 3x3 camera matrix in pixel units,
    # @return  distCoeffs distortion coefficients, 5~14 float array
    # @return  depthscale scale of depth value
    def get_camera_config(self):
        return self.config_list

    ##
    # @brief   get aligned RGB image and depthmap
    def get_image(self):
        if not self.remote_cam:
            color_image, depth_image = self.camera.get_image_depthmap()
        else:
            with grpc.insecure_channel('{}:{}'.format(HOST_IP, GRPC_PORT),
                                       options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)]) as channel:
                stub = RemoteCam_pb2_grpc.RemoteCamProtoStub(channel)
                request_id = 0
                resp = stub.GetImageDepthmap(RemoteCam_pb2.GetImageDepthmapRequest(request_id=request_id))
                print("request {} -> response {}".format(request_id, resp.response_id))
                color_image = np.array(resp.color).reshape((resp.height, resp.width, 3))
                depth_image = np.array(resp.depth).reshape((resp.height, resp.width))
                print("==== Received color, depth image from remote camera ====")
                self.img_dim = (resp.height, resp.width)
        Q = self.crob.get_real_robot_pose()
        return color_image, depth_image, Q

    ##
    # @brief  add micp_dict, hrule_dict, combined robot, geometry scene and shared detector
    # @param  micp_dict MultiICP class for each object
    # @param  sd   shared detector to detect object
    # @param  crob   CombinedRobot
    # @param  viewpoint     GeometryItem for viewpoint
    def set_config(self, micp_dict, sd, crob, viewpoint):
        self.micp_dict = micp_dict
        self.crob = crob
        self.sd = sd
        self.viewpoint = viewpoint
        self.gscene = viewpoint.gscene

    ##
    # @brief  add arbitrary color, depth image and joint value
    # @param  color_image   color image of object
    # @param  depth_image   depth image of object
    # @param  Q             joint values of robot
    def cache_sensor(self, color_image, depth_image, Q):
        self.cache = color_image, depth_image, Q

    ##
    # @brief change threshold value to find correspondenc pair during ICP
    # @param  thres_ICP     setting value of threshold
    # @param  thres_front_ICP   setting value of threshold
    def set_ICP_thres(self, thres_ICP=0.15, thres_front_ICP=0.10):
        self.thres_ICP = thres_ICP
        self.thres_front_ICP = thres_front_ICP

    ##
    # @abrief set the number of object which has multiple instance
    # @param  num     setting value of num
    def set_multiobject_num(self, num=1):
        self.multiobject_num = num

    ##
    # @brief  set merget option of masks for one object
    # @param  merge   whether separate masking merge or not
    def set_merge_mask(self, merge=True):
        self.merge_mask = merge
        self.multiobject_num = 1


    ##
    # @brief  setting the parameter remove points that have few neighbors in a given sphere around them
    # @param  nb_points which lets you pick the minimum amount of points that the sphere should contain
    # @param  radius   which defines the radius of the sphere that will be used for counting the neighbors
    def set_outlier_removal(self, nb_points=25, radius=0.04):
        self.outlier_removal = [nb_points, radius]


    def set_pcd_ratio(self, ratio=0.3):
        self.ratio = ratio


    def set_inlier_ratio(self, ratio=0.1):
        self.inlier_thres = ratio

    ##
    # @brief    detect 3D objects pose
    # @param    name_mask   object names to detect
    # @param    level_mask  list of rnb-planning.src.pkg.detector.detector_interface.DetectionLevel
    # @param    visualize   visualize the detection result, especially visualization in Open3D during ICP
    def detect(self, name_mask=None, level_mask=None, visualize=False):
        if level_mask is not None:
            name_mask_level = self.get_targets_of_levels(level_mask)
            if name_mask is not None:
                name_mask = list(set(name_mask).intersection(set(name_mask_level)))
            else:
                name_mask = name_mask_level
        print("name_mask is {}".format(name_mask))

        if self.crob is None:
            TextColors.YELLOW.println("[WARN] CombinedRobot is not set: call set_config()")
            return {}

        if self.cache is None:
            color_image, depth_image, Q = self.get_image()
        else:
            color_image, depth_image, Q = self.cache
            self.cache = None
        camera_mtx = self.config_list[0]
        cam_intrins = [self.img_dim[1], self.img_dim[0],
                       camera_mtx[0, 0], camera_mtx[1, 1],
                       camera_mtx[0, 2], camera_mtx[1, 2]]
        depth_scale = self.config_list[2]

        cdp = ColorDepthMap(color_image, depth_image, cam_intrins, depth_scale)
        if len(Q) == 13:
            Tc = self.viewpoint.get_tf(Q)
        elif len(Q) == 4:
            Tc = Q
        T_cb = SE3_inv(Tc)
        self.objectPose_dict = {}

        if self.sd is None:
            TextColors.YELLOW.println("[WARN] SharedDetector is not set: call set_config()")
            return {}


        print("Maximun num of object for detection : {}".format(self.multiobject_num))

        # Output of inference(mask for detected object)
        mask_out_list = self.inference(color_img=cdp.color)

        mask_dict = {}
        for idx in range(80):
            if np.any(mask_out_list[idx]):
                for name, value in class_dict.items():
                    if name_mask is not None:
                        if name in name_mask:
                            if value == idx:
                                num = int(np.max(mask_out_list[value]))
                                print('===== Detected : {}, {} object(s) ====='.format(name, num))
                                mask_dict[name] = mask_out_list[value]
                    else:
                        if value == idx:
                            num = int(np.max(mask_out_list[value]))
                            print('===== Detected : {}, {} object(s) ====='.format(name, num))
                            mask_dict[name] = mask_out_list[value]
            else:
                pass

        hrule_targets_dict = {}
        detect_dict = {}

        if name_mask is not None:
            for name, micp in self.micp_dict.items():
                if name in name_mask:
                    detect_dict[name] = micp
        else:
            detect_dict = self.micp_dict

        # obj_num = 0
        for name, micp in detect_dict.items():
            if name in mask_dict.keys():
                # # add to micp
                # masks = mask_dict[name]
                # mask_list = []
                # mask_zero = np.empty((self.img_dim[0],self.img_dim[1]), dtype=bool)
                # mask_zero[:,:] = False
                # mask_zero[np.where(masks ==  1)] = True
                # mask_list.append(mask_zero)

                # multiple instance
                masks = mask_dict[name]
                mask_list = []
                mask_zero = np.empty((self.img_dim[0], self.img_dim[1]), dtype=bool)
                mask_zero[:, :] = False
                if self.multiobject_num == 1:
                    mask_tmp = copy.deepcopy(mask_zero)
                    if self.merge_mask:
                        print("[NOTICE] You choose merge option for mask. Detected masks would be merged.")
                        mask_list.append(masks)
                    else:
                        mask_tmp[np.where(masks == 1)] = True
                        mask_list.append(mask_tmp)
                else:
                    num = min(self.multiobject_num, int(np.max(masks)))
                    for i in range(num):
                        mask_tmp = copy.deepcopy(mask_zero)
                        mask_tmp[np.where(masks==i+1)] = True
                        mask_list.append(mask_tmp)

                obj_num = 0
                for i_m, mask in enumerate(mask_list):
                    cdp_masked = apply_mask(cdp, mask)
                    micp.reset(Tref=Tc)
                    if micp.make_pcd(cdp_masked, ratio=self.ratio):
                        skip_normal_icp = False
                        multi_init_icp = False
                        Tguess = None
                        if name in self.gscene.NAME_DICT:  # if the object exists in gscene, use it as initial
                            g_handle = self.gscene.NAME_DICT[name]
                            print("\n'{}' is already in gscene. Use this to intial guess\n".format(name))
                            Tbo = g_handle.get_tf(Q)
                            Tguess = np.matmul(T_cb, Tbo)
                            skip_normal_icp = True
                        else: # if the object not exists in gscene, use grule
                            if micp.grule is not None: # if grule is defined by user
                                print("\n'{}' is not in gscene. Use manual input for initial guess\n".format(name))
                                Tguess = micp.grule.get_initial(micp.pcd)
                            else: # if grule is not defined by user, then use multi initial for ICP
                                print("\n'{}' is not in gscene. Use multiple initial guess\n".format(name))
                                multi_init_icp = True

                        skip_detection = False
                        # Compute ICP, front iCP
                        if multi_init_icp:
                            Tguess_list = self.get_multi_init_icp(micp.pcd, micp.Tref)
                            T_best = np.identity(4)
                            rmse_best = 1.
                            for it, Tguess in enumerate(Tguess_list):
                                if not skip_normal_icp:
                                    Tguess, inlier_rmse, inlier_ratio = micp.compute_ICP(To=Tguess, thres=self.thres_ICP,
                                                                 outlier_remove= self.outlier_removal, visualize=visualize)
                                if inlier_ratio < self.inlier_thres:
                                    skip_detection = True
                                if not skip_detection:
                                    T_, rmse, inlier_ratio = micp.compute_front_ICP(h_fov_hf=self.h_fov_hf, v_fov_hf=self.v_fov_hf,
                                                                      To=Tguess, thres=self.thres_front_ICP,
                                                                      visualize=visualize)
                                    if rmse < rmse_best and rmse>0:
                                        rmse_best = rmse
                                        T_best = T_
                                    if inlier_ratio < self.inlier_thres:
                                        skip_detection = True

                                print("Lowest rmse", rmse_best)
                                T = T_best
                        else:
                            if not skip_normal_icp:
                                Tguess, inlier_rmse, inlier_ratio = micp.compute_ICP(To=Tguess, thres=self.thres_ICP,
                                                             outlier_remove= self.outlier_removal, visualize=visualize)
                            if inlier_ratio < self.inlier_thres:
                                skip_detection = True
                            if not skip_detection:
                                T, rmse, inlier_ratio = micp.compute_front_ICP(h_fov_hf=self.h_fov_hf, v_fov_hf=self.v_fov_hf,
                                                                 To=Tguess, thres=self.thres_front_ICP, visualize=visualize)
                                if inlier_ratio < self.inlier_thres:
                                    skip_detection = True

                        if not skip_detection:
                            # self.objectPose_dict[name] = np.matmul(Tc, T)
                            name_i = "{}_{:01}".format(name, obj_num+1)
                            self.objectPose_dict[name_i] = np.matmul(Tc, T)
                            print('Found 6DoF pose of {}'.format(name_i))
                            obj_num +=1
            elif micp.hrule is not None:
                hrule_targets_dict[name] = micp
            elif name in class_dict.keys():
                print("{} not detected".format(name))
            else:
                raise (RuntimeError("{} not detected and has no detection rule".format(name)))

        for name, micp in sorted(hrule_targets_dict.items()):
            # add to micp
            micp.reset(Tref=Tc)

            if micp.make_pcd(cdp, ratio=self.ratio):
                hrule = micp.hrule
                print('===== Apply heuristic rule for {} ====='.format(name))
                if hrule.parent in detect_dict:
                    micp_parent = detect_dict[hrule.parent]
                else:
                    if hrule.parent in self.gscene.NAME_DICT:  # if the object exists in gscene, use it as initial
                        g_handle = self.gscene.NAME_DICT[hrule.parent]
                        print(
                            "\nParent object '{}' of '{}' is already in gscene. Apply heuristic rule based on this\n".format(
                                hrule.parent, name))
                        micp_parent = self.micp_dict[hrule.parent]
                        micp_parent.change_Tref(Tc)
                    else:
                        continue
                mrule = hrule.update_rule(micp, micp_parent, Tc=Tc)
                pcd_dict = mrule.apply_rule(micp.pcd, {hrule.parent: micp_parent.pose})

                T_list = []
                for name_i, pcd in pcd_dict.items():
                    micp.pcd = pcd

                    skip_normal_icp = False
                    multi_init_icp = False
                    # Check whether the object exists in gscene
                    if name in self.gscene.NAME_DICT:
                        g_handle = self.gscene.NAME_DICT[name]
                        print("\n'{}' is already in gscene. Use this to intial guess\n".format(name))
                        Tbo = g_handle.get_tf(Q)
                        Tguess = np.matmul(T_cb, Tbo)
                        skip_normal_icp = True
                    else:
                        if micp.grule is not None:  # if grule is defined by user
                            print("\n'{}' is not in gscene. Use manual input for initial guess\n".format(name))
                            Tguess = micp.grule.get_initial(micp.pcd,
                                                            R=detect_dict[hrule.parent].pose[:3, :3])
                        else:  # if grule is not defined by user, then use multi initial for ICP
                            print("\n'{}' is not in gscene. Use multiple initial guess\n".format(name))
                            multi_init_icp = True


                    # Compute ICP, front iCP
                    if multi_init_icp:
                        Tguess_list = self.get_multi_init_icp(micp.pcd, micp.Tref)
                        T_best = np.identity(4)
                        rmse_best = 1.
                        for it, Tguess in enumerate(Tguess_list):
                            if not skip_normal_icp:
                                Tguess, _ = micp.compute_ICP(To=Tguess, thres=self.thres_ICP,
                                                             outlier_remove= self.outlier_removal, visualize=visualize)
                            T_, rmse = micp.compute_front_ICP(h_fov_hf=self.h_fov_hf, v_fov_hf=self.v_fov_hf,
                                                              To=Tguess, thres=self.thres_front_ICP, visualize=visualize)
                            if rmse < rmse_best and rmse>0:
                                rmse_best = rmse
                                T_best = T_
                        print("Lowest rmse", rmse_best)
                        T = T_best
                    else:
                        if not skip_normal_icp:
                            Tguess, _ = micp.compute_ICP(To=Tguess, thres=self.thres_ICP,
                                                         outlier_remove= self.outlier_removal, visualize=visualize)
                        T, rmse = micp.compute_front_ICP(h_fov_hf=self.h_fov_hf, v_fov_hf=self.v_fov_hf,
                                                         To=Tguess, thres=self.thres_front_ICP, visualize=visualize)
                    T_list.append(T)
                    self.objectPose_dict[name_i] = np.matmul(Tc, T)
                    print('Found 6DoF pose of {}'.format(name_i))

            # for i_t, T in enumerate(T_list):
            #     name_i = "{}_{:02}".format(name, i_t)
            #     self.objectPose_dict[name_i] = T
            #     print('Found 6DoF pose of {}'.format(name_i))

        self.last_input = color_image, depth_image, Q, cam_intrins, depth_scale
        return self.objectPose_dict

    ##
    # @brief    list registered targets of specific detection level
    # @param    detection_level list of target detection levels
    # @return   names list of target names
    def get_targets_of_levels(self, detection_levels=None):
        names = []
        for name, info in self.micp_dict.items():
            obj_info = info.get_info()
            if obj_info.dlevel in detection_levels:
                names.append(name)
        return names

    ##
    # @brief    Acquire geometry kwargs of item
    # @param    name    item name
    # @return   kwargs  kwargs if name is available object name. None if not available.
    def get_geometry_kwargs(self, name):
        if "_" in name:
            name_cat = name.split("_")[0]
        else:
            name_cat = name
        if name_cat not in self.micp_dict:
            return None
        micp = self.micp_dict[name_cat]
        model = micp.model
        return dict(gtype=GEOTYPE.MESH, name=name_cat,
                    dims=(0.1, 0.1, 0.1), color=(0.8, 0.8, 0.8, 1),
                    display=True, fixed=True, collision=False,
                    vertices=np.matmul(np.asarray(model.vertices), micp.Toff_inv[:3,:3].transpose())+micp.Toff_inv[:3,3],
                    triangles=np.asarray(model.triangles))

    ##
    # @brief    add axis marker to GeometryHandle
    def add_item_axis(self, gscene, hl_key, item, axis_name=None):
        oname = item.oname
        axis_name = axis_name or oname
        if oname in gscene.NAME_DICT:
            aobj = gscene.NAME_DICT[oname]
            link_name = aobj.link_name
            Toff = np.matmul(aobj.Toff, item.Toff)
        else:
            link_candis = list(set([lname for lname in gscene.link_names
                                    if oname in lname
                                    and lname in [child_pair[1]
                                                  for child_pair
                                                  in gscene.urdf_content.child_map["base_link"]]
                                    ]))
            if len(link_candis) == 0:
                link_name = "base_link"
            elif len(link_candis) == 1:
                link_name = link_candis[0]
            else:
                raise(RuntimeError("Multiple object link candidates - marker link cannot be determined"))
            Toff = item.Toff
        gscene.add_highlight_axis(hl_key, axis_name, link_name, Toff[:3,3], Toff[:3,:3], axis="xyz")

    ##
    # @brief resize and inference image
    def inference(self, color_img):
        img_res = cv2.resize(color_img, dsize=self.dsize)
        mask_out_list_res = self.sd.inference(color_img=img_res)
        mask_out_list = np.zeros((80,) + tuple(color_img.shape[:2]), dtype=mask_out_list_res.dtype)
        for idx in range(80):
            if np.any(mask_out_list_res[idx]):
                for i_obj in range(1, np.max(mask_out_list_res[idx])+1):
                    mask_res = (cv2.resize((mask_out_list_res[idx] == i_obj).astype(np.uint8) * 255,
                                           dsize=tuple(reversed(self.img_dim)), interpolation=cv2.INTER_AREA
                                           ).astype(float) / 255
                                ).astype(np.uint8) * i_obj
                    mask_out_list[idx][np.where(mask_res>0)] = mask_res[np.where(mask_res>0)]
        return mask_out_list

   ##
    # @brief generate multiple initials for ICP
    # @param pcd point cloud
    # @param base_coord camera coordinate w.r.t base coord
    def get_multi_init_icp(self, pcd, Tbc):
        T_list = []
        center = pcd.get_center()
        off_x, off_y, off_z = (pcd.get_max_bound() - pcd.get_min_bound())/4.
        R_ = np.linalg.inv(Tbc[:3,:3])
        DIVIDE_NUM = 8
        for ir in range(DIVIDE_NUM):
            R = np.matmul(R_, np.linalg.inv(Rot_axis(3, ir*2*np.pi / DIVIDE_NUM)))
            t = np.array([center[0], center[1], center[2]])
            T = SE3(R,t)
            # for it in range(2):
            #     t_x = center[0] + float((1 - 2 * it) * off_x)
            #     for jt in range(2):
            #         t_y = center[1] + float((1 - 2 * jt) * off_y)
            #         for kt in range(2):
            #             t_z = center[2] + float((1 - 2 * kt) * off_z)
            #             t = np.array([t_x,t_y,t_z])
            #             T = SE3(R, t)
            #             T_list.append(T)
            T_list.append(T)
        return T_list


##
# @class    MultiICP_Obj
# @brief    class for each object which has pcd, model information only to be dected
class MultiICP_Obj:
    ##
    # @param obj_info   object information
    # @param hrule      heuristic rule class
    # @param grule      initial guess rule class
    def __init__(self, obj_info, hrule=None, grule=None):
        self.depth_trunc = 10.0
        self.model = None
        self.cdp = None
        self.pcd = None
        self.pcd_Tc_stack = []
        self.model_sampled = None
        self.obj_info = obj_info
        self.hrule = hrule
        self.grule = grule
        self.pose = None
        self.model = o3d.io.read_triangle_mesh(self.obj_info.url)
        self.model.vertices = o3d.utility.Vector3dVector(
            np.asarray(self.model.vertices) * np.array([self.obj_info.scale[0],
                                                        self.obj_info.scale[1],
                                                        self.obj_info.scale[2]]))
        self.set_Toff()
        self.Tref = np.identity(4)


    def get_info(self):
        return self.obj_info

    def get_rule(self):
        return self.hrule, self.grule

    ##
    # @brief add model mesh
    # @param model_name name of CAD model
    def set_Toff(self):
        # obj_info = get_obj_info()
        # model_info = obj_info[model_name]
        model_info = self.obj_info
        if model_info.Toff is None:
            self.Toff = np.identity(4)
            self.Toff_inv = SE3_inv(self.Toff)
        else:
            self.Toff = model_info.Toff
            self.Toff_inv = SE3_inv(self.Toff)

    ##
    # @brief reset pcd and set reference coordinate
    def reset(self, Tref=None):
        self.clear()
        self.Tref = Tref

    ##
    # @briref change reference coordinate and transform pcd & pose
    def change_Tref(self, Tref_new):
        Trr0 = np.matmul(np.linalg.inv(Tref_new), self.Tref)
        self.pcd.points = o3d.utility.Vector3dVector(
            np.matmul(Trr0[:3, :3], np.asarray(self.pcd.points).T).T + Trr0[:3, 3])
        self.pose = np.matmul(Trr0, self.pose)
        self.Tref = Tref_new

    ##
    # @brief add pcd from image, sampled pcd from mesh.
    # @param cdp_masked ColorDepthMap
    # @param Tc camera coord w.r.t base coord
    # @param ratio ratio of number of points
    def make_pcd(self, cdp_masked, Tc=None, ratio=0.3):
        if Tc is None:
            Tc = np.identity(4)
        pcd_cam = cdp2pcd(cdp_masked, depth_trunc=self.depth_trunc)
        pcd = cdp2pcd(cdp_masked, Tc=Tc, depth_trunc=self.depth_trunc)

        self.pcd_Tc_stack.append((pcd_cam, Tc, pcd))
        self.pcd = self.pcd_Tc_stack[0][2]
        for _pcd in self.pcd_Tc_stack[1:]:
            self.pcd += _pcd[2]
        if len(self.pcd_Tc_stack) > 1:
            self.pcd = self.pcd.uniform_down_sample(every_k_points=len(self.pcd_Tc_stack))
        self.pcd = self.pcd.uniform_down_sample(every_k_points=int(1/ratio))
        self.model.compute_vertex_normals()
        try:
            self.model_sampled = self.model.sample_points_uniformly(
                number_of_points=int(len(np.array(self.pcd.points)) * ratio))
            # self.model_sampled = self.model.sample_points_poisson_disk(
            #                                             number_of_points=int(len(np.array(self.pcd.points) * ratio)))
        except Exception as e:
            print("[WARN] Not obtained point cloud of object")
            # self.model_sampled = self.model.sample_points_uniformly(number_of_points=2000)
            return False
        return True

    ##
    # @param To    initial transformation matrix of geometry object in the intended icp origin coordinate
    # @param thres max distance between corresponding points
    def compute_ICP(self, To=None, thres=0.15,
                    relative_fitness=1e-15, relative_rmse=1e-15, max_iteration=500000,
                    voxel_size=0.03, ratio=0.3, outlier_remove=None, visualize=False
                    ):
        if To is None:
            To, fitness = self.auto_init(0, voxel_size)
        target = copy.deepcopy(self.pcd)
        if outlier_remove is None:
            target, ind = target.remove_radius_outlier(nb_points=25, radius=0.05)
        else:
            target, ind = target.remove_radius_outlier(nb_points=outlier_remove[0], radius=outlier_remove[1])
        source = copy.deepcopy(self.model_sampled)
        source_bak = copy.deepcopy(source)


        if visualize:
            self.draw(To)

        To = np.matmul(To, self.Toff_inv)

        # Guess Initial Transformation
        trans_init = To
        threshold = thres

        # # Sampling points to reduct number of points
        # source_down = source.uniform_down_sample(every_k_points=2)
        # target_down = target.uniform_down_sample(every_k_points=2)


        print("Apply point-to-point ICP")
        threshold = thres
        reg_p2p = o3d.registration.registration_icp(source, target, threshold, trans_init,
                                                    o3d.registration.TransformationEstimationPointToPoint(),
                                                    o3d.registration.ICPConvergenceCriteria(
                                                        relative_fitness=relative_fitness,
                                                        relative_rmse=relative_rmse,
                                                        max_iteration=max_iteration))
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        ICP_result = reg_p2p.transformation

        print("Total ICP Transformation is:")
        print(ICP_result)
        ICP_result = np.matmul(ICP_result, self.Toff)
        if visualize:
            self.draw(ICP_result, source, target)

        self.pose = ICP_result

        len_correspends = len(set(np.asarray(reg_p2p.correspondence_set)[:,1]))
        len_tar =  len(np.asarray(target.points))
        inlier_ratio = float(len_correspends) / len_tar if len_tar > 0 else 0
        print("Inlier ratio: {}".format(inlier_ratio))

        self.reg_p2p = reg_p2p
        return ICP_result, reg_p2p.inlier_rmse, inlier_ratio

    ##
    # @param Tc_cur this is new camera transformation in pcd origin
    # @param To    initial transformation matrix of geometry object in the intended icp origin coordinate
    # @param thres max distance between corresponding points
    def compute_front_ICP(self, h_fov_hf, v_fov_hf, Tc_cur=None, To=None, thres=0.07,
                          relative_fitness=1e-19, relative_rmse=1e-19, max_iteration=700000,
                          voxel_size=0.02, visualize=False
                          ):
        if To is None:
            if self.pose is not None:
                To = self.pose
            else:
                To, fitness = self.auto_init(0, voxel_size)

        if Tc_cur is None:
            Tc_cur = SE3(np.identity(3), (0, 0, 0))

        target = copy.deepcopy(self.pcd)

        T_cb = SE3_inv(Tc_cur)  # base here is the point cloud base defined when added
        T_co = np.matmul(np.matmul(T_cb, To), self.Toff_inv)
        # model_mesh = self.model.compute_vertex_normals()
        model_pcd = copy.deepcopy(self.model_sampled)
        source_bak = copy.deepcopy(model_pcd)

        try:
            # remove points whose normal vector direction are opposite to camera direction vector
            normals = np.asarray(model_pcd.normals)
            points = np.asarray(model_pcd.points)
            # point_normals = normals
            # view_vec = SE3_inv(Tguess)[:3,2]
            Poc = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
            P_co = np.matmul(T_co[:3, :3], Poc) + T_co[:3, 3]

            point_normals = np.matmul(T_co[:3, :3], normals.T).T
            view_vec = P_co / np.linalg.norm(P_co)
            idx = []
            for i in range(len(point_normals)):
                if np.dot(view_vec, point_normals[i]) < 0:
                    idx.append(i)

            # remove points which are not in trainge plane from traingles
            pts_md = np.array(points[idx])

            point_c = np.asarray(np.matmul(pts_md, np.transpose(T_co[:3, :3])) + T_co[:3, 3])
            points_sph = np.transpose(cart2spher(*np.transpose(np.matmul(point_c, Rot_axis(1, np.pi / 2)))))
            points_xyz = np.transpose(spher2cart(*np.transpose(points_sph)))

            verts = np.asarray(np.matmul(self.model.vertices, np.transpose(T_co[:3, :3])) + T_co[:3, 3])
            verts_sph = np.transpose(cart2spher(*np.transpose(np.matmul(verts, Rot_axis(1, np.pi / 2)))))
            trigs = np.asarray(self.model.triangles)

            pts = points_sph[:, 1:]
            dists = points_sph[:, 0] + 3e-3
            in_mask_accum = np.zeros(len(points_sph), dtype=bool)
            for count, (i, j, k) in enumerate(trigs):
                r = np.max(verts_sph[[i, j, k]][:, 0])
                p1, p2, p3 = verts_sph[[i, j, k]][:, 1:]
                p12 = p1 - p2
                pt2 = pts - p2
                p23 = p2 - p3
                pt3 = pts - p3
                p31 = p3 - p1
                pt1 = pts - p1
                sign2 = np.sign(np.cross(p12, pt2))
                sign3 = np.sign(np.cross(p23, pt3))
                sign1 = np.sign(np.cross(p31, pt1))

                in_mask = np.all([sign1 == sign2,
                                  sign2 == sign3,
                                  dists > r], axis=0)

                in_mask_accum = np.logical_or(in_mask_accum, in_mask)
            idc_masked = np.logical_not(in_mask_accum)

            points_front = np.asarray(pts_md)[idc_masked]

            front_pcd = o3d.geometry.PointCloud()
            front_pcd.points = o3d.utility.Vector3dVector(points_front)
            source = copy.deepcopy(front_pcd)
            #
            # if visualize:
            #     cam_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
            #     cam_coord.transform(Tc_cur)
            #     self.draw(To, source, target, [cam_coord])

            To = np.matmul(To, self.Toff_inv)

            # Guess Initial Transformation
            trans_init = To

            # remove points which are not in FOV
            center_point = target.get_center()
            points_converted = np.matmul(To[:3,:3], points_front.T).T + To[:3,3]

            points_converted = points_converted[np.where(np.abs(points_converted[:,0]/points_converted[:,2]) < np.tan(h_fov_hf))]
            points_converted = points_converted[np.where(np.abs(points_converted[:,1]/points_converted[:,2]) < np.tan(v_fov_hf))]

            points_remain = np.matmul(np.linalg.inv(To)[:3,:3], points_converted.T).T + np.linalg.inv(To)[:3,3]

            front_pcd = o3d.geometry.PointCloud()
            front_pcd.points = o3d.utility.Vector3dVector(points_remain)
            source = copy.deepcopy(front_pcd)
            source_bak = copy.deepcopy(source)
        except Exception as e:
            print(e)
            print("[WARN] Number of points after front ICP pre-processing <=0")
            source = copy.deepcopy(model_pcd)
            src_num = len(np.asarray(source.points))
            target_num = len(np.asarray(target.points))
            source = source.uniform_down_sample(every_k_points=int(src_num/target_num))


        # match the number of points between model_sampled pcd and data pcd
        # discrepancy = float(len(np.asarray(target.points))/len(np.asarray(source.points)))
        # target = target.uniform_down_sample(every_k_points=int(discrepancy))
        # target, ind = target.remove_radius_outlier(nb_points=40, radius=0.03)
        # source = source.voxel_down_sample(voxel_size)
        # target = target.voxel_down_sample(voxel_size)

        # source_num = np.asarray(source.points).shape[0]
        # target_num = np.asarray(target.points).shape[0]
        # total_num = min(int(source_num/2.5), int(target_num/2.5))
        # source_indices = np.random.choice(source_num, total_num, replace=False)
        # target_indices = np.random.choice(target_num, total_num, replace=False)
        # source_rand = o3d.geometry.PointCloud()
        # target_rand = o3d.geometry.PointCloud()
        # for i in range(len(source_indices)):
        #     source_rand.points.append(source.points[source_indices[i]])
        # for i in range(len(target_indices)):
        #     target_rand.points.append(target.points[target_indices[i]])
        #
        # source = source_rand
        # target = target_rand

        if visualize:
            print("initial: \n{}".format(np.round(To, 2)))
            cam_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
            cam_coord.transform(Tc_cur)
            self.draw(np.matmul(To, self.Toff), source, target, [cam_coord])

        # # Sampling points to reduct number of points
        # source_down = source.uniform_down_sample(every_k_points=2)
        # target_down = target.uniform_down_sample(every_k_points=2)

        # if visualize:
        #     vis = o3d.visualization.Visualizer()
        #     vis.create_window()
        #     vis.add_geometry(source)
        #     vis.add_geometry(target)
        # fitness_prev = 0.0
        # inlier_rmse_prev = 1.0
        # print("Apply point-to-point ICP")
        # while (True):
        #     reg_p2p = o3d.registration.registration_icp(source, target, threshold, np.identity(4),
        #                                                 o3d.registration.TransformationEstimationPointToPoint(),
        #                                                 o3d.registration.ICPConvergenceCriteria(
        #                                                     max_iteration=1))
        #     source.transform(reg_p2p.transformation)
        #     if visualize:
        #         time.sleep(0.03)
        #         vis.update_geometry(source)
        #         vis.poll_events()
        #         vis.update_renderer()
        #     ICP_result = np.matmul(reg_p2p.transformation, ICP_result)
        #     delta_fitness = abs(reg_p2p.fitness - fitness_prev)
        #     delta_rmse = abs(reg_p2p.inlier_rmse - inlier_rmse_prev)
        #     if delta_fitness < relative_fitness or delta_rmse < relative_rmse:
        #         time.sleep(0.04)
        #         break
        #     else:
        #         fitness_prev = reg_p2p.fitness
        #         inlier_rmse_prev = reg_p2p.inlier_rmse
        # if visualize:
        #     vis.destroy_window()
        print("Apply point-to-point ICP")
        reg_p2p = o3d.registration.registration_icp(source, target, thres, trans_init,
                                                    o3d.registration.TransformationEstimationPointToPoint(),
                                                    o3d.registration.ICPConvergenceCriteria(
                                                        relative_fitness=relative_fitness,
                                                        relative_rmse=relative_rmse,
                                                        max_iteration=max_iteration))
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        ICP_result = reg_p2p.transformation

        print("Total ICP Transformation is:")
        print(ICP_result)
        ICP_result = np.matmul(ICP_result, self.Toff)
        if visualize:
            print("result: \n{}".format(np.round(ICP_result, 2)))
            self.draw(ICP_result, source, target)

        # # BEV ICP
        # T_bo = np.matmul(self.Tref, ICP_result)
        # source_b_np = np.matmul(T_bo[:3, :3], np.asarray(source_bak.points).T).T + T_bo[:3, 3]
        # source_b_np[:, 2] = 0
        # source_b = o3d.geometry.PointCloud()
        # source_b.points = o3d.utility.Vector3dVector(source_b_np)
        # source_b.voxel_down_sample(0.03)
        #
        # target_b_np = np.matmul(self.Tref[:3, :3], np.asarray(target.points).T).T + self.Tref[:3, 3]
        # target_b_np[:, 2] = 0
        # target_b = o3d.geometry.PointCloud()
        # target_b.points = o3d.utility.Vector3dVector(target_b_np)
        # target_b.voxel_down_sample(0.03)
        #
        # if visualize:
        #     draw_registration_result(source_b, target_b, np.identity(4))
        #
        # print("Apply point-to-point ICP, BEV version")
        # reg_p2p = o3d.registration.registration_icp(source_b, target_b, thres, np.identity(4),
        #                                             o3d.registration.TransformationEstimationPointToPoint(),
        #                                             o3d.registration.ICPConvergenceCriteria(
        #                                                 relative_fitness=relative_fitness,
        #                                                 relative_rmse=relative_rmse,
        #                                                 max_iteration=max_iteration))
        # BEV_ICP_result = reg_p2p.transformation
        # print("Total BEV ICP Transformation is:")
        # print(BEV_ICP_result)
        # if visualize:
        #     draw_registration_result(source_b, target_b, BEV_ICP_result)
        #
        # Too_ = np.matmul(np.linalg.inv(ICP_result),
        #                  np.matmul(np.linalg.inv(self.Tref),
        #                            np.matmul(np.matmul(self.Tref, ICP_result), BEV_ICP_result)))
        # self.pose = np.matmul(np.matmul(ICP_result, Too_), self.Toff)
        # if visualize:
        #     self.draw(self.pose, source_bak, target)
        self.pose = ICP_result

        len_correspends = len(set(np.asarray(reg_p2p.correspondence_set)[:,1]))
        len_tar =  len(np.asarray(target.points))
        inlier_ratio = float(len_correspends) / len_tar if len_tar > 0 else 0
        print("Inlier ratio: {}".format(inlier_ratio))

        self.reg_p2p = reg_p2p
        return ICP_result, reg_p2p.inlier_rmse, inlier_ratio

    def draw(self, To, source=None, target=None, option_geos=[]):
        if source is None: source = self.model_sampled
        if target is None: target = self.pcd
        To = np.matmul(To, self.Toff_inv)
        draw_registration_result(source, target, To, option_geos)

    def auto_init(self, init_idx=0, voxel_size=0.04):
        pcd_cam, Tc, _ = self.pcd_Tc_stack[init_idx]
        Tc_inv = SE3_inv(Tc)
        source_down, source_fpfh = preprocess_point_cloud(pcd_cam, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(self.model_sampled, voxel_size)

        distance_threshold = voxel_size * 1.4
        result = o3d.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), 3, [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))

        To = matmul_series(Tc, result.transformation, self.Toff)
        return To, result.fitness

    def clear(self):
        self.pcd = None
        self.pcd_Tc_stack = []
        self.cdp = None
        self.model_sampled = None

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    FOR_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])

    FOR_model = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    FOR_model.transform(transformation)
    FOR_model.translate(source_temp.get_center() - FOR_model.get_center())

    FOR_target = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=target.get_center())
    o3d.visualization.draw_geometries([source_temp, target, FOR_origin, FOR_model, FOR_target])

##
# @param cdp ColorDepthMap
# @param mask segmented result from object detection algorithm
def apply_mask(cdp, mask):
    mask_u8 = np.zeros_like(mask).astype(np.uint8)
    mask_u8[np.where(mask)] = 255
    color_masked = cv2.bitwise_and(cdp.color, cdp.color, mask=mask_u8).astype(np.uint8)
    depth_masked = cv2.bitwise_and(cdp.depth, cdp.depth, mask=mask_u8).astype(np.uint16)
    return ColorDepthMap(color_masked, depth_masked, cdp.intrins, cdp.depth_scale)

# @brief adjust T upright around roi pcd center
def fit_vertical(T_bc, Tbo, pcd_roi, height=0):
    pcd_center_prev = np.matmul(T_bc[:3,:3],
                                pcd_roi.get_center()
                               ) + T_bc[:3,3]
    T_bo_p = SE3(Tbo[:3,:3], pcd_center_prev)
    T_pooc = np.matmul(SE3_inv(T_bo_p), Tbo)
    T_bo_p[:3,:3] = Rot_axis(3, Rot2axis(Tbo[:3,:3], 3))
    T_bo_c_fix = np.matmul(T_bo_p, T_pooc)
    T_bo_c_fix[2,3] = height
    return T_bo_c_fix