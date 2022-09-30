import os
import sys
import numpy as np
import copy
import open3d as o3d
from enum import Enum
from ...geometry.geometry import GEOTYPE
from ...utils.rotation_utils import *
from heuristics import *
from ...global_config import RNB_PLANNING_DIR

def get_obj_info():
    obj_info = {
        'cup': ObjectInfo('cup', dlevel=DetectionLevel.MOVABLE, gtype=GEOTYPE.CYLINDER,
                          dims=(0.4, 0.3, 0.01), color=(0.9, 0.9, 0.9, 0.2),
                          Toff=SE3(np.identity(3), (0, 0, 0)), scale=(1e-3,1e-3,1e-3),
                          url=RNB_PLANNING_DIR+'release/multiICP_data/cup.STL'),

        'bed': ObjectInfo('bed', dlevel=DetectionLevel.ENVIRONMENT, gtype=GEOTYPE.BOX,
                          dims=(1.70,0.91,0.66), color=(0.9,0.9,0.9,0.2),
                          Toff=SE3([[0,1,0],[0,0,1],[1,0,0]], (0.455,0,1.02)), scale=(1e-3,1e-3,1e-3),
                          url=RNB_PLANNING_DIR+'release/multiICP_data/bed.STL'),

        'closet': ObjectInfo('closet', dlevel=DetectionLevel.ENVIRONMENT, gtype=GEOTYPE.BOX,
                             dims=(0.4, 0.3, 0.01), color=(0.9, 0.9, 0.9, 0.2),
                             Toff=SE3([[1,0,0],[0,0,1],[0,-1,0]], (0.3,0,0.2725)), scale=(1e-3,1e-3,1e-3),
                             url=RNB_PLANNING_DIR+'release/multiICP_data/top_table.STL'),

        'dining table': ObjectInfo('dining table', dlevel=DetectionLevel.ENVIRONMENT, gtype=GEOTYPE.BOX,
                                   dims=(1.6, 0.8, 0.725), color=(0.9, 0.9, 0.9, 0.2),
                                   Toff=SE3(np.identity(3), (0, 0, 0)), scale=(1., 1., 1.),
                                   url=RNB_PLANNING_DIR + 'ws_ros/src/my_mesh/meshes/stl/table_floor_centered_m_scale.STL'),

        'suitcase': ObjectInfo('suitcase', dlevel=DetectionLevel.ENVIRONMENT, gtype=GEOTYPE.BOX,
                               dims=(0.4, 0.29, 0.635), color=(0.9, 0.9, 0.9, 0.2),
                               Toff=SE3(np.identity(3), (0, 0, 0)), scale=(1., 1., 1.),
                               url=RNB_PLANNING_DIR + 'ws_ros/src/my_mesh/meshes/stl/carrier_centered_m_scale.STL'),

        'clock': ObjectInfo('clock', dlevel=DetectionLevel.ENVIRONMENT, gtype=GEOTYPE.BOX,
                            dims=(0.138, 0.05, 0.078), color=(0.9, 0.9, 0.9, 0.2),
                            Toff=SE3(np.identity(3), (0, 0, 0)), scale=(1., 1., 1.),
                            url=RNB_PLANNING_DIR + 'ws_ros/src/my_mesh/meshes/stl/tableclock_centered_m_scale.STL'),
        'chair': ObjectInfo('chair', dlevel=DetectionLevel.ENVIRONMENT, gtype=GEOTYPE.BOX,
                            dims=(0.37, 0.37, 0.455), color=(0.9, 0.9, 0.9, 0.2),
                            Toff=SE3(np.identity(3), (0, 0, 0)), scale=(1., 1., 1.),
                            url=RNB_PLANNING_DIR + 'ws_ros/src/my_mesh/meshes/stl/chair_floor_centered_m_scale.STL')
    }
    return obj_info


class ClosetRuleFun(UpdateRuleFun):
    ##
    # @param micp_target on camera coords
    # @param micp_parent on camera coords
    def __call__(self, micp_target, micp_parent, Tc=None):
        return hrule_closet(micp_target, micp_parent, self.mbr)

# mbr.update_rule = ClosetRuleFun(mbr)

def remove_bed(pcd_original, pcd_bed):
    dists = pcd_original.compute_point_cloud_distance(pcd_bed)
    dists = np.asarray(dists)

    idx = np.where(dists > 0.07)[0]
    p_inliers = []
    for i in range(len(idx)):
        p_inliers.append(pcd_original.points[idx[i]])

    return p_inliers

##
# @param pcd_total  on cam coord
# @param pcd_bed    on cam coord
# @param T_bed      on cam coord
def check_closet_location(pcd_total, pcd_bed, T_bed, bed_dims, floor_margin=0.1, visualize=False):
    pcd_total = pcd_total.uniform_down_sample(every_k_points=11)
    if visualize:
        o3d.visualization.draw_geometries([pcd_total])

    # Remove bed
    pcd_top_table = o3d.geometry.PointCloud()
    pcd_top_table.points = o3d.utility.Vector3dVector(remove_bed(pcd_total, pcd_bed))
    if visualize:
        o3d.visualization.draw_geometries([pcd_top_table])

    # Remove other noise
    cl, ind = pcd_top_table.remove_radius_outlier(nb_points=25, radius=0.15)
    pcd_top_table = cl
    if visualize:
        o3d.visualization.draw_geometries([pcd_top_table])

    # Determine rough location of top_table
    points = np.asarray(pcd_top_table.points)
    points_4d = np.pad(points, ((0,0), (0,1)), "constant", constant_values=1)
    points_transformed = []

    points_transformed = np.matmul(points_4d, np.linalg.inv(T_bed).transpose())

    points_transformed_np = np.array(points_transformed)[:,:3]

    # Remove background based on bed_vis coord
    out_x = np.where(np.abs(points_transformed_np[:,0])>bed_dims[0]/2)[0]
    out_y = np.where(np.abs(points_transformed_np[:,1])>bed_dims[1]/2+bed_dims[1])[0]
    out_z = np.where(points_transformed_np[:,2]<floor_margin)[0]
    out_all = sorted(set(out_x).union(out_y).union(out_z))
    in_all = sorted(set(np.arange(len(points_transformed_np))) - set(out_all))
    points_transformed = np.array(points_transformed)[in_all, :3]

    if visualize:
        vis_pointcloud_np(points_transformed)

    # Determine closet location by checking num of points
    check_left = 0
    check_right = 0
    for i in range(len(points_transformed)):
        if points_transformed[i][1] > 0:
            check_right += 1
        elif points_transformed[i][1] < 0:
            check_left += 1

    TOP_TABLE_MODE = "LEFT"
    if check_left > check_right:
        TOP_TABLE_MODE = "LEFT"
    else:
        TOP_TABLE_MODE = "RIGHT"

    print("CLOSET ON {}".format(TOP_TABLE_MODE))
    return TOP_TABLE_MODE


##
# @param micp_closet on camera coordinates
# @param micp_bed on camera coordinates
def hrule_closet(micp_closet, micp_bed, mrule_closet):
    obj_info = get_obj_info()
    bed_dims = obj_info["bed"].dims
    CLOSET_LOCATION = check_closet_location(micp_closet.pcd, micp_bed.pcd, micp_bed.pose, bed_dims)

    mrule_closet.box_clear()
    # bed_box
    mrule_closet.add_box(MaskBox(Toff=SE3(np.identity(3), (0.02, 0, 0.5)),
                                 dims=(3, 1.6, 1.3), include=False))
    # bed_wall
    mrule_closet.add_box(MaskBox(Toff=SE3(np.identity(3), (-1.27, 0, 1.5)),
                                 dims=(0.5, 0.7, 0.3), include=False))
    # floor_box
    mrule_closet.add_box(MaskBox(Toff=SE3(np.identity(3), (0, 0, 0)), dims=(15, 15, 0.4), include=False))

    if CLOSET_LOCATION == "LEFT":
        # bed_left_space
        mrule_closet.add_box(MaskBox(Toff=SE3(np.identity(3), (0.02, -0.9, 1)),
                                     dims=(2.5, 1, 3), include=True))
    elif CLOSET_LOCATION == "RIGHT":
        # bed_right_space
        mrule_closet.add_box(MaskBox(Toff=SE3(np.identity(3), (0.02, 0.9, 1)),
                                     dims=(2.5, 1, 3), include=True))

    return mrule_closet