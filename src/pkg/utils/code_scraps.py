import random
from .rotation_utils import *


##
# @brief get looking motion to a target point. com_link to tip link distance is maintained.
# @param target_point target point in global coords
# @param view_dir     looking direction in tip link
def get_look_motion(mplan, rname, from_Q, target_point, com_link,
                    cam_link=None, view_dir=[0,0,1], timeout=1):
    gscene = mplan.gscene
    if isinstance(target_point, GeometryItem):
        target_point = target_point.get_tf(from_Q)[:3,3]
    if cam_link is None:
        cam_link = mplan.chain_dict[rname]['tip_link']
    ref_link = mplan.chain_dict[rname]['link_names'][0]
    Trb = SE3_inv(gscene.get_tf(ref_link, from_Q))
    target_point = np.matmul(Trb[:3,:3], target_point)+Trb[:3,3]
    Tcur = gscene.get_tf(cam_link, from_Q, from_link=ref_link)
    Tcom = gscene.get_tf(com_link, from_Q, from_link=ref_link)
    cur_dir = np.matmul(Tcur[:3,:3], view_dir)
    target_dir = target_point - Tcom[:3,3]
    dR = Rotation.from_rotvec(calc_rotvec_vecs(cur_dir, target_dir)).as_dcm()
    dRr = matmul_series(Tcom[:3,:3].transpose(), dR, Tcom[:3,:3])

    Toc = np.matmul(SE3_inv(Tcom), Tcur)
    Tot = np.matmul(SE3(dRr, (0,)*3), Toc)
    Ttar_ref = np.matmul(Tcom, Tot)
    retract_N = 10
    retract_step = 0.05
    mplan.update_gscene()
    for i_ret in range(retract_N):
        dP = np.multiply(view_dir, retract_step*i_ret)
        Ttar = np.copy(Ttar_ref)
        Ttar[:3,3] = Ttar_ref[:3,3] - np.matmul(Ttar_ref[:3,:3], dP)
        # xyzquat = T2xyzquat(Ttar)
        # traj, succ = mplan.planner.plan_py(rname, cam_link, xyzquat[0]+xyzquat[1], ref_link, from_Q,
        #                                    timeout=timeout)
        traj, succ = mplan.get_incremental_traj(cam_link, np.identity(4), Ttar,
                                                from_Q, step_size=0.01, ERROR_CUT=0.01,
                                                SINGULARITY_CUT=0.01, VERBOSE=False,
                                                ref_link=ref_link, VISUALIZE=False)
        if succ:
            Qtar = traj[-1]
            Qdiff = Qtar-from_Q
            steps = int(np.max(np.abs(Qdiff)) / 0.01)
            traj = from_Q[np.newaxis,:] + Qdiff[np.newaxis, :]*np.arange(steps+1)[:,np.newaxis].astype(float)/steps
            if mplan.validate_trajectory(traj):
                break
            else:
                succ = False
    return traj, succ


from scipy.cluster.vq import kmeans2


##
# @brief get scanning motion that covers target
# @param mplan MoveitPlanner
# @param viewpoint GeometryItem for viewpoint, of which +z is viewing direction
# @param target    GeometryItem for target
# @param Q_ref     Reference pose to try scanning
# @param fov_def   Field of View of the camera, in degrees
# @param N_max     max. number of view poses
def get_scan_motions(mplan, viewpoint, target, Q_ref, fov_deg=60, N_max=5):
    cam_link = viewpoint.link_name
    robot_name = [rname for rname, info in mplan.chain_dict.items() if cam_link in info['link_names']][0]
    T_ref = mplan.gscene.get_tf(viewpoint.link_name, Q_ref)
    T_ref_inv = np.linalg.inv(T_ref)

    T_tar = target.get_tf(Q_ref)
    traj, succ = get_look_motion(mplan, robot_name, np.array(Q_ref), T_tar[:3, 3],
                                 viewpoint.link_name, view_dir=viewpoint.orientation_mat[:, 2])

    assert succ, "Failed to get initial ref view"

    Q_ref = traj[-1]
    T_ref = mplan.gscene.get_tf(viewpoint.link_name, Q_ref)
    T_ref_inv = np.linalg.inv(T_ref)

    # get vertices and internal point samples
    verts, radi = target.get_vertice_radius_from(Q_ref)
    vtx_samples = []
    for _ in range(1000):
        v1, v2, v3 = random.sample(verts, 3)
        a, b, c = np.random.rand(3)
        vtx_samples.append((a * v1 + b * v2 + c * v3) / (a + b + c))
    vtx_samples = np.array(vtx_samples)
    verts_c = np.matmul(T_ref_inv[:3, :3], vtx_samples.T).T + T_ref_inv[:3, 3]

    # get theta and psi of vertices (rotate 90 degrees to convert in continuous region)
    verts_sph = np.transpose(cart2spher(*np.transpose(np.matmul(verts_c, Rot_axis(1, np.pi / 2)))))
    verts_sph2 = verts_sph[:, 1:]

    # makes largest clusters with sizes smaller than fov
    fov_rad_hf = np.deg2rad(fov_deg) / 2
    for N in range(1, N_max + 1):
        centroid, label = kmeans2(verts_sph2, N)
        max_dists = []
        for i in range(N):
            if np.sum(label == i) > 0:
                max_dists.append(np.max(np.linalg.norm(verts_sph2[label == i] - centroid[i], axis=-1)))
        if np.max(max_dists) < fov_rad_hf:  # stop if max angular dist < fov
            break
        if N == N_max:
            raise (RuntimeError("Failed to generate look motion in given number"))

    # makes look trajectories for view centers
    view_traj_list = []
    for ctr in centroid:
        ctr_c = np.matmul(spher2cart(1, *ctr), Rot_axis(1, np.pi / 2).T)
        P_tar = np.matmul(T_ref[:3, :3], ctr_c) + T_ref[:3, 3]
        traj, succ = get_look_motion(mplan, robot_name, np.array(Q_ref), P_tar, viewpoint.link_name,
                                     view_dir=viewpoint.orientation_mat[:, 2])
        if not succ:
            raise (RuntimeError("Failed to generate look motion"))
        view_traj_list.append(traj)
    return view_traj_list
