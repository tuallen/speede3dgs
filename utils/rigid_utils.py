import os
import numpy as np
import torch
import tqdm
from utils.general_utils import get_expon_lr_func


def skew(w: torch.Tensor) -> torch.Tensor:
    """Build a skew matrix ("cross product matrix") for vector w.

    Modern Robotics Eqn 3.30.

    Args:
      w: (N, 3) A 3-vector

    Returns:
      W: (N, 3, 3) A skew matrix such that W @ v == w x v
    """
    zeros = torch.zeros(w.shape[0], device=w.device)
    w_skew_list = [zeros, -w[:, 2], w[:, 1],
                   w[:, 2], zeros, -w[:, 0],
                   -w[:, 1], w[:, 0], zeros]
    w_skew = torch.stack(w_skew_list, dim=-1).reshape(-1, 3, 3)
    return w_skew


def rp_to_se3(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotation and translation to homogeneous transform.

    Args:
      R: (3, 3) An orthonormal rotation matrix.
      p: (3,) A 3-vector representing an offset.

    Returns:
      X: (4, 4) The homogeneous transformation matrix described by rotating by R
        and translating by p.
    """
    bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=R.device).repeat(R.shape[0], 1, 1)
    transform = torch.cat([torch.cat([R, p], dim=-1), bottom_row], dim=1)

    return transform


def exp_so3(w: torch.Tensor, theta: float) -> torch.Tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

    Args:
      w: (3,) An axis of rotation.
      theta: An angle of rotation.

    Returns:
      R: (3, 3) An orthonormal rotation matrix representing a rotation of
        magnitude theta about axis w.
    """
    W = skew(w)
    identity = torch.eye(3).unsqueeze(0).repeat(W.shape[0], 1, 1).to(W.device)
    W_sqr = torch.bmm(W, W)  # batch matrix multiplication
    R = identity + torch.sin(theta.unsqueeze(-1)) * W + (1.0 - torch.cos(theta.unsqueeze(-1))) * W_sqr
    return R


def exp_se3(S: torch.Tensor, theta: float) -> torch.Tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.88.

    Args:
      S: (6,) A screw axis of motion.
      theta: Magnitude of motion.

    Returns:
      a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
        motion of magnitude theta about S for one second.
    """
    w, v = torch.split(S, 3, dim=-1)
    W = skew(w)
    R = exp_so3(w, theta)

    identity = torch.eye(3).unsqueeze(0).repeat(W.shape[0], 1, 1).to(W.device)
    W_sqr = torch.bmm(W, W)
    theta = theta.view(-1, 1, 1)

    p = torch.bmm((theta * identity + (1.0 - torch.cos(theta)) * W + (theta - torch.sin(theta)) * W_sqr),
                  v.unsqueeze(-1))
    return rp_to_se3(R, p)


def to_homogenous(v: torch.Tensor) -> torch.Tensor:
    """Converts a vector to a homogeneous coordinate vector by appending a 1.

    Args:
        v: A tensor representing a vector or batch of vectors.

    Returns:
        A tensor with an additional dimension set to 1.
    """
    return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)


def from_homogenous(v: torch.Tensor) -> torch.Tensor:
    """Converts a homogeneous coordinate vector to a standard vector by dividing by the last element.

    Args:
        v: A tensor representing a homogeneous coordinate vector or batch of homogeneous coordinate vectors.

    Returns:
        A tensor with the last dimension removed.
    """
    return v[..., :3] / v[..., -1:]


def compute_combined_rigidity_distance(trajectories1: torch.Tensor, trajectories2=None, alpha=0.5, return_full=False, batch_size=8192) -> torch.Tensor:
    """
    Combine pairwise rigidity std and spatial distance.

    Args:
        trajectories (torch.Tensor): (N, T, 3)
        alpha (float): weight for rigidity (0~1). 1 = only rigidity, 0 = only spatial

    Returns:
        combined_dist (torch.Tensor): (N, N) combined distance matrix
    """
    N, T, _ = trajectories1.shape
    trajectories2 = trajectories1 if trajectories2 is None else trajectories2
    M = trajectories2.shape[0]

    # Allocate the result tensors directly to save memory
    rigidity_std = torch.zeros((N, M), dtype=trajectories1.dtype, device=trajectories1.device)
    spatial_dist = torch.zeros((N, M), dtype=trajectories1.dtype, device=trajectories1.device)

    # Process trajectories1 in batches to reduce memory usage
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # Select a batch from trajectories1
        batch_trajectories1 = trajectories1[start:end]  # (B, T, 3)

        # Efficiently compute pairwise distances for the current batch
        traj_i = batch_trajectories1.unsqueeze(1)  # (B, 1, T, 3)
        traj_j = trajectories2.unsqueeze(0)        # (1, M, T, 3)

        # Distance calculation over time: (B, T, M)
        dist_time = torch.norm(traj_i - traj_j, dim=-1)  # (B, M, T, 3) -> (B, M, T)

        # Calculate spatial distance (mean across time)
        spatial_dist_batch = torch.mean(dist_time, dim=-1)  # (B, M)

        # Calculate rigidity standard deviation (Welford's method)\
        rigidity_std_batch = torch.std(dist_time, dim=-1)      # (B, M)

        # Store the results in the final tensors
        rigidity_std[start:end] = rigidity_std_batch
        spatial_dist[start:end] = spatial_dist_batch

    combined = alpha * rigidity_std + (1 - alpha) * spatial_dist

    if return_full:
        return rigidity_std, spatial_dist
    else:
        return combined
    

@torch.no_grad()
def grouping_stage_one(gaussians, scene, dataset, deform, gnum=50, early_group_flag=True, nframes_max=-1):
    
    # ### 0.1 Semantic grouping
    # gaussians.test_cluster()

    ### 0.2 Static region identification

    ### 1. Initialize control pts groups labels
    # For all gaussians, 
        # 1.1 Furthest pts sampling: set control pts (Nc=500)
        # 1.2 Compute pairwise rigidity std (N, Nc), mean distance
        #   If OOM, Run KNN, for each pts, select top-3 control points
        # 1.3 classify pts into groups with high similarity

    ### 2. Initialze rigid transformation with trained deform net
    # For all the groups,
        # 2.1 Gather the per-point trajectory
        # 2.2 Initialize group [R|T]_t at each frame (w.r.t frame_0)
        
        # 2.a (optional) Choose center point (under sem-label based method, we may do not need this) ?
        # 2.b (optional) test rigidity
        # 2.c (optional) cycle consistency loss


    ### 1. Initialize control pts groups labels

    all_gs_xyz = gaussians.get_xyz

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_stack.sort(key=lambda obj: obj.fid)   # sort the views by time stamp
    vnum = len(viewpoint_stack)
    time_stamps = torch.tensor([view.fid for view in viewpoint_stack]).to(all_gs_xyz.device)

    vid_base = 0    # view 0 as basis
    v_cam = viewpoint_stack[vid_base]
    fid_base = v_cam.fid
    N = all_gs_xyz.shape[0]
    time_input = fid_base.unsqueeze(0).expand(N, -1)
    d_xyz, _, _ = deform.step(all_gs_xyz.detach(), time_input)
    
    if dataset.is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D_0 = all_gs_xyz
        else:
            means3D_0 = from_homogenous(torch.bmm(d_xyz, to_homogenous(all_gs_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D_0 = all_gs_xyz + d_xyz
    
    ### Initialize groups via Furthest pts sampling
    node_num = min(gnum, means3D_0.shape[0])    # in case gnum > pointnum (means3D_0.shape[0])
    init_nodes_idx = farthest_point_sample(means3D_0.detach()[None], node_num)[0]

    ### Sample trajectory for all gaussians
    def inference_deform_net(pts, views, deform_net, nframes):
        """
        pts: [N, 3]
        """
        flow_list = []
        vnum = len(views)
        vid_list = torch.linspace(0, vnum-1, nframes).long()
        # vid_list = torch.arange(0, vnum, step=vnum//nframes)
        if vid_list[-1] != vnum-1:      # guarantee start and end time are included
            vid_list = torch.cat([vid_list, torch.tensor([vnum-1])]).long()
        for vid in vid_list:
            v_cam = views[vid]
            fid = v_cam.fid
            N = pts.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            d_xyz, _, _ = deform_net.step(pts.detach(), time_input)
            
            if dataset.is_6dof:
                if torch.is_tensor(d_xyz) is False:
                    means3D = pts
                else:
                    means3D = from_homogenous(torch.bmm(d_xyz, to_homogenous(pts).unsqueeze(-1)).squeeze(-1))
            else:
                means3D = pts + d_xyz
            flow_list.append(means3D)
        flow_list = torch.stack(flow_list, dim=1)     # (N, T, 3)
        return flow_list

    pts_list = inference_deform_net(pts=all_gs_xyz, views=viewpoint_stack, deform_net=deform, nframes=8)

    # compute_pairwise_rigidity_std
    rigidity = compute_combined_rigidity_distance(pts_list, pts_list[init_nodes_idx], alpha=0.5)  # (N, Nc)

    label_assignment = torch.argmin(rigidity, dim=1)
    label_assignment_all = torch.argsort(rigidity, dim=1)    # (Np, Ng)
    labels_uni = torch.unique(label_assignment)

    cmap = np.random.randint(0, 256, [labels_uni.shape[0], 3])
    cmap[-1, :] = 0

    rgb = cmap[label_assignment.detach().cpu().numpy(), :]
    ply_name = os.path.join(scene.model_path, "group_label_f0.ply")
    save_ply_cluster(ply_name, xyz=pts_list[:, 0, :].detach().cpu().numpy(), rgb=(rgb).astype(np.uint8))


    ### 2. Initialize rigid transformation with trained deform net

    # For all the groups,
        # 2.1 Gather the per-point trajectory
        # 2.2 Initialize group [R|T]_t at each frame (w.r.t frame_0)

    gflow_means_list = []
    gflow_list = []
    gflowse3_list = []
    
    max_pnum_per_group = 100

    max_p = max_pnum_per_group
    gmask_all = label_assignment.unsqueeze(0) == labels_uni.unsqueeze(1)   # (G, Npts)
    gcount = gmask_all.sum(dim=1)   # (G)
    
    g_xyz_stack = []
    g_pnum_stack = torch.zeros(node_num, dtype=torch.int64, device=all_gs_xyz.device)
    for i in range(node_num):
        g_pnum = gcount[i]
        if g_pnum < max_p:  # max group pts number
            g_xyz = all_gs_xyz[gmask_all[i, :]]
        else:
            chosen_p = torch.randint(g_pnum, (max_p,))
            g_xyz = all_gs_xyz[gmask_all[i, :]][chosen_p]
            g_pnum = max_p
        g_xyz_stack.append(g_xyz)
        g_pnum_stack[i] = g_pnum
    g_xyz_stack = torch.cat(g_xyz_stack, dim=0)

    ### limit time step number (for compressing)
    if nframes_max > 0:
        nframes = min(vnum, nframes_max)
        if vnum > nframes_max:
            vid_list = torch.linspace(0, vnum-1, nframes).long()
            # vid_list = torch.arange(0, vnum, step=vnum//nframes)
            if vid_list[-1] != vnum-1:      # guarantee start and end time are included
                vid_list = torch.cat([vid_list, torch.tensor([vnum-1])]).long()
            time_stamps = time_stamps[vid_list]
    else:
        nframes = vnum

    gflow_all = inference_deform_net(pts=g_xyz_stack, views=viewpoint_stack, deform_net=deform, nframes=nframes)     # (Np_samp, T, 3)

    cnt = 0
    for i in tqdm.trange(node_num):
        gflow = gflow_all[cnt:cnt+g_pnum_stack[i], :, :]    # (Np_one_group, T, 3)

        gflow_mean = gflow.mean(dim=0)     # (T, 3)
        gflow_means_list.append(gflow_mean)
        
        # gflow_lie, gflow_se3 = compute_frame_to_frame_lie(gflow)   # (T-1, 6)
        gflow_lie, gflow_se3 = compute_frame_to_frame_lie_batch(gflow)   # (T-1, 6) faster version

        gflow_list.append(gflow_lie)
        gflowse3_list.append(gflow_se3)

        cnt += g_pnum_stack[i]
    
    gflow_means_list = torch.stack(gflow_means_list, dim=0)   # (Ng, T, 3)
    gflow_list = torch.stack(gflow_list, dim=0)     # optimizable parameters
    gflowse3_list = torch.stack(gflowse3_list, dim=0)   # (Ng, T-1, 3, 4)

    print("Grouping Done! \t group number:", gflow_means_list.shape[0], "\t tstep number:", gflow_means_list.shape[1])

    return_dict = {"means3D_0": means3D_0, 
              "gflow_list": gflow_list, 
              "gflowse3_list": gflowse3_list,
              "time_stamps": time_stamps, 
              "label_assignment": label_assignment, 
              "label_assignment_all": label_assignment_all,
              "gflow_means": gflow_means_list}
    return return_dict


class GroupFlowModel():
    def __init__(self):
        self.spatial_lr_scale = 0.00001
        self.version = 1
    
    def set_model(self, gflow_dict, training_args):
        self.gflow = torch.nn.Parameter(gflow_dict["gflow_list"].requires_grad_(True))    # (Ng, T-1, 6)
        self.gfmeans = gflow_dict["gflow_means"]  # (Ng, T, 3)
        self.time_stamps = gflow_dict["time_stamps"]  # float: (T)
        self.labels = gflow_dict["label_assignment"]    # int: (Np,)
        self.group_num = self.gflow.shape[0]
        self.frame_num = self.gflow.shape[1] + 1

        self.set_optimizer(training_args)
    
    def set_optimizer(self, training_args):
        l = [
                {
                    'params':[self.gflow], 
                    'lr': 1e-4, 
                    'name': 'group_lie'
                }
            ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.gflow_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "gflow/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save([self.gflow, self.time_stamps, self.labels], os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            from utils.system_utils import searchForMaxIteration
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "gflow/iteration_{}/deform.pth".format(loaded_iter))

        gflow, time_stamps, labels = torch.load(weights_path, weights_only=False)

        self.gflow = torch.nn.Parameter(gflow.requires_grad_(True))    # (Ng, T-1, 6)
        self.time_stamps = time_stamps  # float: (T)
        self.labels = labels    # int: (Np,)
        self.group_num = self.gflow.shape[0]
        self.frame_num = self.gflow.shape[1] + 1

    def step_vid(self, vid):
        """
        vid: frame index (int)
        """
        if vid == 0:   # no deform applied
            return 0
        else:
            # lie = self.gflow[:, time, :][labels, :]   # (Np, 6)
            lie = self.gflow[:, vid-1, :]   # (Ng, 6)
            R = exp_so3_batch(lie[:, :3])   # (Ng, 3, 3)
            t = lie[:, 3:, None] # (Ng, 3)

            bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=R.device).repeat(R.shape[0], 1, 1)
            gtransform = torch.cat([torch.cat([R, t], dim=-1), bottom_row], dim=1)   # (Ng, 4, 4)

            ptransform = gtransform[self.labels, :, :]  # (Np, 4, 4)

            return ptransform
    
    def precompute_interp(self):
        delta_w_list = []
        R_floor_list = []
        for i in range(0, self.frame_num-1):
            lie_ceil = self.gflow[:, i, :]
            if i == 0:
                lie_floor = torch.zeros_like(lie_ceil)
            else:
                lie_floor = self.gflow[:, i-1, :]
            R_floor = exp_so3_batch(lie_floor[:, :3])   # (Ng, 3, 3)
            R_ceil = exp_so3_batch(lie_ceil[:, :3])     # (Ng, 3, 3)
            delta_R = R_floor.transpose(1,2) @ R_ceil       # (Ng, 3, 3)
            delta_w = log_so3_batch(delta_R)                # (Ng, 3)
            delta_w_list.append(delta_w)
            R_floor_list.append(R_floor)
        self.delta_w_list = torch.stack(delta_w_list, dim=1) # (Ng, T-1, 3)
        self.R_floor_list = torch.stack(R_floor_list, dim=1) # (Ng, T-1, 3, 3)

    def step_t(self, t, use_precomputed_interp=False):
        """
        t: frame stamp (float)
        """
        if t == 0:   # no deform applied
            return 0
        else:
            vid_floor = torch.argwhere(self.time_stamps - t < 0)[-1].squeeze()
            t_floor = self.time_stamps[vid_floor]
            t_ceil = self.time_stamps[vid_floor+1]
            ratio = (t - t_floor) / (t_ceil - t_floor)

            lie_ceil = self.gflow[:, vid_floor, :]   # (Ng, 6)
            if vid_floor == 0:
                lie_floor = torch.zeros_like(lie_ceil)
            else:
                lie_floor = self.gflow[:, vid_floor-1, :]   # (Ng, 6)

            if use_precomputed_interp:
                delta_w = self.delta_w_list[:, vid_floor, :]
                R_floor = self.R_floor_list[:, vid_floor, :, :]
            else:
                R_floor = exp_so3_batch(lie_floor[:, :3])   # (Ng, 3, 3)
                R_ceil = exp_so3_batch(lie_ceil[:, :3])     # (Ng, 3, 3)
                delta_R = R_floor.transpose(1,2) @ R_ceil       # (Ng, 3, 3)
                delta_w = log_so3_batch(delta_R)                # (Ng, 3)

            inference_R = exp_so3_batch(ratio * delta_w)    # (Ng, 3, 3)
            R = R_floor @ inference_R   # (Ng, 3, 3)

            t = lie_floor[:, 3:, None] * (1 - ratio) + lie_ceil[:, 3:, None] * (ratio) # (Ng, 3)

            bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=R.device).repeat(R.shape[0], 1, 1)
            gtransform = torch.cat([torch.cat([R, t], dim=-1), bottom_row], dim=1).contiguous()   # (Ng, 4, 4)

            ptransform = gtransform[self.labels, :, :]

            return ptransform


class GroupFlowModel_v2():
    """
    Translate version: borrow idea from SC-GS
    u' = R @ (u-p) + p + T
    p: control pt coord
    u: inferenced pt
    """
    def __init__(self):
        self.spatial_lr_scale = 0.00001
        self.version = 2

        ### Compile slow functions

        self.rotation_matrix_to_quaternion_fast = rotation_matrix_to_quaternion_fast
        self.quaternion_multiply_fast = quaternion_multiply_fast
        self.exp_so3_batch = exp_so3_batch
        self.log_so3_batch = log_so3_batch

    def set_model(self, gflow_dict, training_args, scene_scale=1.0):

        self.time_stamps = gflow_dict["time_stamps"]  # float: (T)
        self.labels = gflow_dict["label_assignment"]    # int: (Np,)

        gflow_lie = gflow_dict["gflow_list"]    # (Ng, T-1, 6)
        self.gf_rotation = torch.nn.Parameter(gflow_lie[:,:,:3].requires_grad_(True))      # (Ng, T-1, 3): NOTE: actually orientation

        ### initialize translate with use group means 
        gflow_means = gflow_dict["gflow_means"]
        self.gf_translation = torch.nn.Parameter((gflow_means[:, 1:, :] - gflow_means[:, [0], :]).requires_grad_(True))  # (Ng, T-1, 3)

        self.gf_nodes_xyz = torch.nn.Parameter(gflow_means[:, 0, :].requires_grad_(True))      # (Ng, 3), at the first frame

        self.group_num = self.gf_translation.shape[0]
        self.frame_num = self.gf_translation.shape[1] + 1

        self.gflow_local_rot = training_args.gflow_local_rot
        self.LBS_flag = training_args.LBS_flag
        self.annealing_lr_flag = training_args.gflow_annealing_lr_flag
        self.gflow_local_rot_for_train = training_args.gflow_local_rot_for_train

        # if self.LBS_flag:
        self.scene_scale = scene_scale
        self.gf_nodes_radius = torch.nn.Parameter(0.1 * scene_scale * torch.ones(self.group_num).cuda().requires_grad_(True))      # (Ng, 3), at the first frame

        self.set_optimizer(training_args)
    
    def set_optimizer(self, training_args):
        l = [
                {
                    'params':[self.gf_nodes_xyz], 
                    'lr': training_args.gflow_xyz_lr,   #1e-5, 
                    'name': 'group_xyz'
                },
                {
                    'params':[self.gf_rotation], 
                    'lr': training_args.gflow_rotation_lr,  # training_args.rotation_lr,# * 1e-4, 
                    'name': 'group_rotation'
                },
                {
                    'params':[self.gf_translation], 
                    'lr': training_args.gflow_translation_lr,   # training_args.translation_lr,# * 1e-4, 
                    'name': 'group_translation'
                },
                {
                    'params':[self.gf_nodes_radius], 
                    'lr': training_args.gflow_radius_lr, # training_args.scaling_lr,# * 1e-4, 
                    'name': 'group_radius'
                }
            ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.gflow_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            lr = self.gflow_scheduler_args(iteration)
            param_group['lr'] = lr
            return lr

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "gflow/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        flags = [self.LBS_flag, self.gflow_local_rot, self.gflow_local_rot_for_train, self.annealing_lr_flag]
        torch.save([self.gf_nodes_xyz, self.gf_rotation, self.gf_translation, self.gf_nodes_radius, self.time_stamps, self.labels, flags], os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            from utils.system_utils import searchForMaxIteration
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "gflow/iteration_{}/deform.pth".format(loaded_iter))

        gf_nodes_xyz, gf_rotation, gf_translation, gf_nodes_radius, time_stamps, labels, flags = torch.load(weights_path, weights_only=False)

        self.gf_nodes_xyz = torch.nn.Parameter(gf_nodes_xyz.requires_grad_(True))    # (Ng, 3)
        self.gf_rotation = torch.nn.Parameter(gf_rotation.requires_grad_(True))    # (Ng, T-1, 3)
        self.gf_translation = torch.nn.Parameter(gf_translation.requires_grad_(True))    # (Ng, T-1, 3)
        self.gf_nodes_radius = torch.nn.Parameter(gf_nodes_radius.requires_grad_(True))      # (Ng, 3), at the first frame
        self.time_stamps = time_stamps  # float: (T)
        self.labels = labels    # int: (Np,)
        self.group_num = self.gf_translation.shape[0]
        self.frame_num = self.gf_translation.shape[1] + 1

        self.LBS_flag, self.gflow_local_rot, self.gflow_local_rot_for_train, self.annealing_lr_flag = flags

    def precompute_interp(self, x=None):
        delta_w_list = []
        R_floor_list = []
        for i in range(0, self.frame_num-1):
            lie_ceil = self.gf_rotation[:, i, :]
            if i == 0:
                lie_floor = torch.zeros_like(lie_ceil)
            else:
                lie_floor = self.gf_rotation[:, i-1, :]
            R_floor = exp_so3_batch(lie_floor)   # (Ng, 3, 3)
            R_ceil = exp_so3_batch(lie_ceil)     # (Ng, 3, 3)
            delta_R = R_floor.transpose(1,2) @ R_ceil       # (Ng, 3, 3)
            delta_w = log_so3_batch(delta_R)                # (Ng, 3)
            delta_w_list.append(delta_w)
            R_floor_list.append(R_floor)
        self.delta_w_list = torch.stack(delta_w_list, dim=1) # (Ng, T-1, 3)
        self.R_floor_list = torch.stack(R_floor_list, dim=1) # (Ng, T-1, 3, 3)

        if self.LBS_flag and x is not None:
            # Pre-compute for LBS
            Knn = 5
            self.dist_all = torch.norm(x[:, None, :] - self.gf_nodes_xyz[None, :, :], dim=2) # (Np, Ng) distance under canonical space
            self.labels_all = torch.argsort(self.dist_all, dim=1)[:, :Knn]    # (Np, K)

            dist = torch.gather(self.dist_all, dim=1, index=self.labels_all)  # (Np, K)
            sigma = self.gf_nodes_radius[self.labels_all]               # (Np, K)
            weights = torch.exp( -dist / (2*sigma) )                    # (Np, K)
            self.weights = weights / weights.sum(dim=1, keepdim=True)        # (Np, K)
        
        ### Pre-compile functions for faster inference
        self.rotation_matrix_to_quaternion_fast = torch.compile(rotation_matrix_to_quaternion_fast, mode="reduce-overhead", fullgraph=True, dynamic=False)
        self.quaternion_multiply_fast = torch.compile(quaternion_multiply_fast, mode="reduce-overhead", fullgraph=True, dynamic=False)
        self.exp_so3_batch = torch.compile(exp_so3_batch, mode="reduce-overhead", fullgraph=True, dynamic=False)
        self.log_so3_batch = torch.compile(log_so3_batch, mode="reduce-overhead", fullgraph=True, dynamic=False)

        ### Warm-up: The first call will be slow, 
        # but in `render.py` if you render_with_save and then render_test_speed, render_with_save will be a warm-up step natually
        # R = torch.randn([1000, 3, 3])
        # self.rotation_matrix_to_quaternion_fast(R)
    
    def step_t_func(self, t, use_precomputed_interp=False):
        """
        t: frame stamp (float)
        """
        if t <= self.time_stamps[0]:   # no deform applied
            return 0
        else:
            # import pdb
            # pdb.set_trace()

            t = torch.clip(t, self.time_stamps[0], self.time_stamps[-1] - 1e-5)

            vid_floor = torch.argwhere(self.time_stamps - t < 0)[-1].squeeze()
            t_floor = self.time_stamps[vid_floor]
            t_ceil = self.time_stamps[vid_floor+1]
            ratio = (t - t_floor) / (t_ceil - t_floor)

            lie_ceil = self.gf_rotation[:, vid_floor, :]    # (Ng, 3)
            trans_ceil = self.gf_translation[:, vid_floor, :]  # (Ng, 3)
            if vid_floor == 0:
                lie_floor = torch.zeros_like(lie_ceil)
                trans_floor = torch.zeros_like(trans_ceil)
            else:
                lie_floor = self.gf_rotation[:, vid_floor-1, :]   # (Ng, 3)
                trans_floor = self.gf_translation[:, vid_floor-1, :]

            if use_precomputed_interp:
                delta_w = self.delta_w_list[:, vid_floor, :]
                R_floor = self.R_floor_list[:, vid_floor, :, :]
            else:
                R_floor = self.exp_so3_batch(lie_floor[:, :3]).clone()   # (Ng, 3, 3)
                R_ceil = self.exp_so3_batch(lie_ceil[:, :3])     # (Ng, 3, 3)
                delta_R = R_floor.transpose(1,2) @ R_ceil       # (Ng, 3, 3)
                delta_w = self.log_so3_batch(delta_R)                # (Ng, 3)

            inference_R = self.exp_so3_batch(ratio * delta_w)    # (Ng, 3, 3)
            R = R_floor @ inference_R   # (Ng, 3, 3)

            T = trans_floor * (1 - ratio) + trans_ceil * (ratio) # (Ng, 3)

            gtransform = torch.cat([R, T[:,:,None]], dim=-1)      # (Ng, 3, 4)

            return gtransform
        
    def step_t(self, x, t, use_precomputed_interp=False, Q=None):
        LBS_flag = self.LBS_flag
        if t <= self.time_stamps[0]:
            if Q is None:
                return 0
            else:
                return 0, 0
        else:
            gtransform = self.step_t_func(t, use_precomputed_interp)    # (Ng, 3, 4)

            if LBS_flag:    ### Apply Linear Blend Skinning (KNN)
                Knn = 5
                if use_precomputed_interp:
                    dist_all = self.dist_all
                    labels_all = self.labels_all
                    weights = self.weights
                else:
                    dist_all = torch.norm(x[:, None, :] - self.gf_nodes_xyz[None, :, :], dim=2) # (Np, Ng) distance under canonical space
                    labels_all = torch.argsort(dist_all, dim=1)[:, :Knn]    # (Np, K)

                    dist = torch.gather(dist_all, dim=1, index=labels_all)  # (Np, K)
                    sigma = self.gf_nodes_radius[labels_all]                # (Np, K)
                    weights = torch.exp( -dist / (2*sigma) )            # (Np, K)
                    weights = weights / weights.sum(dim=1, keepdim=True)

                ptransform = gtransform[labels_all, :, :]    # (Np, K, 3, 4)
                R = ptransform[:, :, :, :3]     # (Np, K, 3, 3)
                T = ptransform[:, :, :, 3]      # (Np, K, 3)
                nodes = self.gf_nodes_xyz[labels_all, :]     # (Np, K, 3)
                d_xyz_K = (R @ (x[:, None, :] - nodes)[:,:,:,None]).squeeze() + nodes + T - x[:, None, :]   # (Np, K, 3)
                d_xyz = (weights[:,:,None] * d_xyz_K).sum(dim=1)    # (Np, 3)
            else:
                ptransform = gtransform[self.labels, :, :]                      # (Np, 3, 4)
                R = ptransform[:, :, :3]
                T = ptransform[:, :, 3]
                nodes = self.gf_nodes_xyz[self.labels, :]                       # (Np, 3)
                d_xyz = (R @ (x - nodes)[:,:,None]).squeeze() + nodes + T - x   # (Np, 3)

            ### Optional: Rotate per-gaussian orientation
            if (self.gflow_local_rot is True) and (Q is not None):
                ### 1. Get quaternion dQ from rotation matrix (Use compiled func to get faster speed)
                # dQ_groups = rotation_matrix_to_quaternion(gtransform[:, :, :3]) # (Ng, 3, 3) -> (Ng, 4)
                # dQ_groups = rotation_matrix_to_quaternion_fast(gtransform[:, :, :3]) # (Ng, 3, 3) -> (Ng, 4)
                dQ_groups = self.rotation_matrix_to_quaternion_fast(gtransform[:, :, :3]) # (Ng, 3, 3) -> (Ng, 4)
                ### 2. Map dQ to all gaussian points
                dQ = dQ_groups[self.labels, :]
                ### 3. Get rotation/pose after rotation (Use compiled func to get faster speed)
                # new_rotation = quaternion_multiply(dQ, Q)
                # new_rotation = quaternion_multiply_fast(dQ, Q)
                new_rotation = self.quaternion_multiply_fast(dQ, Q)
                ### 4. Normalize to get difference/d_rotation as input of De3DGS's gaussian_render/__init__.py
                d_rotation = torch.nn.functional.normalize(new_rotation) - Q

                if torch.isnan(d_rotation).any().item() or torch.isinf(d_rotation).any().item():
                    import pdb
                    pdb.set_trace()
                return d_xyz, d_rotation
            
            else:
                return d_xyz


def exp_so3_batch(w):
    """
    Exponential map from so(3) to SO(3)
    Input: w (B, 3) axis-angle vector
    Output: R (B, 3, 3) rotation matrices
    """
    theta = torch.norm(w, dim=-1, keepdim=True).clamp(min=1e-8)
    k = w / theta  # (B, 3)
    K = hat(k)     # (B, 3, 3)

    I = torch.eye(3, device=w.device).expand(w.shape[0], 3, 3)
    R = I + torch.sin(theta)[..., None] * K + (1 - torch.cos(theta)[..., None]) * K @ K
    return R

def log_so3_batch(R):
    """
    Logarithm map from SO(3) to so(3)
    Input: R (B, 3, 3)
    Output: w (B, 3)
    """
    cos_theta = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1) / 2
    cos_theta = cos_theta.clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)

    w_hat = (R - R.transpose(-1, -2)) / (2 * torch.sin(theta)[:, None, None].clamp(min=1e-6))
    w = vee(w_hat) * theta[:, None]

    return w

def hat(w):
    """
    Input: w (B, 3)
    Output: skew-symmetric matrix (B, 3, 3)
    """
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    O = torch.zeros_like(wx)
    return torch.stack([
        torch.stack([O, -wz, wy], dim=-1),
        torch.stack([wz, O, -wx], dim=-1),
        torch.stack([-wy, wx, O], dim=-1),
    ], dim=1)

def vee(W):
    """
    Input: skew-symmetric (B, 3, 3)
    Output: vector (B, 3)
    """
    return torch.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], dim=-1)

def compute_frame_to_frame_lie(pts: torch.Tensor):
    """
    Given pts of shape (N, T, 3), compute per-frame rigid transform:
    Rotation as Lie algebra vector (T-1, 3), and translation (T-1, 3)

    Returns:
        w_list: (T-1, 3) Lie algebra axis-angle rotation vectors
        t_list: (T-1, 3) translation vectors
    """
    N, T, _ = pts.shape
    R_list = []
    w_list = []
    t_list = []
    error = 0

    for t in range(T - 1):
        A = pts[:, 0]       # (N, 3)    # relative to the first frame
        B = pts[:, t + 1]   # (N, 3)

        # Center the point clouds
        centroid_A = A.mean(dim=0)
        centroid_B = B.mean(dim=0)
        A_centered = A - centroid_A
        B_centered = B - centroid_B

        # Compute rotation via SVD
        H = A_centered.T @ B_centered
        U, _, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T
        if torch.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Convert R to Lie algebra (axis-angle vector)
        R_batch = R[None, :, :]  # (1, 3, 3)
        w = log_so3_batch(R_batch)[0]  # (3,)

        # Compute translation
        t_vec = centroid_B - R @ centroid_A  # (3,)

        R_list.append(R)
        w_list.append(w)
        t_list.append(t_vec)

        pts_pred = (R @ A.T).T + t_vec  # inference rigid transformation
        error += torch.norm(pts_pred - B, dim=1).mean() # compare with the next frame

    w_stack = torch.stack(w_list, dim=0)    # (T-1, 3)
    t_stack = torch.stack(t_list, dim=0)    # (T-1, 3)
    lie_stack = torch.cat([w_stack, t_stack], dim=-1)  # (T-1, 6)

    R_stack = torch.stack(R_list, dim=0)    # (T-1, 3, 3)
    Se3_stack = torch.cat([R_stack, t_stack[:,:,None]], dim=-1) # (T-1, 3, 4)

    return lie_stack, Se3_stack


def compute_frame_to_frame_lie_batch(pts: torch.Tensor):
    """
    Vectorized 0/1-style rigid fit between each consecutive frame.

    Args:
        pts: Tensor of shape (N, T, 3)

    Returns:
        lie_stack: (T-1, 6) — [wx, wy, wz, tx, ty, tz] for each frame step
        Se3_stack: (T-1, 3, 4) — [R | t] for each frame step
    """
    N, T, _ = pts.shape
    assert pts.dim() == 3 and pts.size(-1) == 3

    # 1) form A, B for each step: shapes (N, T-1, 3)
    A = pts[:, [0], :]     # frames 0..T-2
    B = pts[:,  1:, :]     # frames 1..T-1

    # 2) compute centroids per frame: (T-1, 3)
    centroid_A = A.mean(dim=0)
    centroid_B = B.mean(dim=0)

    # 3) center the point clouds: broadcast over N
    A_c = A - centroid_A.unsqueeze(0)   # (N, T-1, 3)
    B_c = B - centroid_B.unsqueeze(0)   # (N, T-1, 3)

    # 4) cross-covariance H for each step via einsum: (T-1, 3, 3)
    #    H[t]_{ij} = sum_n A_c[n,t,i] * B_c[n,t,j]
    H = torch.einsum('nti,ntj->tij', A_c, B_c)

    # 5) batch SVD on H: U, S, Vt each (T-1, 3, 3)
    U, S, Vt = torch.linalg.svd(H)

    # 6) initial R = Vt^T @ U^T  (batched matmul)
    R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)

    # 7) fix reflections (det(R)<0) by flipping last row of Vt where needed
    detR = torch.linalg.det(R)                  # (T-1,)
    mask = detR < 0
    if mask.any():
        # flip the last row of Vt for those batches
        Vt[mask, -1, :] *= -1
        R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)

    # 8) convert R to axis-angle (Lie vector) in one batch call
    #    assumes you have a log_so3_batch function:
    #    log_so3_batch(R) -> (T-1, 3)
    w = log_so3_batch(R)  # (T-1, 3)

    # 9) compute translations: t = centroid_B - R @ centroid_A
    #    centroid_A is (T-1,3), so we do a batched matmul:
    # t = centroid_B - (R @ centroid_A.unsqueeze(-1)).squeeze(-1)  # (T-1,3)
    # t = t - (centroid_A - (R @ centroid_A.unsqueeze(-1)).squeeze(-1))   # (T-1,3)
    t = centroid_B - centroid_A     # (T-1,3)

    # 10) pack outputs
    lie_stack = torch.cat([w, t], dim=-1)          # (T-1, 6)
    Se3_stack = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # (T-1, 3, 4)

    return lie_stack, Se3_stack


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def save_ply_cluster(path, xyz, rgb):
    """
    xyz: np.float32
    rgb: np.uint8
    """

    from utils.system_utils import mkdir_p
    from plyfile import PlyData, PlyElement

    mkdir_p(os.path.dirname(path))

    dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z']] + [(attribute, 'u1') for attribute in ['red', 'green', 'blue']]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def rotation_matrix_to_quaternion(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Converts a batch of rotation matrices (..., 3, 3) to quaternions (..., 4) in [w, x, y, z] format,
    with safeguards against sqrt-of-zero and division-by-zero instabilities.
    """
    assert R.shape[-2:] == (3, 3), "Input must be a batch of 3x3 matrices"

    # extract components
    m00 = R[..., 0, 0]; m01 = R[..., 0, 1]; m02 = R[..., 0, 2]
    m10 = R[..., 1, 0]; m11 = R[..., 1, 1]; m12 = R[..., 1, 2]
    m20 = R[..., 2, 0]; m21 = R[..., 2, 1]; m22 = R[..., 2, 2]

    trace = m00 + m11 + m22

    # Prepare output
    q = torch.empty(R.shape[:-2] + (4,), dtype=R.dtype, device=R.device)

    # Case 1: trace > 0
    cond1 = trace > 0
    if cond1.any():
        s1 = (trace[cond1] + 1.0).clamp(min=0.0).sqrt() * 2.0
        s1 = s1 + eps  # never zero
        qw = 0.25 * s1
        qx = (m21[cond1] - m12[cond1]) / s1
        qy = (m02[cond1] - m20[cond1]) / s1
        qz = (m10[cond1] - m01[cond1]) / s1
        q[cond1] = torch.stack([qw, qx, qy, qz], dim=-1)

    # Case 2: m00 is largest diagonal and not cond1
    cond2 = (~cond1) & (m00 > m11) & (m00 > m22)
    if cond2.any():
        s2 = (1.0 + m00[cond2] - m11[cond2] - m22[cond2]).clamp(min=0.0).sqrt() * 2.0
        s2 = s2 + eps
        qw = (m21[cond2] - m12[cond2]) / s2
        qx = 0.25 * s2
        qy = (m01[cond2] + m10[cond2]) / s2
        qz = (m02[cond2] + m20[cond2]) / s2
        q[cond2] = torch.stack([qw, qx, qy, qz], dim=-1)

    # Case 3: m11 is largest diagonal
    cond3 = (~cond1) & (~cond2) & (m11 > m22)
    if cond3.any():
        s3 = (1.0 + m11[cond3] - m00[cond3] - m22[cond3]).clamp(min=0.0).sqrt() * 2.0
        s3 = s3 + eps
        qw = (m02[cond3] - m20[cond3]) / s3
        qx = (m01[cond3] + m10[cond3]) / s3
        qy = 0.25 * s3
        qz = (m12[cond3] + m21[cond3]) / s3
        q[cond3] = torch.stack([qw, qx, qy, qz], dim=-1)

    # Case 4: m22 is largest diagonal
    cond4 = (~cond1) & (~cond2) & (~cond3)
    if cond4.any():
        s4 = (1.0 + m22[cond4] - m00[cond4] - m11[cond4]).clamp(min=0.0).sqrt() * 2.0
        s4 = s4 + eps
        qw = (m10[cond4] - m01[cond4]) / s4
        qx = (m02[cond4] + m20[cond4]) / s4
        qy = (m12[cond4] + m21[cond4]) / s4
        qz = 0.25 * s4
        q[cond4] = torch.stack([qw, qx, qy, qz], dim=-1)

    # Final normalization to guard against any residual numerical drift
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=eps)

    return q


def rotation_matrix_to_quaternion_fast(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Vectorized, branchless conversion of (...,3,3) rotation matrices to quaternions (...,4) in [w,x,y,z].
    Avoids boolean-advanced indexing; enables kernel fusion and is friendlier to torch.compile/TorchScript.
    """
    assert R.shape[-2:] == (3,3)
    orig_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3).contiguous()  # [B,3,3]
    B = R.shape[0]

    m00 = R[:,0,0]; m01 = R[:,0,1]; m02 = R[:,0,2]
    m10 = R[:,1,0]; m11 = R[:,1,1]; m12 = R[:,1,2]
    m20 = R[:,2,0]; m21 = R[:,2,1]; m22 = R[:,2,2]

    trace = m00 + m11 + m22

    # Candidates:
    # t0 -> m00 largest; t1 -> m11 largest; t2 -> m22 largest; t3 -> trace largest
    t0 = 1.0 + m00 - m11 - m22
    t1 = 1.0 - m00 + m11 - m22
    t2 = 1.0 - m00 - m11 + m22
    t3 = 1.0 + trace

    # We use s = 2 * sqrt(t) as in the standard formulas
    def _s(x): 
        return 2.0 * torch.sqrt(torch.clamp(x, min=0.0)) + eps

    s0 = _s(t0)
    s1 = _s(t1)
    s2 = _s(t2)
    s3 = _s(t3)

    # Case 0 (m00 largest)
    q0w = (m21 - m12) / s0
    q0x = 0.25 * s0
    q0y = (m01 + m10) / s0
    q0z = (m02 + m20) / s0

    # Case 1 (m11 largest)
    q1w = (m02 - m20) / s1
    q1x = (m01 + m10) / s1
    q1y = 0.25 * s1
    q1z = (m12 + m21) / s1

    # Case 2 (m22 largest)
    q2w = (m10 - m01) / s2
    q2x = (m02 + m20) / s2
    q2y = (m12 + m21) / s2
    q2z = 0.25 * s2

    # Case 3 (trace largest)
    q3w = 0.25 * s3
    q3x = (m21 - m12) / s3
    q3y = (m02 - m20) / s3
    q3z = (m10 - m01) / s3

    # Stack candidates: [B, 4(cases), 4(comps)]
    Qc = torch.stack([
        torch.stack([q0w, q0x, q0y, q0z], dim=-1),
        torch.stack([q1w, q1x, q1y, q1z], dim=-1),
        torch.stack([q2w, q2x, q2y, q2z], dim=-1),
        torch.stack([q3w, q3x, q3y, q3z], dim=-1),
    ], dim=1)

    # Pick the best case per row with argmax over t0..t3
    idx = torch.argmax(torch.stack([t0, t1, t2, t3], dim=-1), dim=-1)  # [B]
    q = Qc[torch.arange(B, device=R.device), idx, :]  # [B,4]

    # Normalize (guards residual drift)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=eps)

    return q.reshape(orig_shape + (4,))


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    Both inputs are (..., 4) in [w, x, y, z] format
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def quaternion_multiply_fast(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # q = [w, x, y, z]
    w1 = q1[..., :1]
    v1 = q1[..., 1:]
    w2 = q2[..., :1]
    v2 = q2[..., 1:]

    w = w1 * w2 - (v1 * v2).sum(dim=-1, keepdim=True)
    v = w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=-1)
    return torch.cat([w, v], dim=-1)


def do_group_flow(gaussians, opt, dataset, scene, deform):

    print("Runing grouping ...")
    gflow_dict = grouping_stage_one(gaussians, scene, dataset, deform, gnum=opt.gflow_num, early_group_flag=False, nframes_max=opt.gflow_tnum_max)
    means3D_0 = gflow_dict["means3D_0"]

    ### Re-initialize optimizer with pts xyz at frame 0
    gaussians_xyz_old = gaussians._xyz.clone().detach()
    gaussians._xyz = torch.nn.Parameter(means3D_0.clone().detach().requires_grad_(True))
    gaussians.training_setup(opt)

    if opt.gflow_opt == 1:
        gflow_model = GroupFlowModel()
        gflow_model.set_model(gflow_dict, training_args=opt)
    elif opt.gflow_opt == 2:
        gflow_model = GroupFlowModel_v2()
        gflow_model.set_model(gflow_dict, training_args=opt, scene_scale=scene.cameras_extent)
    else:
        print("gflow_opt not defined, please check your config.")
        exit()

    debug = False
    if debug:
        frame_id = 50
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_stack.sort(key=lambda obj: obj.fid)

        ### 1. New Deform Network
        if opt.gflow_opt == 1:
            d_xyz = gflow_model.step_t(viewpoint_stack[frame_id].fid)   # 6-DoF: (Np, 4, 4), non-6-DoF: (Np, 3)
        elif opt.gflow_opt == 2:
            d_xyz = gflow_model.step_t(means3D_0, viewpoint_stack[frame_id].fid)

        if len(d_xyz.shape) > 2:    # 6dof
            means3D = from_homogenous(torch.bmm(d_xyz, to_homogenous(means3D_0).unsqueeze(-1)).squeeze(-1))
        else:
            means3D = means3D_0 + d_xyz

        A = means3D_0.clone().detach()
        pts_pred = means3D.clone().detach()
        rgb = ((A - A.min(0).values) / (A.max(0).values - A.min(0).values) * 255).detach().cpu().numpy()
        ply_name = os.path.join(scene.model_path, "debug_full_A.ply")
        gaussians.save_ply_cluster(ply_name, xyz=A.detach().cpu().numpy(), rgb=(rgb).astype(np.uint8))
        ply_name = os.path.join(scene.model_path, "debug_full_est.ply")
        gaussians.save_ply_cluster(ply_name, xyz=pts_pred.detach().cpu().numpy(), rgb=(rgb).astype(np.uint8))

        ### 2. Original Deform Network
        d_xyz, d_rotation, d_scaling = deform.step(gaussians_xyz_old.detach(), viewpoint_stack[frame_id].fid.unsqueeze(0).expand(means3D_0.shape[0], -1))
        if len(d_xyz.shape) > 2:    # 6dof
            means3D = from_homogenous(torch.bmm(d_xyz, to_homogenous(gaussians_xyz_old).unsqueeze(-1)).squeeze(-1))
        else:
            means3D = gaussians_xyz_old + d_xyz
        B = means3D.clone().detach()
        ply_name = os.path.join(scene.model_path, "debug_full_B.ply")
        gaussians.save_ply_cluster(ply_name, xyz=B.detach().cpu().numpy(), rgb=(rgb).astype(np.uint8))
    
    return gflow_model


def step_group_flow(gflow_model, opt, fid, gaussians, use_precomputed_interp=False):

    if gflow_model.version == 1:
        fix_gflow_flag = True
    elif gflow_model.version == 2:
        fix_gflow_flag = False

    if gflow_model.version == 1:
        d_rotation, d_scaling = 0.0, 0.0
        d_xyz = gflow_model.step_t(fid, use_precomputed_interp=use_precomputed_interp)
        if torch.is_tensor(d_xyz):
            if fix_gflow_flag:
                d_xyz = d_xyz.detach()
    elif gflow_model.version == 2:
        if gflow_model.gflow_local_rot == False:
            d_rotation, d_scaling = 0.0, 0.0                
            d_xyz = gflow_model.step_t(gaussians.get_xyz.detach(), fid, use_precomputed_interp=use_precomputed_interp)
            if torch.is_tensor(d_xyz):
                if fix_gflow_flag:
                    d_xyz = d_xyz.detach()
        else:
            d_scaling = 0.0
            d_xyz, d_rotation = gflow_model.step_t(gaussians.get_xyz.detach(), fid, Q=gaussians.get_rotation.detach(), use_precomputed_interp=use_precomputed_interp)
            if torch.is_tensor(d_xyz):
                if fix_gflow_flag:
                    d_xyz = d_xyz.detach()
                    d_rotation = d_rotation.detach()
                elif gflow_model.gflow_local_rot_for_train == False:   # if use gflow_local_rot as another gradient branch for training, if True, it would be slower. Default: False
                    d_rotation = d_rotation.detach()
    
    return d_xyz, d_rotation, d_scaling