import torch
from m3p2i_aip.utils import skill_utils
import m3p2i_aip.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from torch.profiler import record_function


# import sys
# sys.path.append('../../../../../mlp_learn/')
# from sdf.robot_sdf import RobotSdfCollisionNet
#
# params = {'device': 'cpu', 'dtype': torch.float32}
#
# # robot parameters
# DOF = 2
#
# # nn loading
# s = 256
# n_layers = 5
# skips = []
# fname = '%ddof_sdf_%dx%d_mesh.pt' % (DOF, s, n_layers)
# if skips == []:
#     n_layers -= 1
# nn_model = RobotSdfCollisionNet(in_channels=DOF + 3, out_channels=DOF, layers=[s] * n_layers, skips=skips)
# nn_model.load_weights('../../../mlp_learn/models/' + fname, params)
# nn_model.model.to(**params)
# nn_model.model_jit = nn_model.model
# nn_model.model_jit = torch.jit.script(nn_model.model_jit)
# nn_model.model_jit = torch.jit.optimize_for_inference(nn_model.model_jit)
# nn_model.update_aot_lambda()

class Objective(object):
    def __init__(self, cfg, nn_model):
        self.cfg = cfg
        self.multi_modal = cfg.multi_modal
        self.num_samples = cfg.mppi.num_samples
        self.half_samples = int(cfg.mppi.num_samples / 2)
        self.device = self.cfg.mppi.device
        self.pre_height_diff = cfg.pre_height_diff
        self.tilt_cos_theta = 0.5

        self.N_traj = 200
        #self.N_traj = 32
        self.nn_model = nn_model
        self.n_closest_obs = 4
        self.dst_thr = 0.5
        self.N_KERNEL_MAX = 50

        
        self.tensor_args = {'device': torch.device('cuda:0'), 'dtype': torch.float32}
        self.nn_model.allocate_gradients(self.N_traj*self.n_closest_obs, self.tensor_args)
        #self.nn_model.allocate_gradients(self.N_traj + self.N_KERNEL_MAX, self.tensor_args)


    def update_objective(self, task, goal):
        self.task = task
        #print("self.task", self.task)
        self.goal = goal if torch.is_tensor(goal) else torch.tensor(goal, device=self.device)

    def compute_cost(self, sim: wrapper):
        print("self.task", self.task)
        print("self.goal", self.goal)

        q0 = sim._dof_state[0]
        self.n_dof = q0.shape[0]

        self.ignored_links = [0, 1, 2]
        if self.n_dof < 7:
            self.ignored_links = []

        task_cost = 0
        if self.task == "navigation":
            task_cost = self.get_navigation_cost(sim)
        elif self.task == "push":
            return self.get_push_cost(sim, self.goal)
        elif self.task == "pull":
            return self.get_pull_cost(sim, self.goal)
        elif self.task == "push_pull":
            return torch.cat((self.get_push_cost(sim, self.goal)[:self.half_samples],
                              self.get_pull_cost(sim, self.goal)[self.half_samples:]), dim=0)
        elif self.task == "reach":
            return self.get_panda_reach_cost(sim, self.goal)
        elif self.task == "pick":
            task_cost = self.get_panda_pick_cost(sim, self.goal)
        elif self.task == "place":
            return self.get_panda_place_cost(sim)
        #return task_cost + self.get_motion_cost(sim)
        print(f"task_cost shape: {task_cost.shape}")

        get_motion_cost_1 = self.get_motion_cost_1(sim)
        print(f"motion_cost shape: {get_motion_cost_1.shape}")
        return get_motion_cost_1 + task_cost

    def get_navigation_cost(self, sim: wrapper):
        return torch.linalg.norm(sim.robot_pos - self.goal, axis=1)

    def calculate_dist(self, sim: wrapper, block_goal: torch.tensor):
        self.block_pos = sim.get_actor_position_by_name("box")[:, :2]  # x, y position
        robot_to_block = sim.robot_pos - self.block_pos
        block_to_goal = block_goal - self.block_pos

        robot_to_block_dist = torch.linalg.norm(robot_to_block, axis=1)
        block_to_goal_dist = torch.linalg.norm(block_to_goal, axis=1)

        self.dist_cost = robot_to_block_dist + block_to_goal_dist * 10
        self.cos_theta = torch.sum(robot_to_block * block_to_goal, 1) / (robot_to_block_dist * block_to_goal_dist)

    def get_push_cost(self, sim: wrapper, block_goal: torch.tensor):
        # Calculate dist cost
        self.calculate_dist(sim, block_goal)

        # Force the robot behind block and goal, align_cost is actually cos(theta)+1
        align_cost = torch.zeros(self.num_samples, device=self.device)
        align_cost[self.cos_theta > 0] = self.cos_theta[self.cos_theta > 0]

        return 3 * self.dist_cost + 1 * align_cost

    def get_pull_cost(self, sim: wrapper, block_goal: torch.tensor):
        self.calculate_dist(sim, block_goal)
        pos_dir = self.block_pos - sim.robot_pos
        robot_to_block_dist = torch.linalg.norm(pos_dir, axis=1)

        # True means the velocity moves towards block, otherwise means pull direction
        flag_towards_block = torch.sum(sim.robot_vel * pos_dir, 1) > 0

        # simulate a suction to the box
        suction_force = skill_utils.calculate_suction(self.cfg, sim)
        # Set no suction force if robot moves towards the block
        suction_force[flag_towards_block] = 0
        if self.multi_modal:
            suction_force[:self.half_samples] = 0
        sim.apply_rigid_body_force_tensors(suction_force)

        self.calculate_dist(sim, block_goal)

        # Force the robot to be in the middle between block and goal, align_cost is actually 1-cos(theta)
        align_cost = torch.zeros(self.num_samples, device=self.device)
        align_cost[self.cos_theta < 0] = -self.cos_theta[self.cos_theta < 0]  # (1 - cos_theta)

        # Add the cost when the robot is close to the block and moves towards the block
        vel_cost = torch.zeros(self.num_samples, device=self.device)
        robot_block_close = robot_to_block_dist <= 0.5
        vel_cost[flag_towards_block * robot_block_close] = 0.6

        return 3 * self.dist_cost + 3 * vel_cost + 7 * align_cost

    def get_panda_reach_cost(self, sim, pre_pick_goal):
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_r_state = sim.get_actor_link_by_name("panda", "panda_rightfinger")
        ee_state = (ee_l_state + ee_r_state) / 2
        cube_state = sim.get_actor_link_by_name("cubeA", "box")
        if not self.multi_modal:
            pre_pick_goal = cube_state[0, :3].clone()
            pre_pick_goal[2] += self.pre_height_diff
            reach_cost = torch.linalg.norm(ee_state[:, :3] - pre_pick_goal, axis=1)
        else:
            pre_pick_goal = cube_state[:, :3].clone()
            pre_pick_goal_1 = cube_state[0, :3].clone()
            pre_pick_goal_2 = cube_state[0, :3].clone()
            pre_pick_goal_1[2] += self.pre_height_diff
            pre_pick_goal_2[0] -= self.pre_height_diff * self.tilt_cos_theta
            pre_pick_goal_2[2] += self.pre_height_diff * (1 - self.tilt_cos_theta ** 2) ** 0.5
            pre_pick_goal[:self.half_samples, :] = pre_pick_goal_1
            pre_pick_goal[self.half_samples:, :] = pre_pick_goal_2
            reach_cost = torch.linalg.norm(ee_state[:, :3] - pre_pick_goal, axis=1)

            # Compute the tilt value between ee and cube
        tilt_cost = self.get_pick_tilt_cost(sim)

        return 10 * reach_cost + tilt_cost

    def get_panda_pick_cost(self, sim, pre_place_state):
        cube_state = sim.get_actor_link_by_name("cubeA", "box")

        # Move to pre-place location
        goal_cost = torch.linalg.norm(pre_place_state[:3] - cube_state[:, :3], axis=1)
        cube_quaternion = cube_state[:, 3:7]
        goal_quatenion = pre_place_state[3:7].repeat(self.num_samples).view(self.num_samples, 4)
        ori_cost = skill_utils.get_general_ori_cube2goal(cube_quaternion, goal_quatenion)

        return 10 * goal_cost + 15 * ori_cost

    def get_panda_place_cost(self, sim):
        # task planner will send discrete gripper commands instead of sampling

        # Just to make mppi running! Actually this is not useful!!
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_r_state = sim.get_actor_link_by_name("panda", "panda_rightfinger")
        gripper_dist = torch.linalg.norm(ee_l_state[:, :3] - ee_r_state[:, :3], axis=1)
        gripper_cost = 2 * (1 - gripper_dist)

        return gripper_cost

    def get_pick_tilt_cost(self, sim):
        # This measures the cost of the tilt angle between the end effector and the cube
        ee_l_state = sim.get_actor_link_by_name("panda", "panda_leftfinger")
        ee_quaternion = ee_l_state[:, 3:7]
        cubeA_ori = sim.get_actor_orientation_by_name("cubeA")
        # cube_quaternion = cube_state[:, 3:7]
        if not self.multi_modal:
            # To make the z-axis direction of end effector to be perpendicular to the cube surface
            ori_ee2cube = skill_utils.get_general_ori_ee2cube(ee_quaternion, cubeA_ori, tilt_value=0)
            # ori_ee2cube = skill_utils.get_ori_ee2cube(ee_quaternion, cubeA_ori)
        else:
            # To combine costs of different tilt angles
            cost_1 = skill_utils.get_general_ori_ee2cube(ee_quaternion[:self.half_samples],
                                                         cubeA_ori[:self.half_samples], tilt_value=0)
            cost_2 = skill_utils.get_general_ori_ee2cube(ee_quaternion[self.half_samples:],
                                                         cubeA_ori[self.half_samples:], tilt_value=self.tilt_cos_theta)
            ori_ee2cube = torch.cat((cost_1, cost_2), dim=0)

        return 3 * ori_ee2cube

    def get_motion_cost(self, sim):
        if self.cfg.env_type == 'point_env':
            obs_force = sim.get_actor_contact_forces_by_name("dyn-obs", "box")  # [num_envs, 3]
        elif self.cfg.env_type == 'panda_env':
            obs_force = sim.get_actor_contact_forces_by_name("table", "box")
            obs_force += 4 * sim.get_actor_contact_forces_by_name("shelf_stand", "box")
            obs_force += sim.get_actor_contact_forces_by_name("cubeB", "box")

            table_position = sim.get_actor_position_by_name("table")
            shelf_position = sim.get_actor_position_by_name("shelf_stand")
            cubeB_position = sim.get_actor_position_by_name("cubeB")
            cubeA_position = sim.get_actor_position_by_name("cubeA")
            print("table的位置:", table_position)
            print("shelf的位置:", shelf_position)
            print("cubeB的位置:", cubeB_position)
            print("cubeA的位置:", cubeA_position)

        coll_cost = torch.sum(torch.abs(obs_force[:, :2]), dim=1)  # [num_envs]
        # Binary check for collisions.
        coll_cost[coll_cost > 0.1] = 1
        coll_cost[coll_cost <= 0.1] = 0

        return 1000 * coll_cost

    def collision_cost(self):

        # Binary check for collisions.
        self.closest_dist_all[self.closest_dist_all < 0] = 1
        self.closest_dist_all[self.closest_dist_all > 0] = 0

        return 1000 * self.closest_dist_all

        # return (self.closest_dist_all < 0).sum(dim=1)

    def get_motion_cost_1(self, sim: wrapper):
        #q_prev = sim._dof_state[0]
        q_prev = sim._dof_state
        print("q_prev:", q_prev)
        #print("sim.robot_pos", sim.robot_pos)
        print("sim._dof_state:", sim._dof_state)
        print("sim._dof_state.shape:", sim._dof_state.shape)
        print(f"sim._dof_state[0].shape: {sim._dof_state[0].shape}")

        # TODO:通过sim得到各障碍物的点
        # cubeA_size = sim.get_actor_size("cubeA")
        # dynObs_size = sim.get_actor_size("dyn-obs")
        # dynObs_position = sim.get_actor_position_by_name("dyn-obs")

        # if self.cfg.env_type == 'panda_env':
        #     self.obs

        # 示例：
        # # Obstacle spheres (x, y, z, r)
        # obs = torch.tensor([[6, 2, 0, .5],
        #                     [4., -1, 0, .5],
        #                     [5, 0, 0, .5]]).to(**params)

        # evaluate NN. Calculate kernel bases on first iteration
        distance, self.nn_grad = self.distance_repulsion_nn(sim, q_prev, aot=False)
        self.nn_grad = self.nn_grad[0:self.N_traj, :]  # fixes issue with aot_function cache
        # distance, self.nn_grad = self.distance_repulsion_fk(q_prev) #not implemented for Franka

        distance -= self.dst_thr
        self.closest_dist_all = distance[0:self.N_traj]
        distance = distance[0:self.N_traj]

        print("closest_dist_all", self.closest_dist_all)
        # Binary check for collisions.
        # self.closest_dist_all[self.closest_dist_all < 0] = 1
        # self.closest_dist_all[self.closest_dist_all > 0] = 0

        self.closest_dist_all[self.closest_dist_all < 1.8] = 1
        self.closest_dist_all[self.closest_dist_all > 1.8] = 0

        #return 1000 * self.closest_dist_all
        #return 200 * self.closest_dist_all
        return 190 * self.closest_dist_all

    #计算障碍物的各点的位置
    def obs_positions(self, sim):

        device = sim.device

        actor_names = ["table_stand", "dyn-obs", "shelf_stand", "table"]
        all_vertices = []
        for actor_name in actor_names:

            size = sim.get_actor_size(actor_name)
            position = sim.get_actor_position_by_name(actor_name)

            # 给定的参数
            center = position[0, :2]  # 中点位置，形状为 (2,)
            length = size[0]  # 长度
            width = size[1]  # 宽度
            radius = size[2]/2

            # 转换 radius 为 Tensor，并确保设备一致
            radius = torch.tensor([radius], device=position.device) if not isinstance(radius, torch.Tensor) else radius

            center = center.to(device)


            # 计算偏移量
            half_length = length / 2
            half_width = width / 2

            # 计算四个顶点的坐标

            top_left = center + torch.tensor([-half_length, half_width], device=device)
            top_right = center + torch.tensor([half_length, half_width], device=device)
            bottom_left = center + torch.tensor([-half_length, -half_width], device=device)
            bottom_right = center + torch.tensor([half_length, -half_width], device=device)

            top_left = torch.cat([top_left, position[0, 2].unsqueeze(0), radius])
            top_right = torch.cat([top_right, position[0, 2].unsqueeze(0), radius])
            bottom_left = torch.cat([bottom_left, position[0, 2].unsqueeze(0), radius])
            bottom_right = torch.cat([bottom_right, position[0, 2].unsqueeze(0), radius])

            #top_left = torch.cat([top_left, position[0, 2], radius])
            # top_right = torch.cat([top_right, position[0, 2], radius])
            # bottom_left = torch.cat([bottom_left, position[0, 2], radius])
            # bottom_right = torch.cat([bottom_right, position[0, 2], radius])

            # 拼接成一个张量
            vertices = torch.stack([top_left, top_right, bottom_left, bottom_right], dim=0)
            all_vertices.append(vertices)

        return torch.vstack(all_vertices).to(device)



    def distance_repulsion_nn(self, sim, q_prev, aot=False):
        device = sim.device

        # 打印设备信息以便调试
        # print(f"sim.device: {device}")
        #q_prev = q_prev[:32]

        n_inputs = q_prev.shape[0]
        self.obs = self.obs_positions(sim).to(device)
        self.n_obs = self.obs.shape[0]

        with record_function("TAG: evaluate NN_1 (build input)"):
            # building input tensor for NN (N_traj * n_obs, n_dof + 3)
            nn_input = self.build_nn_input(q_prev, self.obs).to(device)

            # print(f"nn_input shape before slicing: {nn_input.shape}")

        # 打印输入张量和模型设备
        # print(f"nn_input device: {nn_input.device}")
        # print(f"Model: {self.nn_model.model_jit}")

        # parameters = list(self.nn_model.model_jit.parameters())
        # if not parameters:
        #     print("No parameters found in the TorchScript model.")
        # else:
        #     print(f"Model has {len(parameters)} parameters.")

        #print(f"Model device: {next(self.nn_model.model_jit.parameters()).device}")

        with record_function("TAG: evaluate NN_2 (forward pass)"):
            # 确保模型在目标设备上
            self.nn_model.model_jit = self.nn_model.model_jit.to(device)

            print(f"Input shape: {nn_input[:, 0:-1].shape}")
            print(f"nn_input shape: {nn_input.shape}")
            #
            # print(self.nn_model.model_jit.graph)
            # print(self.nn_model.model_jit)

            #nn_input = nn_input[:, :10]
            

            # doing single forward pass to figure out the closest obstacle for each configuration
            #nn_dist = self.nn_model.model_jit.forward(nn_input[:, 0:-1])
            #nn_dist = self.nn_model.model_jit.forward(nn_input)

            padding = torch.zeros(nn_input.size(0), 31 - nn_input.size(1), device=nn_input.device)
            nn_input_padded = torch.cat((nn_input, padding), dim=1)
            nn_dist = self.nn_model.model_jit.forward(nn_input_padded[:, 0:-1])

            print("nn_dist:", nn_dist)
            print("nn_dist.shape:", nn_dist.shape)

            if self.nn_model.out_channels == 9:
                nn_dist = nn_dist / 100  # scale down to meters
        with record_function("TAG: evaluate NN_3 (get closest obstacle)"):
            # rebuilding input tensor to only include closest obstacles
            nn_input_resized = nn_input[:nn_dist.shape[0], -1].unsqueeze(1)
            nn_dist -= nn_input_resized

            #nn_dist -= nn_input[:, -1].unsqueeze(1)  # subtract radius
            nn_dist[:, self.ignored_links] = 1e6  # ignore specified links
            mindist, _ = nn_dist.min(1)

            # required_size = self.n_obs * n_inputs
            # if mindist.numel() < required_size:
            #     padded_mindist = torch.zeros(required_size, device=device)
            #     padded_mindist[:mindist.numel()] = mindist
            #     mindist = padded_mindist

            mindist_matrix = mindist.reshape(self.n_obs, n_inputs).transpose(0, 1)
            mindist, sphere_idx = mindist_matrix.min(1)
            sort_dist, sort_idx = mindist_matrix.sort(dim=1)
            mindist_arr = sort_dist[:, 0:self.n_closest_obs]
            sphere_idx_arr = sort_idx[:, 0:self.n_closest_obs]
            # mask_idx = self.traj_range[:n_inputs] + sphere_idx * n_inputs
            #mask_idx = torch.arange(n_inputs) + sphere_idx * n_inputs
            # Ensure all tensors are on the same device
            # sphere_idx = sphere_idx.to(device)
            # sphere_idx_arr = sphere_idx_arr.to(device)

            mask_idx = torch.arange(n_inputs, device=device) + sphere_idx * n_inputs

            # nn_input = nn_input[mask_idx, :]

            # new_mask_idx = torch.arange(n_inputs).unsqueeze(1).repeat(1, self.n_closest_obs) + sphere_idx_arr * n_inputs
            # nn_input = nn_input[new_mask_idx.flatten(), :]

            # Handle new mask_idx for closest obstacles
            new_mask_idx = (
                    torch.arange(n_inputs, device=device)
                    .unsqueeze(1)
                    .repeat(1, self.n_closest_obs)
                    + sphere_idx_arr * n_inputs
            )

            # 限制索引范围
            # if new_mask_idx.max() >= nn_input.shape[0]:
            #     print("Index out of bounds detected. Adjusting indices...")
            #     new_mask_idx = new_mask_idx.clamp(0, nn_input.shape[0] - 1)

            nn_input = nn_input[new_mask_idx.flatten(), :]

        with record_function("TAG: evaluate NN_4 (forward+backward pass)"):
            # forward + backward pass to get gradients for closest obstacles
            # nn_dist, nn_grad, nn_minidx = self.nn_model.compute_signed_distance_wgrad(nn_input[:, 0:-1], 'closest')
            # print(f"x_nerf shape: {x_nerf.shape}")
            # print(f"Layer 0 expected in_features: {self.layers[0].in_features}")

            #nn_dist, nn_grad, nn_minidx = self.nn_model.dist_grad_closest(nn_input)


            if aot:
                nn_dist, nn_grad, nn_minidx = self.nn_model.dist_grad_closest_aot(nn_input)
            else:
                #nn_dist, nn_grad, nn_minidx = self.nn_model.dist_grad_closest(nn_input)
                print("神经网络nn_input_padded[:, 0:-1]调用前", nn_input_padded[:, 0:-1].shape)
                nn_dist, nn_grad, nn_minidx = self.nn_model.dist_grad_closest(nn_input_padded[:, 0:-1])
                nn_grad = nn_grad.squeeze(2)

                print("神经网络nn_dist", nn_dist.shape)
                print("神经网络nn_input_padded[:, 0:-1]调用后", nn_input_padded[:, 0:-1].shape)
            # if aot:
            #     nn_dist, nn_grad, nn_minidx = self.nn_model.dist_grad_closest_aot(nn_input)
            # else:
            #     nn_dist, nn_grad, nn_minidx = self.nn_model.dist_grad_closest(nn_input)
            #     nn_grad = nn_grad.squeeze(2)

            self.nn_grad = nn_grad[:nn_input.shape[0], 0:self.n_dof]
            if self.nn_model.out_channels == 9:
                nn_dist = nn_dist / 100  # scale down to meters

        with record_function("TAG: evaluate NN_5 (process outputs)"):

            # cleaning up to get distances and gradients for closest obstacles
            print(f"nn_dist shape: {nn_dist.shape}")
            print(f"nn_input shape: {nn_input.shape}")
            nn_dist -= nn_input[:, -1].unsqueeze(1)  # subtract radius and some threshold
            # extract closest link distance
            nn_dist = nn_dist[torch.arange(self.n_closest_obs * n_inputs).unsqueeze(1), nn_minidx.unsqueeze(1)]
            # reshape to match n_traj x n_closest_obs
            nn_dist = nn_dist.reshape(n_inputs, self.n_closest_obs)
            self.nn_grad = self.nn_grad.reshape(n_inputs, self.n_closest_obs, self.n_dof)
            # weight gradients according to closest distance
            weighting = (-10 * nn_dist).softmax(dim=-1)
            self.nn_grad = torch.sum(self.nn_grad * weighting.unsqueeze(2), dim=1)
            # distance - mindist
            distance = nn_dist[:, 0]
        return distance, self.nn_grad

    # def build_nn_input(self, q_tens, obs_tens):
    #     print("q_prev shape:", q_tens.shape)
    #     print("obs shape:", obs_tens.shape)
    #
    #     self.nn_input = torch.hstack(
    #         (q_tens.tile(obs_tens.shape[0], 1), obs_tens.repeat_interleave(q_tens.shape[0], 0)))
    #     return self.nn_input

    def build_nn_input(self, q_tens, obs_tens):
        print(f"q_tens shape: {q_tens.shape}")
        print(f"obs_tens shape: {obs_tens.shape}")

        self.nn_input = torch.hstack(
            (q_tens.tile(obs_tens.shape[0], 1), obs_tens.repeat_interleave(q_tens.shape[0], 0)))
        return self.nn_input

    # def build_nn_input(self, q_tens, obs_tens):
    #     print("q_prev shape:", q_tens.shape)
    #     print("obs shape:", obs_tens.shape)
    #
    #     # 确保 q_tens 是 2D 张量
    #     if q_tens.dim() == 1:
    #         q_tens = q_tens.unsqueeze(0)  # [1, 18]
    #
    #     # 扩展 q_tens 和 obs_tens
    #     q_tens_expanded = q_tens.repeat(obs_tens.shape[0], 1)  # [16, 18]
    #     obs_tens_expanded = obs_tens.repeat_interleave(q_tens.shape[0], dim=0)  # [16, 4]
    #
    #     # 打印调试
    #     print("q_tens_expanded shape:", q_tens_expanded.shape)
    #     print("obs_tens_expanded shape:", obs_tens_expanded.shape)
    #
    #     # 拼接张量
    #     self.nn_input = torch.hstack((q_tens_expanded, obs_tens_expanded))
    #     return self.nn_input

    def update_obstacles(self, obs):
        self.obs = obs
        self.n_obs = obs.shape[0]
        return 0
