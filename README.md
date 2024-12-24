# mppi

## Navigation among obstacles
### Download the trained model for obstacle avoidance
Download the neutral network trained model for obstacle avoidance from https://github.com/epfl-lasa/OptimalModulationDS/tree/master/python_scripts/mlp_learn and move it to https://github.com/greengreen12345/mppi/tree/main/src/m3p2i_aip/planners. 

### Load the neural network model
Load the nn_model "7dof_sdf_256x5_mesh.pt" in "scripts/reactive_tamp.py def __init__()" as follows:
````bash
        DOF = 7
        L = 1

        # Load nn model
        s = 256
        n_layers = 5
        skips = []
        fname = '%ddof_sdf_%dx%d_mesh.pt' % (DOF, s, n_layers) # planar robot
        
        nn_model = RobotSdfCollisionNet(in_channels=DOF + 3, out_channels=DOF, layers=[s] * n_layers, skips=skips)
       
````
Then input nn_model into the "Objective" class:
````bash
self.objective = Objective(cfg, nn_model)
````

### Modify "cost_function.py" to "cost_function1.py"
#### Define the obstacle positions in the function "def obs_positions(self, sim):"
Refer to the definition of obstacles in the "OptimalModulationDS" method as follows: 
````bash
    # Obstacle spheres (x, y, z, r)
    obs = torch.tensor([[6, 2, 0, .5],
                        [4., -1, 0, .5],
                        [5, 0, 0, .5]]).to(**params)
````
Among them, for each vector in the tensor, the first three elements represent the position of the obstacle's center, and the fourth element is the defined threshold for the minimum distance from the obstacle. Each of them can be approximated as a sphere.

The obstacles in the environment are as ["table_stand", "dyn-obs", "shelf_stand", "table"]. For each obstacle, the positions of the four vertices of the cubes' median plane parallel to the ground are calculated, and the minimum threshold of distance from the vertices is defined as half the height of the cubes. Finally all the vertices are concatenated into a tensor.

````bash
def obs_positions(self, sim):

        device = sim.device

        actor_names = ["table_stand", "dyn-obs", "shelf_stand", "table"]
        all_vertices = []
        for actor_name in actor_names:

            size = sim.get_actor_size(actor_name)
            position = sim.get_actor_position_by_name(actor_name)

            # The given parameters
            center = position[0, :2]  # Midpoint position with a shape of (2,)
            length = size[0]  
            width = size[1]  
            radius = size[2]/2

            # Convert radius to a Tensor and ensure device consistency
            radius = torch.tensor([radius], device=position.device) if not isinstance(radius, torch.Tensor) else radius
            center = center.to(device)

            # Calculate the offset
            half_length = length / 2
            half_width = width / 2

            # Calculate the coordinates of the four vertices
            top_left = center + torch.tensor([-half_length, half_width], device=device)
            top_right = center + torch.tensor([half_length, half_width], device=device)
            bottom_left = center + torch.tensor([-half_length, -half_width], device=device)
            bottom_right = center + torch.tensor([half_length, -half_width], device=device)

            top_left = torch.cat([top_left, position[0, 2].unsqueeze(0), radius])
            top_right = torch.cat([top_right, position[0, 2].unsqueeze(0), radius])
            bottom_left = torch.cat([bottom_left, position[0, 2].unsqueeze(0), radius])
            bottom_right = torch.cat([bottom_right, position[0, 2].unsqueeze(0), radius])

            # Concatenate into a tensor
            vertices = torch.stack([top_left, top_right, bottom_left, bottom_right], dim=0)
            all_vertices.append(vertices)

        return torch.vstack(all_vertices).to(device)
````

#### Define the function "def get_motion_cost_1(self, sim: wrapper):" to calculate the cost for obstacle avoidance

````bash
def get_motion_cost_1(self, sim: wrapper):
        
        q_prev = sim._dof_state
        
        # evaluate NN. Calculate kernel bases on first iteration
        distance, self.nn_grad = self.distance_repulsion_nn(sim, q_prev, aot=False)
        self.nn_grad = self.nn_grad[0:self.N_traj, :]  # fixes issue with aot_function cache

        distance -= self.dst_thr
        self.closest_dist_all = distance[0:self.N_traj]
        distance = distance[0:self.N_traj]

        print("closest_dist_all", self.closest_dist_all)
        # Binary check for collisions.
        self.closest_dist_all[self.closest_dist_all < 1.8] = 1
        self.closest_dist_all[self.closest_dist_all > 1.8] = 0

        return 190 * self.closest_dist_all
````
Among them, the function "def distance_repulsion_nn(self, sim, q_prev, aot=False):" is used to calculate the minimum distance to obstacles for 200 sampled states at each time step.     

````bash
distance, self.nn_grad = self.distance_repulsion_nn(sim, q_prev, aot=False)  
````


After calculating the total cost in function "def compute_cost(self, sim: wrapper):" in cost_functions1.py, this function is called in 
````bash
c = self._running_cost(state)
````
in the [mppi file](https://github.com/greengreen12345/mppi/blob/main/src/m3p2i_aip/planners/motion_planner/mppi.py#L309) in function "def _compute_rollout_costs(self, perturbed_actions):" as follows:

````bash
for t in range(T):
            u = self.u_scale * perturbed_actions[:, t]

            # Last rollout is a braking manover
            if self.sample_null_action:
                u[self.K -1, :] = torch.zeros_like(u[self.K -1, :])
                self.perturbed_action[self.K - 1][t] = u[self.K -1, :]

            state, u = self._dynamics(state, u, t)
            c = self._running_cost(state) # every time step you get nsamples cost, we need that as output for the discount factor
            # Update action if there were changes in M3P2I due for instance to suction constraints
            self.perturbed_action[:,t] = u

            cost_samples += c
            cost_horizon[:, t] = c 

            # Save total states/actions
            states.append(state)
            actions.append(u)
            ee_state = 'None' #(self.ee_l_state[:, :3] + self.ee_r_state[:, :3])/2 if self.ee_l_state != 'None' else 'None'
            ee_states.append(ee_state) if ee_state != 'None' else []
````
