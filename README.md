# mppi

## Navigation among obstacles
### Download the trained model for obstacle avoidance
Download the neutral network trained model for obstacle avoidance from https://github.com/epfl-lasa/OptimalModulationDS/tree/master/python_scripts/mlp_learn and move it to https://github.com/greengreen12345/mppi/tree/main/src/m3p2i_aip/planners. 

### Define the obstacle positions
Refer to the definition of obstacles in the "OptimalModulationDS" method as follows. For each vector in the tensor, the first three elements represent the position of the obstacle's center, and the fourth element is the defined threshold for the minimum distance from the obstacle. Each of them can be approximated as a sphere.
>[!]
    # Obstacle spheres (x, y, z, r)
    obs = torch.tensor([[6, 2, 0, .5],
                        [4., -1, 0, .5],
                        [5, 0, 0, .5]]).to(**params)

      
