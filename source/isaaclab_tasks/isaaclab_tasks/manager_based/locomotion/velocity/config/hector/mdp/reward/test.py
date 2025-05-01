import torch


def bilinear_interpolation(
    foot_position_2d: torch.Tensor, 
    costmap_2d: torch.Tensor,
    resolution: float = 0.1,
):
    """
    foot position_2d: (num_envs, num_sample, 2, 2)
    costmap_2d: (num_envs, height, width)
    
    P1 ----- P2
    |    x   |
    P3 ----- P4
    """
    num_envs = foot_position_2d.shape[0]
    num_foot_edge = foot_position_2d.shape[2]
    num_foot = foot_position_2d.shape[1]
    
    env_ids = torch.arange(num_envs)
    height, width = costmap_2d.shape[1:3]
    
    # in image space
    # foot position_2d_flat: (num_envs, num_foot_edge*num_foot, 2)
    foot_position_2d_flat = foot_position_2d.clone().reshape(num_envs, -1, 2)
    foot_position_discrete = torch.zeros_like(foot_position_2d_flat)
    foot_position_discrete[:, :, 0] = height//2 - (foot_position_2d_flat[:, :, 1]/resolution) # row 
    foot_position_discrete[:, :, 1] = width//2 + (foot_position_2d_flat[:, :, 0]/resolution) # col
    
    # (num_envs, num_foot_edge*num_foot, 2)
    P1 = torch.zeros_like(foot_position_discrete).to(torch.long)
    P2 = torch.zeros_like(foot_position_discrete).to(torch.long)
    P3 = torch.zeros_like(foot_position_discrete).to(torch.long)
    P4 = torch.zeros_like(foot_position_discrete).to(torch.long)
    
    P1[:, :, 0] = foot_position_discrete[:, :, 0].floor().clamp(0, height-1)
    P1[:, :, 1] = foot_position_discrete[:, :, 1].floor().clamp(0, width-1)
    P2[:, :, 0] = foot_position_discrete[:, :, 0].floor().clamp(0, height-1)
    P2[:, :, 1] = foot_position_discrete[:, :, 1].ceil().clamp(0, width-1)
    P3[:, :, 0] = foot_position_discrete[:, :, 0].ceil().clamp(0, height-1)
    P3[:, :, 1] = foot_position_discrete[:, :, 1].floor().clamp(0, width-1)
    P4[:, :, 0] = foot_position_discrete[:, :, 0].ceil().clamp(0, height-1)
    P4[:, :, 1] = foot_position_discrete[:, :, 1].ceil().clamp(0, width-1)
    
    P1_height = costmap_2d.reshape(num_envs, -1).repeat(4, 1)[env_ids, P1[:, :, 0] * width + P1[:, :, 1]].view(num_envs, -1)
    P2_height = costmap_2d.reshape(num_envs, -1).repeat(4, 1)[env_ids, P2[:, :, 0] * width + P2[:, :, 1]].view(num_envs, -1)
    P3_height = costmap_2d.reshape(num_envs, -1).repeat(4, 1)[env_ids, P3[:, :, 0] * width + P3[:, :, 1]].view(num_envs, -1)
    P4_height = costmap_2d.reshape(num_envs, -1).repeat(4, 1)[env_ids, P4[:, :, 0] * width + P4[:, :, 1]].view(num_envs, -1)
    
    # Calculate distances from the foot position to each corner
    dist_P1 = (foot_position_discrete - P1).float().abs()
    dist_P2 = (foot_position_discrete - P2).float().abs()
    dist_P3 = (foot_position_discrete - P3).float().abs()
    dist_P4 = (foot_position_discrete - P4).float().abs()

    # Compute weights based on inverse distance
    weights_P1 = (1.0 / (dist_P1[:, :, 0] * dist_P1[:, :, 1] + 1e-6))
    weights_P2 = (1.0 / (dist_P2[:, :, 0] * dist_P2[:, :, 1] + 1e-6))
    weights_P3 = (1.0 / (dist_P3[:, :, 0] * dist_P3[:, :, 1] + 1e-6))
    weights_P4 = (1.0 / (dist_P4[:, :, 0] * dist_P4[:, :, 1] + 1e-6))

    # Normalize weights
    total_weights = weights_P1 + weights_P2 + weights_P3 + weights_P4
    weights_P1 /= total_weights
    weights_P2 /= total_weights
    weights_P3 /= total_weights
    weights_P4 /= total_weights

    # Compute weighted average height
    height_at_foot = (
        weights_P1 * P1_height +
        weights_P2 * P2_height +
        weights_P3 * P3_height +
        weights_P4 * P4_height
    )
    
    height_at_foot = height_at_foot.reshape(-1, num_foot, num_foot_edge).transpose(1, 2) # (num_envs, num_foot_edge, num_foot)
    foot_position_discrete = foot_position_discrete.reshape(num_envs, num_foot, num_foot_edge, 2).transpose(1, 2) # (num_envs, num_foot_edge, num_foot)
    
    return height_at_foot, foot_position_discrete
    

def get_ground_roughness_at_landing_point(
    num_envs: int,
    foot_position: torch.Tensor,
    foot_contact: torch.Tensor,
    costmap_2d: torch.Tensor,
    resolution: float = 0.1,
    ):
    """
    This function returns flatness value which is the difference between the max and min height in a 3x3 grid. 
    So, if robot is stepping on non-flat surface, flatness value results in terrain height value.
    
    
    Foot edge configurations 
    y|
    x-> 
    1 ------ 2
    |    x   |
    3 ------ 4
    """
    
    # in cartesian space 
    foot_position_2d = foot_position.clone().reshape(-1, 2, 3)[:, :, :2] # foot position wrt body frame
    foot_edge_positions = torch.zeros(num_envs, 2, 4, 2)
    foot_edge_positions[:, 0, :, :] = foot_position_2d[:, :1, :].repeat(1, 4, 1)
    foot_edge_positions[:, 1, :, :] = foot_position_2d[:, 1:, :].repeat(1, 4, 1)
    foot_edge_positions[:, :, 0, 0] -= 0.145/2
    foot_edge_positions[:, :, 0, 1] += 0.073/2
    foot_edge_positions[:, :, 1, 0] += 0.145/2
    foot_edge_positions[:, :, 1, 1] += 0.073/2
    foot_edge_positions[:, :, 2, 0] -= 0.145/2
    foot_edge_positions[:, :, 2, 1] -= 0.073/2
    foot_edge_positions[:, :, 3, 0] += 0.145/2
    foot_edge_positions[:, :, 3, 1] -= 0.073/2
    
    height_at_foot, foot_discrete = bilinear_interpolation(
        foot_position_2d=foot_edge_positions, 
        costmap_2d=costmap_2d, 
        resolution=resolution
    ) # (num_envs, 4, 2)
    
    roughness_at_foot = torch.abs(height_at_foot.max(dim=1).values - height_at_foot.min(dim=1).values) # (num_envs, 2)
    roughness_at_foot = (roughness_at_foot * foot_contact).sum(dim=1) # (num_envs, 1)
    return roughness_at_foot, foot_discrete, height_at_foot


if __name__ == "__main__":
    num_envs = 1
    foot_position = torch.zeros(num_envs, 2, 3)
    foot_position[:, 0, 0] = -0.1
    foot_position[:, 0, 1] = 0.08

    foot_position[:, 1, 0] = -0.02
    foot_position[:, 1, 1] = -0.08
    foot_contact = torch.ones(num_envs, 2)
    
    height, width = 11, 11
    resolution = 0.1
    height_map_2d = torch.zeros(num_envs, height, width)
    
    edge_id = width // 2 + 2
    height_map_2d[:, :, edge_id:] = 0.1
    
    costmap_2d = torch.zeros_like(height_map_2d)
    edgemap_2d = torch.zeros_like(height_map_2d)
    edgemap_2d[:, :, 1:] = torch.abs(height_map_2d[:, :, :-1] - height_map_2d[:, :, 1:])
    costmap_2d[:, :, 1:-1] = (torch.abs(edgemap_2d[:, :, 2:] - edgemap_2d[:, :, 1:-1]) + \
                            torch.abs(edgemap_2d[:, :, :-2] - edgemap_2d[:, :, 1:-1])) * 0.5
    
    
    
    ######### raw height map ###############
    roughness_at_foot, foot_discrete, height_at_foot = get_ground_roughness_at_landing_point(
        num_envs=num_envs,
        foot_position=foot_position.reshape(num_envs, -1),
        foot_contact=foot_contact,
        costmap_2d=height_map_2d,
        resolution=resolution,
    )
    # plot
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    cmap = plt.get_cmap('jet')  # Or 'terrain', 'plasma', etc.
    norm = mcolors.Normalize(vmin=0.0, vmax=0.1)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    
    for i in range(0, width):
        for j in range(0, height):
            color = cmap(norm(height_map_2d[0, height-1-j, i]))
            ax[0].scatter(i, j, color=color, s=20)
    
    # plt scatter assume origin lower left corner. 
    # we treat all indices aligned with image coordiante which is upper left corner.
    
    ax[0].scatter(foot_discrete[0, :, 0, 1].cpu().numpy(), 
                height-foot_discrete[0, :, 0, 0].cpu().numpy(), 
                color=cmap(norm(height_at_foot[0, :, 0].cpu().numpy())), 
                ) # left  
    ax[0].scatter(foot_discrete[0, :, 1, 1].cpu().numpy(), 
                height-foot_discrete[0, :, 1, 0].cpu().numpy(), 
                color=cmap(norm(height_at_foot[0, :, 1].cpu().numpy())), 
                ) # right
    ax[0].plot(foot_discrete[0, :, 0, 1].cpu().numpy()[[0, 1, 3, 2, 0]], 
            height-foot_discrete[0, :, 0, 0].cpu().numpy()[[0, 1, 3, 2, 0]], 
            c="k")
    ax[0].plot(foot_discrete[0, :, 1, 1].cpu().numpy()[[0, 1, 3, 2, 0]],
            height-foot_discrete[0, :, 1, 0].cpu().numpy()[[0, 1, 3, 2, 0]],
            c="k")
    ax[0].set_title("raw height map")
    # plt.axis('equal')
    
    
    # ======== cost map ==============
    roughness_at_foot, foot_discrete, height_at_foot = get_ground_roughness_at_landing_point(
        num_envs=num_envs,
        foot_position=foot_position.reshape(num_envs, -1),
        foot_contact=foot_contact,
        costmap_2d=costmap_2d,
        resolution=resolution,
    )
    
    for i in range(0, width):
        for j in range(0, height):
            color = cmap(norm(costmap_2d[0, height-1-j, i]))
            ax[1].scatter(i, j, color=color, s=20)
    
    # plt scatter assume origin lower left corner. 
    # we treat all indices aligned with image coordiante which is upper left corner.
    
    ax[1].scatter(foot_discrete[0, :, 0, 1].cpu().numpy(), 
                height-foot_discrete[0, :, 0, 0].cpu().numpy(), 
                color=cmap(norm(height_at_foot[0, :, 0].cpu().numpy())), 
                ) # left  
    ax[1].scatter(foot_discrete[0, :, 1, 1].cpu().numpy(), 
                height-foot_discrete[0, :, 1, 0].cpu().numpy(), 
                color=cmap(norm(height_at_foot[0, :, 1].cpu().numpy())), 
                ) # right
    ax[1].plot(foot_discrete[0, :, 0, 1].cpu().numpy()[[0, 1, 3, 2, 0]], 
            height-foot_discrete[0, :, 0, 0].cpu().numpy()[[0, 1, 3, 2, 0]], 
            c="k")
    ax[1].plot(foot_discrete[0, :, 1, 1].cpu().numpy()[[0, 1, 3, 2, 0]],
            height-foot_discrete[0, :, 1, 0].cpu().numpy()[[0, 1, 3, 2, 0]],
            c="k")
    ax[1].set_title("cost map")
    
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label='cost')
    
    plt.show()
    