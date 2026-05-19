import torch


def init_edge_rot_mat(edge_distance_vec):
    edge_vec_0 = edge_distance_vec
    try:
        edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))
    except:
        print("edge_distance_vecshould be [N,3] but got", edge_vec_0.shape)
        raise

    # Make sure the atoms are far enough apart
    # assert torch.min(edge_vec_0_distance) < 0.0001
    if torch.min(edge_vec_0_distance) < 0.0001:
        print(
            "Error edge_vec_0_distance: {} "
            "(atoms are too close; minimum graph edge distance is below 1e-4 Angstrom)".format(
                torch.min(edge_vec_0_distance)
            )
        )

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    # Pick a deterministic auxiliary axis that is least aligned with each edge.
    candidate_axes = torch.eye(3, dtype=edge_vec_0.dtype, device=edge_vec_0.device)
    edge_vec_2 = candidate_axes[torch.argmin(torch.abs(norm_x), dim=1)]

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)))
    norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1))
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)))

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    return edge_rot_mat.detach()
