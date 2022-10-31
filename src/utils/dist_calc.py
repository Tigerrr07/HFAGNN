# Base on code from https://github.com/divelab/DIG/blob/dig-stable/dig/threedgraph/method/comenet/comenet.py
import math
import torch
from torch_scatter import scatter_min


def dist_calc(cutoff, vecs, dist, i, j, num_nodes):
    _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
    argmin0[argmin0 >= len(i)] = 0
    n0 = j[argmin0]
    add = torch.zeros_like(dist).to(dist.device)
    add[argmin0] = cutoff
    dist1 = dist + add

    _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
    argmin1[argmin1 >= len(i)] = 0
    n1 = j[argmin1]
    # --------------------------------------------------------

    _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
    argmin0_j[argmin0_j >= len(j)] = 0
    n0_j = i[argmin0_j]

    add_j = torch.zeros_like(dist).to(dist.device)
    add_j[argmin0_j] = cutoff
    dist1_j = dist + add_j

    # i[argmin] = range(0, num_nodes)
    _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
    argmin1_j[argmin1_j >= len(j)] = 0
    n1_j = i[argmin1_j]

    # ----------------------------------------------------------

    # n0, n1 for i
    n0 = n0[i]
    n1 = n1[i]

    # n0, n1 for j
    n0_j = n0_j[j]
    n1_j = n1_j[j]

    # tau: (iref, i, j, jref)
    # when compute tau, do not use n0, n0_j as ref for i and j,
    # because if n0 = j, or n0_j = i, the computed tau is zero
    # so if n0 = j, we choose iref = n1
    # if n0_j = i, we choose jref = n1_j
    mask_iref = n0 == j
    iref = n0.clone()
    iref[mask_iref] = n1[mask_iref]
    idx_iref = argmin0[i]
    idx_iref[mask_iref] = argmin1[i][mask_iref]

    mask_jref = n0_j == i
    jref = n0_j.clone()
    jref[mask_jref] = n1_j[mask_jref]
    idx_jref = argmin0_j[j]
    idx_jref[mask_jref] = argmin1_j[j][mask_jref]

    pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
        vecs,
        vecs[argmin0][i],
        vecs[argmin1][i],
        vecs[idx_iref],
        vecs[idx_jref]
    )

    # print("vecs position:", pos_in0, pos_in1)
    # Calculate angles.
    a = ((-pos_ji) * pos_in0).sum(dim=-1)
    b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
    theta = torch.atan2(b, a)
    theta[theta < 0] = theta[theta < 0] + math.pi
    # print(theta[:10])
    # Calculate torsions.
    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
    assert not torch.any(torch.isnan(dist_ji))  # dost_ji error

    plane1 = torch.cross(-pos_ji, pos_in0)
    plane2 = torch.cross(-pos_ji, pos_in1)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    # print((torch.cross(plane1, plane2) * pos_ji).sum(dim=-1)[:10])
    # print(dist_ji[:10])
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    idx = torch.where(torch.isnan(b))
    # print("idx:", idx, "b:", b, "dist:", dist_ji[idx])
    assert not torch.any(torch.isnan(b))  # b error
    phi = torch.atan2(b, a)
    phi[phi < 0] = phi[phi < 0] + math.pi

    # Calculate right torsions.
    plane1 = torch.cross(pos_ji, pos_jref_j)
    plane2 = torch.cross(pos_ji, pos_iref)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    tau = torch.atan2(b, a)
    tau[tau < 0] = tau[tau < 0] + math.pi
    return theta, phi, tau