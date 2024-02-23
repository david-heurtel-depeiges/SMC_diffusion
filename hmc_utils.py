import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collision_manager(q, p, p_nxt, phi_min_norm, phi_max_norm):
    p_ret = p_nxt
    nparams = q.shape[-1]
    for i in range(nparams):
        crossed_min_boundary = q[..., i] < phi_min_norm[i]
        crossed_max_boundary = q[..., i] > phi_max_norm[i]

        # Reflecting boundary conditions
        p_ret[..., i][crossed_min_boundary] = -p[..., i][crossed_min_boundary]
        p_ret[..., i][crossed_max_boundary] = -p[..., i][crossed_max_boundary]

    return p_ret

def boundary_projection(phi, phi_min_norm, phi_max_norm, eps=1e-3):
    nparams = phi.shape[-1]
    for i in range(nparams):
        phi[..., i] = torch.max(phi_min_norm[i] + eps, torch.min(phi_max_norm[i] - eps, phi[..., i]))
    return phi