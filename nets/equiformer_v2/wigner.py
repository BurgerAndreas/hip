import os
import torch


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
# _Jd is a list of tensors of shape (2l+1, 2l+1)
_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"), weights_only=True)


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
#
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
def wigner_D(ell, alpha, beta, gamma):
    # if not ell < len(_Jd):
    #     raise NotImplementedError(
    #         f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
    #     )
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[ell].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, ell)
    Xb = _z_rot_mat(beta, ell)
    Xc = _z_rot_mat(gamma, ell)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle, ell):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * ell + 1, 2 * ell + 1))
    inds = torch.arange(0, 2 * ell + 1, 1, device=device)
    reversed_inds = torch.arange(2 * ell, -1, -1, device=device)
    frequencies = torch.arange(ell, -ell - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M
