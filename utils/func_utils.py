import torch
import numpy as np
from roma import unitquat_slerp, unitquat_to_rotvec, rotvec_to_unitquat, quat_conjugation, quat_product
from torch.nn.functional import normalize

# M_0 = torch.tensor([1.0], dtype=torch.float32)
# M_1 = torch.tensor([
#     [1.0, 0.0],
#     [-1.0, 1.0]
# ], dtype=torch.float32)
# M_2 = torch.tensor([
#     [1.0, 1.0, 0.0],
#     [-2.0, 2.0, 0.0],
#     [1.0, -2.0, 1.0]
# ], dtype=torch.float32) / 2.0
# M_3 = torch.tensor([
#     [1.0, 4.0, 1.0, 0.0],
#     [-3.0, 0.0, 3.0, 0.0],
#     [3.0, -6.0, 3.0, 0.0],
#     [-1.0, 3.0, -3.0, 1.0]
# ], dtype=torch.float32) / 6.0
# M_4 = torch.tensor([
#  [  1.0,  11.0,  11.0,   1.0,   0.0],
#  [ -4.0, -12.0,  12.0,   4.0,   0.0],
#  [  6.0,  -6.0,  -6.0,   6.0,   0.0],
#  [ -4.0,  12.0, -12.0,   4.0,   0.0],
#  [  1.0,  -4.0,   6.0,  -4.0,   1.0]
# ], dtype=torch.float32) / 24.0
# M = [M_0, M_1, M_2, M_3, M_4]

M = dict()
POINTWISE_M = dict()

def get_deboor_cox_mat(order):
    if order == 0:
        return np.array([[1.0]], dtype=np.float32)
    
    prior_mat = get_deboor_cox_mat(order=order - 1)
    prior_mat_left = np.concatenate([prior_mat, np.zeros((1, prior_mat.shape[1]), dtype=np.float32)], axis=0)
    prior_mat_right = np.concatenate([np.zeros((1, prior_mat.shape[1]), dtype=np.float32), prior_mat], axis=0)
    teo_mat_left = np.zeros((order, order + 1), dtype=np.float32)
    idx = np.arange(order, dtype=np.int32)
    teo_mat_left[idx, idx] = idx + 1
    teo_mat_left[idx, idx + 1] = order - idx - 1

    teo_mat_right = np.zeros((order, order + 1), dtype=np.float32)
    idx = np.arange(order, dtype=np.int32)
    teo_mat_right[idx, idx] = -1
    teo_mat_right[idx, idx + 1] = 1

    return (prior_mat_left @ teo_mat_left + prior_mat_right @ teo_mat_right) / order

def get_fft_basic_func(v, order):
    # v: ..., 1
    freq = torch.linspace(1.0, order, order, dtype=torch.float32, device='cuda') * np.pi # F
    fft_sin, fft_cos = torch.sin(v * freq), torch.cos(v * freq)  # ..., F
    fft = torch.cat([fft_sin, fft_cos], dim=-1)  # ..., F * 2
    return fft

def get_poly_basic_func(v, order):
    # v: ..., 1
    freq = torch.linspace(1.0, order, order, dtype=torch.float32, device='cuda') # F
    poly = v ** freq  # ..., F
    return poly

def get_bspline_basic_func(v, order):
    # v: ..., 1 or float
    global M
    try:
        deboor_mat = M[order]
    except:
        print('Precompute de boor-cox matrix... order:', order)
        deboor_mat = torch.tensor(get_deboor_cox_mat(order), dtype=torch.float32, device='cuda')
        M[order] = deboor_mat

    freq = torch.arange(0.0, order + 1.0, 1.0, dtype=torch.float32, device='cuda')  # order + 1
    bspline = (v ** freq) @ deboor_mat  # ..., order + 1
    return bspline

def get_param_num(args):
    return args[0] + args[2] + 2 * args[3] + args[4]

def set_default_param_order(order_args: dict, frame_num: int, downsample_ratio: int = 3):
    res_args = dict()
    for k, v in order_args.items():
        args = v if v is not None else [None] * 6

        # bspline
        assert args[0] is None or args[0] >= 0, 'The B-Spline ctrl pts num cannot be negative in {}, but find {}.'.format(k, args[0])
        bspline_ctrlpts_num = args[0] if args[0] is not None else int(frame_num // downsample_ratio)

        bspline_order = 0
        if bspline_ctrlpts_num > 0:
            assert args[1] is None or args[1] >= 0, 'The B-Spline order cannot be negative in {}, but find {}.'.format(k, args[1])
            if args[1] is not None and args[1] + 1 > bspline_ctrlpts_num:
                print('[WARNING] The B-Spline order should be lower than the ctrl pts num. Set order to', bspline_ctrlpts_num - 1)
            bspline_order = args[1] if args[1] is not None else 5
            bspline_order = min(bspline_order, bspline_ctrlpts_num - 1)

        # poly
        assert args[2] is None or args[2] >= 0, 'The poly order cannot be negative in {}, but find {}.'.format(k, args[2])
        poly_order = args[2] if args[2] is not None else int(frame_num // downsample_ratio)

        # fft
        assert args[3] is None or args[3] >= 0, 'The fft order cannot be negative in {}, but find {}.'.format(k, args[3])
        fft_order = args[3] if args[3] is not None else 6

        assert args[4] is None or args[4] >= 0, 'The quaternion spline ctrl pts num cannot be negative in {}, but find {}.'.format(k, args[4])
        quat_spline_ctrlpts_num = args[4] if args[4] is not None else int(frame_num // downsample_ratio)

        quat_spline_order = 0
        if quat_spline_ctrlpts_num > 0:
            assert args[5] is None or args[5] >= 0, 'The quaternion spline order cannot be negative in {}, but find {}.'.format(k, args[5])
            if args[5] is not None and args[5] + 1 > quat_spline_ctrlpts_num:
                print('[WARNING] The quaternion spline order should be lower than the ctrl pts num. Set order to', quat_spline_ctrlpts_num - 1)
            quat_spline_order = args[5] if args[5] is not None else 1
            quat_spline_order = min(quat_spline_order, quat_spline_ctrlpts_num - 1)
        
        res_args[k] = [bspline_ctrlpts_num, bspline_order, poly_order, fft_order, quat_spline_ctrlpts_num, quat_spline_order]
    return res_args

def get_func_result(v, param, order_args):
    # v : float
    result = 0.0
    offset = 0

    # bspline
    if order_args[0] != 0:
        interval = order_args[0] - order_args[1]
        start_ctrl_idx = min(int(v * interval), interval - 1)
        ctrl_pts = param[..., start_ctrl_idx + offset: start_ctrl_idx + order_args[1] + offset + 1]  # N, D, k + 1
        u = v * interval - start_ctrl_idx
        func = get_bspline_basic_func(u, order_args[1])  # k + 1
        vector = torch.sum(ctrl_pts * func, dim=-1)  # N, D
        result = result + vector
        offset += order_args[0]

    # poly
    if order_args[2] != 0:
        poly_param = param[..., offset: offset + order_args[2]]
        func = get_poly_basic_func(v, order_args[2])
        # poly_param = poly_param[..., 2:]
        vector = torch.sum(poly_param * func, dim=-1)  # N, D
        result = result + vector
        offset += order_args[2]
    
    # fft
    if order_args[3] != 0:
        fft_param = param[..., offset: offset + order_args[3] * 2]
        func = get_fft_basic_func(v, order_args[3])
        # fft_param = fft_param[..., 2:]
        vector = torch.sum(fft_param * func, dim=-1)  # N, D
        result = result + vector
        offset += order_args[3] * 2

    # quaternion b-spline
    if order_args[4] != 0:
        interval = order_args[4] - order_args[5]
        start_ctrl_idx = min(int(v * interval), interval - 1)
        ctrl_quat = param[..., start_ctrl_idx + offset: start_ctrl_idx + order_args[5] + offset + 1] + torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda').reshape(-1, 1)  # N, 4, k + 1
        ctrl_quat = normalize(torch.permute(ctrl_quat, (0, 2, 1)), dim=-1)[..., [1, 2, 3, 0]]  # N, k + 1, 4
        u = v * interval - start_ctrl_idx
        func = get_bspline_basic_func(u, order_args[5])  # k + 1
        func_cum = torch.flip(torch.cumsum(torch.flip(func, dims=(-1,)), dim=-1), dims=(-1,))[..., 1:]  # k
        conj_q = quat_conjugation(ctrl_quat[:, :-1, :])
        vec = unitquat_to_rotvec(quat_product(conj_q, ctrl_quat[:, 1:, :]))  # N, k, 4
        quat = rotvec_to_unitquat(vec * func_cum[None, :, None]) # N, k, 4
        vector = ctrl_quat[:, 0]  # N, 4
        for i in range(quat.shape[1]):
            vector = quat_product(vector, quat[:, i])
        result = result + vector[..., [3, 0, 1, 2]]  # N, 4
        offset += order_args[4]

    return result


def get_pointwise_func_result(v, param, order_args):
    """Evaluate one temporal function value per parameter row."""
    v = v.reshape(-1, 1).to(device=param.device, dtype=param.dtype)
    if v.shape[0] != param.shape[0]:
        raise ValueError("v and param must have the same leading dimension")

    result = param.new_zeros(param.shape[:-1])
    offset = 0

    def bspline_basis(u, order):
        key = (order, param.device, param.dtype)
        if key not in POINTWISE_M:
            POINTWISE_M[key] = torch.as_tensor(
                get_deboor_cox_mat(order), device=param.device, dtype=param.dtype
            )
        matrix = POINTWISE_M[key]
        powers = torch.arange(order + 1, device=param.device, dtype=param.dtype)
        return (u ** powers) @ matrix

    def gather_controls(ctrl_param, start, count):
        indices = start + torch.arange(count, device=param.device)
        indices = indices[:, None, :].expand(-1, ctrl_param.shape[1], -1)
        return torch.gather(ctrl_param, dim=-1, index=indices)

    if order_args[0] != 0:
        ctrl_num, order = order_args[:2]
        interval = ctrl_num - order
        scaled_time = v * interval
        start = torch.floor(scaled_time).long().clamp(0, interval - 1)
        ctrl_pts = gather_controls(
            param[..., offset:offset + ctrl_num], start, order + 1
        )
        basis = bspline_basis(scaled_time - start, order)
        result = result + torch.sum(ctrl_pts * basis[:, None, :], dim=-1)
        offset += ctrl_num

    if order_args[2] != 0:
        order = order_args[2]
        powers = torch.arange(1, order + 1, device=param.device, dtype=param.dtype)
        basis = v ** powers
        result = result + torch.sum(
            param[..., offset:offset + order] * basis[:, None, :], dim=-1
        )
        offset += order

    if order_args[3] != 0:
        order = order_args[3]
        frequencies = torch.arange(
            1, order + 1, device=param.device, dtype=param.dtype
        ) * np.pi
        basis = torch.cat([torch.sin(v * frequencies), torch.cos(v * frequencies)], dim=-1)
        result = result + torch.sum(
            param[..., offset:offset + 2 * order] * basis[:, None, :], dim=-1
        )
        offset += 2 * order

    if order_args[4] != 0:
        ctrl_num, order = order_args[4:6]
        interval = ctrl_num - order
        scaled_time = v * interval
        start = torch.floor(scaled_time).long().clamp(0, interval - 1)
        ctrl_quat = gather_controls(
            param[..., offset:offset + ctrl_num], start, order + 1
        )
        identity = param.new_tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 4, 1)
        ctrl_quat = normalize((ctrl_quat + identity).permute(0, 2, 1), dim=-1)
        ctrl_quat = ctrl_quat[..., [1, 2, 3, 0]]

        basis = bspline_basis(scaled_time - start, order)
        cumulative_basis = torch.flip(
            torch.cumsum(torch.flip(basis, dims=(-1,)), dim=-1), dims=(-1,)
        )[..., 1:]
        relative = quat_product(
            quat_conjugation(ctrl_quat[:, :-1]), ctrl_quat[:, 1:]
        )
        increments = rotvec_to_unitquat(
            unitquat_to_rotvec(relative) * cumulative_basis[..., None]
        )
        value = ctrl_quat[:, 0]
        for index in range(increments.shape[1]):
            value = quat_product(value, increments[:, index])
        result = result + value[..., [3, 0, 1, 2]]

    return result
