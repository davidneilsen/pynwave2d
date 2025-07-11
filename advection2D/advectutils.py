import numpy as np
from nwave import *

def get_filter_type(filter_str):
    try:
        return filter_type_map[filter_str]
    except KeyError:
        raise ValueError(f"Unknown filter type string: '{filter_str}'")


def get_filter_apply(filter_str):
    try:
        return filter_apply_map[filter_str]
    except KeyError:
        raise ValueError(f"Unknown filter apply string: '{filter_str}'")


def get_d1_type(d_str):
    try:
        return d1_type_map[d_str]
    except KeyError:
        raise ValueError(f"Unknown D1 type string: '{d_str}'")


def get_d2_type(d_str):
    try:
        return d2_type_map[d_str]
    except KeyError:
        raise ValueError(f"Unknown D2 type string: '{d_str}'")

def get_cfd_solve(d_str):
    try:
        return cfd_solve_map[d_str]
    except KeyError:
        raise ValueError(f"Unknown Deriv Solve type string: '{d_str}'")

def init_derivative_operators(x, y, params):
    d1type = get_d1_type(params["D1"])
    d2type = get_d2_type(params["D2"])
    method_str = params.get("DerivSolveMethod", "LUSOLVE")
    method = get_cfd_solve(method_str)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    print(f"init_derivative_operators>> Setting D1 type: {d1type}")
    if d1type == DerivType.D1_E44:
        D1 = ExplicitFirst44_2D(dx, dy)
    elif d1type == DerivType.D1_E642:
        D1 = ExplicitFirst642_2D(dx, dy)
    elif d1type in CompactFirstDerivatives:
        d1_x = NCompactDerivative.deriv(x, d1type, method)
        d1_y = NCompactDerivative.deriv(y, d1type, method)
        D1 = CompactFirst2D(x, y, d1_x, d1_y)
    else:
        raise NotImplementedError("D1 Type = {d1type}")

    print(f"init_derivative_operators>>  Setting D2 type: {d2type}")
    if d2type == DerivType.D2_E44:
        D2 = ExplicitSecond44_2D(dx, dy)
    elif d2type == DerivType.D2_E642:
        D2 = ExplicitSecond642_2D(dx, dy)
    elif d2type in CompactSecondDerivatives:
        d2_xx = NCompactDerivative.deriv(x, d2type, method)
        d2_yy = NCompactDerivative.deriv(y, d2type, method)
        D2 = CompactSecond2D(x, y, d2_xx, d2_yy)
    else:
        raise NotImplementedError("D2 Type = {d2type}")

    return D1, D2