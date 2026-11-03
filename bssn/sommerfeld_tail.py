"""Optional one-anchor R^-3 correction to the Sommerfeld boundary RHS.

For X (or q = gtt/r**2 - 1), continue the nonradiative residual
E = bulk_RHS - Sommerfeld_RHS from the nearest physical interior point:
RHS(R) = Sommerfeld_RHS(R) + (R_anchor/R)**3 * E_anchor.
Geometry is cached per grid; only boundary cells are written.
"""
import numpy as np
from numba import njit


def geometry(r, ng, extended_domain):
    """Return boundary indices, one physical anchor per side, and R^-3 weights."""
    if ng < 1 or len(r) < 2 * ng + 1:
        raise ValueError("Insufficient physical points for Sommerfeld tail")
    boundaries = [np.arange(len(r) - ng, len(r))]
    anchors = [len(r) - ng - 1]
    if extended_domain:
        boundaries.append(np.arange(ng))
        anchors.append(ng)
    boundaries = np.asarray(boundaries, dtype=np.int64)
    anchors = np.asarray(anchors, dtype=np.int64)
    rb, ra = np.abs(r[boundaries]), np.abs(r[anchors])
    if np.any(rb == 0) or np.any(ra == 0):
        raise ValueError("Tail boundary and anchors must have nonzero radius")
    return boundaries, anchors, (ra[:, None] / rb)**3


@njit(cache=True)
def apply(dtu, x, dx, r, power, asymptote, speed,
          boundaries, anchors, weights, spherical_gtt=False):
    """Apply a tail using current-stage ordinary derivatives and bulk RHS.

    For spherical_gtt, x and dx are q and Dq; RHS units are converted here.
    Physical anchors are never overwritten by this routine.
    """
    for side in range(anchors.shape[0]):
        a = anchors[side]
        R = abs(r[a])
        S = -speed * (np.sign(r[a]) * dx[a] + power * (x[a] - asymptote) / R)
        bulk = dtu[a] / (R * R) if spherical_gtt else dtu[a]
        residual = bulk - S
        for j in range(boundaries.shape[1]):
            b = boundaries[side, j]
            R = abs(r[b])
            value = -speed * (np.sign(r[b]) * dx[b] + power * (x[b] - asymptote) / R)
            value += weights[side, j] * residual
            dtu[b] = R * R * value if spherical_gtt else value
