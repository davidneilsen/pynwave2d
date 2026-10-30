"""
Lapse gauges.

Bona-Masso slicing:  d_t alpha = beta d_r alpha - alpha^2 f(alpha) K.

A Bona-Masso gauge only supplies alpha2_f(alpha) = alpha^2 f(alpha); the
solver multiplies it by K.  Returning the pre-multiplied form keeps
rhs_gauge free of division by alpha.
"""

import numpy as np

from .registry import register_lapse


class BonaMasso:
    def alpha2_f(self, alpha):
        raise NotImplementedError

    def speed_inf(self):
        """
        Asymptotic gauge speed alpha sqrt(f(alpha)) at alpha = 1, i.e.
        sqrt(f(1)).  Used as the Sommerfeld speed for alpha at the outer
        boundary.
        """
        return np.sqrt(self.alpha2_f(1.0))

    def __repr__(self):
        args = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.name}({args})"


@register_lapse("1+log")
class OnePlusLog(BonaMasso):
    """f = k / alpha  (k = 2 is standard 1+log slicing)."""

    def __init__(self, k=2.0):
        if k <= 0.0:
            raise ValueError("1+log: k must be positive")
        self.k = k

    def alpha2_f(self, alpha):
        return self.k * alpha


@register_lapse("N1")
class N1(BonaMasso):
    """f = (2/alpha)(1 - alpha/2 + alpha^2/3)."""

    def alpha2_f(self, alpha):
        return alpha * (2.0 - alpha + (2.0 / 3.0) * alpha**2)


@register_lapse("N2")
class N2(BonaMasso):
    """f = (2/alpha)(1 - alpha + alpha^2 - alpha^3/4)."""

    def alpha2_f(self, alpha):
        return alpha * (2.0 - 2.0 * alpha + 2.0 * alpha**2 - 0.5 * alpha**3)


@register_lapse("N3")
class N3(BonaMasso):
    """f = (2/alpha)(1 - 3 alpha/2 + 2 alpha^2 - alpha^3 + alpha^4/5)."""

    def alpha2_f(self, alpha):
        return alpha * (
            2.0 - 3.0 * alpha + 4.0 * alpha**2 - 2.0 * alpha**3 + 0.4 * alpha**4
        )


@register_lapse("ShockFull")
class ShockFull(BonaMasso):
    """f = 1 + kappa / alpha^2  (shock-avoiding)."""

    def __init__(self, kappa=1.0):
        if kappa <= 0.0:
            raise ValueError("ShockFull: kappa must be positive")
        self.kappa = kappa

    def alpha2_f(self, alpha):
        return alpha**2 + self.kappa
