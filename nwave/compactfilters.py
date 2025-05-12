from abc import ABC, abstractmethod
import numpy as np
from . grid import Grid


class Filter2D(ABC):
    """
    Abstract base class for a filters
    """

    @abstractmethod
    def filter_x():
        pass

    @abstractmethod
    def filter_y():
        pass


class CompactFilter2D(Filter2D):
    def __init__(self, x, y, type, use_banded=False):
        self.Nx = len(x)
        self.Ny = len(y)
        self.x = x
        self.y = y
        self.apply_filter = apply_filter
        self.filter_type = filter_type
        self.lusolve = use_banded
        self.F_x = CompactFilter(x, type, lusolve=use_banded)
        self.F_y = CompactFilter(y, type, lusolve=use_banded)
        super().__init__(dx, dy)

    def filter_x(self, u):
        # Apply compact scheme row-wise (derivative in x)
        return self.F_x.grad(u)

    def filter_y(self, u):
        # Apply compact scheme column-wise (derivative in y)
        du = self.F_y.grad(np.transpose(u))
        return np.transpose(du)
