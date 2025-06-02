import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import solve_banded, solve
from .compactderivs import CompactDerivative
from .types import *

# Finite difference base and subclasses


# NOTE: it is assumed that in the 2D case, meshgrid is called via "ij" ordering.
# This puts x in the "first" indexing dimension and y in the "second".


class FirstDerivative1D(ABC):
    def __init__(self, dx, aderiv=False):
        self.dx = dx
        self.HAVE_ADVECTIVE_DERIV = aderiv

    @abstractmethod
    def grad(self, u) -> np.ndarray:
        pass


class SecondDerivative1D(ABC):
    def __init__(self, dx):
        self.dx = dx

    @abstractmethod
    def grad2(self, u) -> np.ndarray:
        raise NotImplementedError


class FirstDerivative2D(ABC):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    @abstractmethod
    def grad_x(self, u) -> np.ndarray:
        pass

    @abstractmethod
    def grad_y(self, u) -> np.ndarray:
        pass


class SecondDerivative2D(ABC):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    @abstractmethod
    def grad_xx(self, u):
        raise NotImplementedError

    @abstractmethod
    def grad_yy(self, u):
        raise NotImplementedError


class CompactFirst1D(FirstDerivative1D):
    def __init__(self, x, type: DerivType, method: CFDSolve):
        self.Nx = len(x)
        self.x = x
        dx = x[1] - x[0]
        self.dx = dx
        self.deriv_type = type
        self.method = method
        self.dxf = CompactDerivative(x, type, method)
        super().__init__(dx, aderiv=False)

    def grad(self, u) -> np.ndarray:
        # Apply compact scheme row-wise (derivative in x)
        return self.dxf.grad(u)


class CompactSecond1D(SecondDerivative1D):
    def __init__(self, x, type: DerivType, method: CFDSolve):
        self.Nx = len(x)
        self.x = x
        dx = x[1] - x[0]
        self.dx = dx
        self.deriv_type = type
        self.method = method
        self.dxxf = CompactDerivative(x, type, method)
        super().__init__(dx)

    def grad2(self, u):
        # Apply compact scheme row-wise (derivative in x)
        return self.dxxf.grad(u)


class CompactFirst2D(FirstDerivative2D):
    def __init__(self, x, y, type: DerivType, method: CFDSolve):
        self.Nx = len(x)
        self.Ny = len(y)
        self.x = x
        self.y = y
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        self.dx = dx
        self.dy = dy
        self.deriv_type = type
        self.method = method
        self.dxf = CompactDerivative(x, type, method)
        self.dyf = CompactDerivative(y, type, method)
        super().__init__(dx, dy)

    def grad_x(self, u) -> np.ndarray:
        # Apply compact scheme row-wise (derivative in x)
        return self.dxf.grad(u)

    def grad_y(self, u):
        # Apply compact scheme column-wise (derivative in y)
        du = self.dyf.grad(np.transpose(u))
        return np.transpose(du)


class CompactSecond2D(SecondDerivative2D):
    def __init__(self, x, y, type: DerivType, method: CFDSolve):
        self.Nx = len(x)
        self.Ny = len(y)
        self.x = x
        self.y = y
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        self.dx = dx
        self.dy = dy
        self.deriv_type = type
        self.method = method
        self.dxxf = CompactDerivative(x, type, method)
        self.dyyf = CompactDerivative(y, type, method)
        super().__init__(dx, dy)

    def grad_xx(self, u):
        # Apply compact scheme row-wise (derivative in x)
        return self.dxxf.grad(u)

    def grad_yy(self, u):
        # Apply compact scheme column-wise (derivative in y)
        du = self.dyyf.grad(np.transpose(u))
        return np.transpose(du)
