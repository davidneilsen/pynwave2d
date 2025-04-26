import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import solve_banded, solve
from . compactderivs import CompactDerivative

# Finite difference base and subclasses


# NOTE: it is assumed that in the 2D case, meshgrid is called via "ij" ordering.
# This puts x in the "first" indexing dimension and y in the "second".


class FirstDerivative2D(ABC):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    @abstractmethod
    def grad_x(self, u):
        pass

    @abstractmethod
    def grad_y(self, u):
        pass


class SecondDerivative2D(ABC):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def grad_xx(self, u):
        raise NotImplementedError

    def grad_yy(self, u):
        raise NotImplementedError


class ExplicitFirst44_2D(FirstDerivative2D):
    def __init__(self, dx, dy):
        super().__init__(dx, dy)

    def grad_x(self, u):
        dudx = np.zeros_like(u)
        idx_by_12 = 1.0 / (12 * self.dx)

        # center stencil
        dudx[2:-2, :] = (
            -u[4:, :] + 8 * u[3:-1, :] - 8 * u[1:-3, :] + u[0:-4, :]
        ) * idx_by_12

        # 4th order boundary stencils
        dudx[0, :] = (
            -25 * u[0, :] + 48 * u[1, :] - 36 * u[2, :] + 16 * u[3, :] - 3 * u[4, :]
        ) * idx_by_12
        dudx[1, :] = (
            -3 * u[0, :] - 10 * u[1, :] + 18 * u[2, :] - 6 * u[3, :] + u[4, :]
        ) * idx_by_12
        dudx[-2, :] = (
            -u[-5, :] + 6 * u[-4, :] - 18 * u[-3, :] + 10 * u[-2, :] + 3 * u[-1, :]
        ) * idx_by_12
        dudx[-1, :] = (
            3 * u[-5, :] - 16 * u[-4, :] + 36 * u[-3, :] - 48 * u[-2, :] + 25 * u[-1, :]
        ) * idx_by_12

        return dudx

    def grad_y(self, u):
        dudy = np.zeros_like(u)
        idy_by_12 = 1.0 / (12 * self.dy)

        # center stencil
        dudy[:, 2:-2] = (
            -u[:, 4:] + 8 * u[:, 3:-1] - 8 * u[:, 1:-3] + u[:, 0:-4]
        ) * idy_by_12

        # 4th order boundary stencils
        dudy[:, 0] = (
            -25 * u[:, 0] + 48 * u[:, 1] - 36 * u[:, 2] + 16 * u[:, 3] - 3 * u[:, 4]
        ) * idy_by_12
        dudy[:, 1] = (
            -3 * u[:, 0] - 10 * u[:, 1] + 18 * u[:, 2] - 6 * u[:, 3] + u[:, 4]
        ) * idy_by_12
        dudy[:, -2] = (
            -u[:, -5] + 6 * u[:, -4] - 18 * u[:, -3] + 10 * u[:, -2] + 3 * u[:, -1]
        ) * idy_by_12
        dudy[:, -1] = (
            3 * u[:, -5] - 16 * u[:, -4] + 36 * u[:, -3] - 48 * u[:, -2] + 25 * u[:, -1]
        ) * idy_by_12

        return dudy


class ExplicitSecond44_2D(SecondDerivative2D):
    def __init__(self, dx, dy):
        super().__init__(dx, dy)

    def grad_xx(self, u):
        idx_sqrd = 1.0 / self.dx**2
        idx_sqrd_by_12 = idx_sqrd / 12.0

        dxxu = np.zeros_like(u)
        dxxu[2:-2, :] = (
            -u[4:, :] + 16 * u[3:-1, :] - 30 * u[2:-2, :] + 16 * u[1:-3, :] - u[0:-4, :]
        ) * idx_sqrd_by_12

        # boundary stencils
        dxxu[0, :] = (
            45 * u[0, :]
            - 154 * u[1, :]
            + 214 * u[2, :]
            - 156 * u[3, :]
            + 61 * u[4, :]
            - 10 * u[5, :]
        ) * idx_sqrd_by_12
        dxxu[1, :] = (
            10 * u[0, :]
            - 15 * u[1, :]
            - 4 * u[2, :]
            + 14 * u[3, :]
            - 6 * u[4, :]
            + u[5, :]
        ) * idx_sqrd_by_12
        dxxu[-2, :] = (
            u[-6, :]
            - 6 * u[-5, :]
            + 14 * u[-4, :]
            - 4 * u[-3, :]
            - 15 * u[-2, :]
            + 10 * u[-1, :]
        ) * idx_sqrd_by_12
        dxxu[-1, :] = (
            -10 * u[-6, :]
            + 61 * u[-5, :]
            - 156 * u[-4, :]
            + 214 * u[-3, :]
            - 154 * u[-2, :]
            + 45 * u[-1, :]
        ) * idx_sqrd_by_12
        return dxxu

    def grad_yy(self, u):
        idy_sqrd = 1.0 / self.dy**2
        idy_sqrd_by_12 = idy_sqrd / 12.0
        dyyu = np.zeros_like(u)

        # centered stencils
        dyyu[:, 2:-2] = (
            -u[:, 4:] + 16 * u[:, 3:-1] - 30 * u[:, 2:-2] + 16 * u[:, 1:-3] - u[:, 0:-4]
        ) * idy_sqrd_by_12

        # boundary stencils
        dyyu[:, 0] = (
            45 * u[:, 0]
            - 154 * u[:, 1]
            + 214 * u[:, 2]
            - 156 * u[:, 3]
            + 61 * u[:, 4]
            - 10 * u[:, 5]
        ) * idy_sqrd_by_12
        dyyu[:, 1] = (
            10 * u[:, 0]
            - 15 * u[:, 1]
            - 4 * u[:, 2]
            + 14 * u[:, 3]
            - 6 * u[:, 4]
            + u[:, 5]
        ) * idy_sqrd_by_12
        dyyu[:, -2] = (
            u[:, -6]
            - 6 * u[:, -5]
            + 14 * u[:, -4]
            - 4 * u[:, -3]
            - 15 * u[:, -2]
            + 10 * u[:, -1]
        ) * idy_sqrd_by_12
        dyyu[:, -1] = (
            -10 * u[:, -6]
            + 61 * u[:, -5]
            - 156 * u[:, -4]
            + 214 * u[:, -3]
            - 154 * u[:, -2]
            + 45 * u[:, -1]
        ) * idy_sqrd_by_12
        return dyyu


class CompactFirst2D(FirstDerivative2D):
    def __init__(self, x, y, type, use_banded=False):
        self.Nx = len(x)
        self.Ny = len(y)
        self.x = x
        self.y = y
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        self.dx = dx
        self.dy = dy
        self.deriv_type = type
        self.lusolve = use_banded
        self.dxf = CompactDerivative(x, type, lusolve=use_banded)
        self.dyf = CompactDerivative(y, type, lusolve=use_banded)
        super().__init__(dx, dy)

    def grad_x(self, u):
        # Apply compact scheme row-wise (derivative in x)
        return self.dxf.grad(u)

    def grad_y(self, u):
        # Apply compact scheme column-wise (derivative in y)
        du = self.dyf.grad(np.transpose(u))
        return np.transpose(du)

class CompactSecond2D(SecondDerivative2D):
    def __init__(self, x, y, type, use_banded=False):
        self.Nx = len(x)
        self.Ny = len(y)
        self.x = x
        self.y = y
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        self.dx = dx
        self.dy = dy
        self.deriv_type = type
        self.lusolve = use_banded
        self.dxxf = CompactDerivative(x, type, lusolve=use_banded)
        self.dyyf = CompactDerivative(y, type, lusolve=use_banded)
        super().__init__(dx, dy)

    def grad_xx(self, u):
        # Apply compact scheme row-wise (derivative in x)
        return self.dxxf.grad(u)

    def grad_yy(self, u):
        # Apply compact scheme column-wise (derivative in y)
        du = self.dyyf.grad(np.transpose(u))
        return np.transpose(du)

