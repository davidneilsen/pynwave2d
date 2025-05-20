from .eqs import Equations
from .grid import Grid
import numpy as np


class RK4:
    def __init__(self, e: Equations, g: Grid):
        self.k1 = []
        self.k2 = []
        self.k3 = []
        self.k4 = []
        self.us = []

        for i in range(e.Nu):
            self.k1.append(np.zeros(tuple(e.shp)))
            self.k2.append(np.zeros(tuple(e.shp)))
            self.k3.append(np.zeros(tuple(e.shp)))
            self.k4.append(np.zeros(tuple(e.shp)))
            self.us.append(np.zeros(tuple(e.shp)))

    def step(self, e: Equations, g: Grid, dt):
        nu = len(e.u)
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4
        us = self.us
        u0 = e.u

        assert len(k4) == e.Nu, "RK: wrong number of work arrays"
        # print(f"k3 shape = {k3[0].shape}")

        # Stage 1
        e.rhs(k1, u0, g)
        for i in range(nu):
            us[i][:] = u0[i][:] + 0.5 * dt * k1[i][:]
        if e.apply_bc == "FUNCTION":
            e.apply_bcs(self.us, g)

        # Stage 2
        e.rhs(k2, us, g)
        for i in range(nu):
            us[i][:] = u0[i][:] + 0.5 * dt * k2[i][:]
        if e.apply_bc == "FUNCTION":
            e.apply_bcs(us, g)

        # Stage 3
        e.rhs(k3, us, g)
        for i in range(nu):
            us[i][:] = u0[i][:] + dt * k3[i][:]
        if e.apply_bc == "FUNCTION":
            e.apply_bcs(us, g)

        # Stage 4
        e.rhs(self.k3, us, g)
        for i in range(nu):
            u0[i][:] += dt / 6 * (k1[i][:] + 2 * k2[i][:] + 2 * k3[i][:] + k4[i][:])
        if e.apply_bc == "FUNCTION":
            e.apply_bcs(u0, g)
