from abc import ABC, abstractmethod
import numpy as np
from . grid import Grid

class Equations(ABC):
    def __init__(self, NU, g : Grid, bc_type=None):
        self.Nu = NU
        self.Nx = g.Nx
        self.Ny = g.Ny
        self.u = []
        self.boundary_type = bc_type

        for i in range (NU):
            d = np.zeros((self.Nx, self.Ny))
            self.u.append(d)
 
    @abstractmethod
    def rhs(dtu, u, g : Grid):
        pass

    @abstractmethod
    def apply_bcs(u, g : Grid):
        pass

    @abstractmethod
    def initialize(self, g : Grid, params):
        pass
