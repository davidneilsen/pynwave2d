from abc import ABC, abstractmethod
import numpy as np
from . grid import Grid

class Equations(ABC):
    """
    Abstract base class for a system of PDEs.
    """

    def __init__(self, NU, g : Grid, apply_bc=None):
        """
        Initialize the PDE system.

        Parameters:
        NU : int
            The number of PDEs in the system
        grid : Grid
            The spatial grid

        apply_bc : "FUNCTION" or "RHS" or None
            Specifies how boundary conditions are applied, either in the 
            RHS routine, or applied to the function after each stage of 
            the time integrator.
        """

        self.Nu = NU
        self.Nx = g.Nx
        self.Ny = g.Ny
        self.u = []

        if apply_bc == "FUNCTION" or apply_bc == "function":
            self.apply_bc = "FUNCTION"
        elif apply_bc == "RHS" or apply_bc == "rhs":
            self.apply_bc = "RHS"
        else:
            self.apply_bc = None

        for i in range (NU):
            d = np.zeros((self.Nx, self.Ny))
            self.u.append(d)
 
    @abstractmethod
    def rhs(dtu, u, g : Grid):
        """
        The RHS update.
        """
        pass

    @abstractmethod
    def apply_bcs(u, g : Grid):
        """
        Routine to apply boundary conditions called from time integrator.
        """
        pass

    @abstractmethod
    def initialize(self, g : Grid, params):
        """
        Set the initial data
        """
        pass
