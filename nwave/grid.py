import numpy as np
from abc import ABC, abstractmethod
from . import finitederivs as fd

class Grid(ABC):
    def __init__(self, shp, xi, dx):
        """
        Abstract class to define a grid for a PDE system.
        Parameters: 
        """
        self.shp = shp
        self.xi = xi
        self.dx = dx
        self.D1 = None
        self.D2 = None

    @abstractmethod
    def set_D1(self, d1):
        """
        Set the first derivative operator.
        """
        pass
    @abstractmethod
    def set_D2(self, d2):
        """
        Set the second derivative operator.
        """
        pass

class Grid1D(Grid):
    """
    Class to define a 1D grid for a PDE system.
    Parameters:
    ----------
    Nx : int
        Number of grid points in the x-direction.
    """
    def __init__(self, params):
        xi = [np.linspace(params["Xmin"], params["Xmax"], params["Nx"])]
        shp = [params["Nx"]]
        dxn = np.array([x[1] - x[0]])
        super().__init__(shp, xi, dxn)

    def set_D1(self, d1: fd.FirstDerivative1D):
        self.D1 = d1

    def set_D2(self, d2: fd.SecondDerivative1D):
        self.D2 = d2

class Grid2D(Grid):
    """
    Class to define a 2D grid for a PDE system.
    Parameters:
    ----------
    Nx : int
        Number of grid points in the x-direction.
    Ny : int
        Number of grid points in the y-direction.
    """
    def __init__(self, params):
        shp = [params["Nx"], params["Ny"]]
        xi = [ np.linspace(params["Xmin"], params["Xmax"], params["Nx"]),
                       np.linspace(params["Ymin"], params["Ymax"], params["Ny"]) ]
        dxn = np.array([xi[0][1] - xi[0][0], xi[1][1] - xi[1][0]])
        print(f"Grid2D: {shp}, {xi}, {dxn}")
        super().__init__(shp, xi, dxn)

    def set_D1(self, d1: fd.FirstDerivative2D):
        self.D1 = d1

    def set_D2(self, d2: fd.SecondDerivative2D):
        self.D2 = d2
