import numpy as np
from abc import ABC, abstractmethod

# from pdb import set_trace as bp
from . import finitederivs as fd
from . import filters as fi


class Grid(ABC):
    def __init__(self, shp, xi, dx, nghost=0):
        """
        Abstract class to define a grid for a PDE system.
        Parameters:
        """
        self.shp = shp
        self.xi = xi
        self.dx = dx
        self.D1 = None
        self.D2 = None
        self.num_filters = 0
        self.Filter = []
        self.nghost = nghost

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

    @abstractmethod
    def set_filter(self, filter):
        """
        Set the filter operator.
        """
        pass

    def get_shape(self):
        """
        Get the shape of the grid.
        Returns:
        -------
        tuple
            Shape of the grid.
        """
        return self.shp

    def get_nghost(self):
        """
        Get the number of ghost cells.
        Returns:
        -------
        int
            Number of ghost cells.
        """
        return self.nghost


class Grid1D(Grid):
    """
    Class to define a 1D grid for a PDE system.
    Parameters:
    ----------
    Nx : int
        Number of grid points in the x-direction.
    """

    def __init__(self, params, cell_centered=False):
        if "Nx" not in params:
            raise ValueError("Nx is required")

        nx = params["Nx"]
        ng = params.get("NGhost", 0)

        xmin = params.get("Xmin", 0.0)
        xmax = params.get("Xmax", 1.0)
        dx = (xmax - xmin) / (nx - 1)

        if cell_centered:
            xmax += 0.5 * dx
            if xmin < 0.0:
                xmin -= 0.5 * dx
                nx += 1
            else:
                xmin += 0.5 * dx

        print(f"Grid1D: nx={nx}, xmin={xmin}, xmax={xmax}, dx={dx}, ng={ng} cell_centered={cell_centered}")
        nx = nx + 2 * ng
        xmin -= ng * dx
        xmax += ng * dx

        shp = [nx]

        xi = [np.linspace(xmin, xmax, nx)]
        dxn = np.array([dx])

        # bp()
        super().__init__(shp, xi, dxn, ng)

    @classmethod
    def BH_grid(cls, nr, rmax):
        """
        Create a grid for a black hole simulation.
        Parameters:
        ----------
        nr : int
            Number of grid points in the radial direction.
        rmax : float
            Maximum radius of the grid.
        Returns:
        -------
        Grid1D
            A grid object with the specified parameters.
        """
        dr = rmax / (nr - 1)
        rmin = 0.5 * dr
        return cls({"Xmin": rmin, "Xmax": rmax, "Nx": nr})

    def set_D1(self, d1: fd.FirstDerivative1D):
        self.D1 = d1

    def set_D2(self, d2: fd.SecondDerivative1D):
        self.D2 = d2

    def set_filter(self, filter: fi.Filter1D):
        if isinstance(filter, list):
            for f in filter:
                self.num_filters += 1
                self.Filter.append(f)
        else:
            self.num_filters += 1
            self.Filter.append(filter)


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
        if "Nx" not in params:
            raise ValueError("Nx is required")
        if "Ny" not in params:
            raise ValueError("Ny is required")

        nx = params["Nx"]
        ny = params["Ny"]
        xmin = params.get("Xmin", 0.0)
        xmax = params.get("Xmax", 1.0)
        ymin = params.get("Ymin", 0.0)
        ymax = params.get("Ymax", 1.0)

        dx = (xmax - xmin) / (nx - 1)
        dy = (ymax - ymin) / (ny - 1)

        ng = params.get("NGhost", 0)
        nx = nx + 2 * ng
        ny = ny + 2 * ng
        xmin -= ng * dx
        xmax += ng * dx
        ymin -= ng * dy
        ymax += ng * dy

        shp = [nx, ny]

        xi = [np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)]

        dxn = np.array([dx, dy])
        #print(f"Grid2D: {shp}, {xi}, {dxn}")
        super().__init__(shp, xi, dxn, ng)

    def set_D1(self, d1: fd.FirstDerivative2D):
        self.D1 = d1

    def set_D2(self, d2: fd.SecondDerivative2D):
        self.D2 = d2

    def set_filter(self, filter: fi.Filter2D):
        self.num_filters += 1
        self.Filter.append(filter)
