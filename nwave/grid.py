import numpy as np
from . import finitederivs as fd


class Grid:
    def __init__(self, params):
        self.Nx = params["Nx"]
        self.Ny = params["Ny"]
        self.x = np.linspace(params["Xmin"], params["Xmax"], self.Nx)
        self.y = np.linspace(params["Ymin"], params["Ymax"], self.Ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.D1 = None
        self.D2 = None

    def set_D1(self, d1: fd.FirstDerivative2D):
        self.D1 = d1

    def set_D2(self, d2: fd.SecondDerivative2D):
        self.D2 = d2
