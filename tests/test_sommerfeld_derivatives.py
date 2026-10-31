"""Radiation derivatives must not inherit beta-directed upwind choices."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'bssn'),str(ROOT)]
from bssneqs import BSSN,BSSNOuterBCType
from nwave import Grid1D,ExplicitFirst642_1D,ExplicitSecond642_1D,FilterApply

class RadiationDerivativeTests(unittest.TestCase):
 def check(self,filtered):
  g=Grid1D({'Nx':161,'Xmin':-20.,'Xmax':20.,'NGhost':3},cell_centered=True)
  r=g.xi[0];N=len(r)//2;r[:N]=-r[-1:-N-1:-1]
  g.set_D1(ExplicitFirst642_1D(g.dx[0]));g.set_D2(ExplicitSecond642_1D(g.dx[0]))
  if filtered:
   g.num_filters=1;g.Filter=[SimpleNamespace(apply_filter=FilterApply.APPLY_DERIVS,filter=lambda x:.9*x)]
  eq=BSSN(g,1.,2.,True);bulk_eq=BSSN(g,1.,2.,True)
  u=[np.zeros_like(r) for _ in range(9)]
  u[0][:]=1.;u[1][:]=1.;u[2][:]=r*r;u[3][:]=1e-8*r**4;u[5][:]=-2/r;u[6][:]=1.;u[8][:]=1e-8*r**3
  u[3][:N]=u[3][-1:-N-1:-1];u[8][:N]=-u[8][-1:-N-1:-1]
  for slope in (0.,1e-6,-1e-6):
   u[7][:]=slope*r
   out=[np.zeros_like(r) for _ in range(9)];bulk=[np.zeros_like(r) for _ in range(9)]
   eq.rhs(out,u,g)
   old_generic,old_gtt=BSSN.bc_sommerfeld,BSSN.bc_spherical_gtt
   try:
    BSSN.bc_sommerfeld=staticmethod(lambda *args:None)
    BSSN.bc_spherical_gtt=staticmethod(lambda *args:None)
    bulk_eq.rhs(bulk,u,g)
   finally:
    BSSN.bc_sommerfeld=staticmethod(old_generic)
    BSSN.bc_spherical_gtt=staticmethod(old_gtt)
   for k in range(9):np.testing.assert_array_equal(out[k][3:-3],bulk[k][3:-3])
   for k,parity in ((3,1),(8,-1)):
    want=out[k].copy();D=g.D1.grad(.9*u[k] if filtered else u[k])
    BSSN.bc_sommerfeld(want,u[k],D,r,eq.u_falloff[k],eq.u_inf[k],eq.u_wavespeed[k],3,True)
    np.testing.assert_array_equal(out[k],want)
    np.testing.assert_allclose(out[k][:3],parity*out[k][-1:-4:-1],atol=2e-18,rtol=0.)
 def test_zero_and_signed_shift_parity(self):self.check(False)
 def test_derivative_filter_policy(self):self.check(True)

if __name__=='__main__':unittest.main()
