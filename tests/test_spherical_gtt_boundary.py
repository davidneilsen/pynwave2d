"""Background-preserving Sommerfeld on normalized angular metric deviation."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'bssn'),str(ROOT)]
from bssneqs import BSSN
from nwave import Grid1D,ExplicitFirst642_1D,ExplicitSecond642_1D,FilterApply

class SphericalGTTBoundaryTests(unittest.TestCase):
 def grid(self):
  g=Grid1D({'Nx':101,'Xmin':-20.,'Xmax':20.,'NGhost':3},cell_centered=True)
  g.set_D1(ExplicitFirst642_1D(g.dx[0]));g.set_D2(ExplicitSecond642_1D(g.dx[0]));return g
 def test_flat_background_zero(self):
  g=self.grid();r=g.xi[0];b=r*r;q=b/r**2-1.;rhs=np.full_like(r,123.)
  BSSN.bc_spherical_gtt(rhs,q,g.D1.grad(q),r,1.,3,True)
  np.testing.assert_array_equal(rhs[:3],0.);np.testing.assert_array_equal(rhs[-3:],0.)
  np.testing.assert_array_equal(rhs[3:-3],123.)
 def test_exact_outgoing_wave_and_signed_parity(self):
  r=np.r_[-np.linspace(22.,18.,8),np.linspace(18.,22.,8)];R=abs(r);ix=np.r_[0:3,13:16]
  for v in (1.,np.sqrt(2.)):
   z=.37*(R-v*.2);q=.03*np.sin(z)/R
   dq=np.sign(r)*(.03*.37*np.cos(z)/R-q/R)
   rhs=np.full_like(r,123.)
   BSSN.bc_spherical_gtt(rhs,q,dq,r,v,3,True)
   want=-v*.03*.37*R*np.cos(z)
   np.testing.assert_allclose(rhs[ix],want[ix],atol=1e-14,rtol=1e-13)
   np.testing.assert_allclose(rhs,rhs[::-1],atol=1e-14)
   np.testing.assert_array_equal(rhs[3:-3],123.)
 def test_half_domain_keeps_inner_untouched(self):
  r=np.linspace(.5,20.,30);rhs=np.full_like(r,123.)
  BSSN.bc_spherical_gtt(rhs,r*0,r*0,r,1.,3,False)
  np.testing.assert_array_equal(rhs[:-3],123.);np.testing.assert_array_equal(rhs[-3:],0.)
 def check_actual_rhs(self,filtered):
  g=self.grid();r=g.xi[0]
  if filtered:
   g.num_filters=1;g.Filter=[SimpleNamespace(apply_filter=FilterApply.APPLY_DERIVS,filter=lambda x:.9*x)]
  u=[np.zeros_like(r) for _ in range(9)]
  u[0][:]=.8;u[1][:]=1.;u[2][:]=r*r*(1+.002*np.cos(.2*r));u[5][:]=-2/r;u[6][:]=1.
  eq=BSSN(g,1.,2.,True)
  for a in eq.u:a[:]=999. # stored state must never replace supplied RK stage
  out=[np.zeros_like(r) for _ in range(9)];eq.rhs(out,u,g)
  q=u[2]/r**2-1.;want=out[2].copy()
  BSSN.bc_spherical_gtt(want,q,g.D1.grad(.9*q if filtered else q),r,1.,3,True)
  np.testing.assert_array_equal(out[2],want)
  # Remove only the new outer update, leaving the bulk evaluation untouched.
  original=BSSN.bc_spherical_gtt
  try:
   BSSN.bc_spherical_gtt=staticmethod(lambda *args:None)
   bulk=[np.zeros_like(r) for _ in range(9)];eq.rhs(bulk,u,g)
  finally:BSSN.bc_spherical_gtt=staticmethod(original)
  for k in range(9):
   np.testing.assert_array_equal(out[k][3:-3],bulk[k][3:-3])
   if k!=2:np.testing.assert_array_equal(out[k],bulk[k])
 def test_actual_rhs_stage_and_bulk(self):self.check_actual_rhs(False)
 def test_actual_rhs_derivative_filter_policy(self):self.check_actual_rhs(True)

if __name__=='__main__':unittest.main()
