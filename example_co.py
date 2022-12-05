#!/usr/bin/env python3

from pyscf import gto, dft
from pyscf.gw.sigma import SIGMA

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    [6 , (0. , 0. ,-0.646514)],
    [8 , (0. , 0. , 0.484886)]]
mol.basis = 'aug-cc-pwCVTZ'
mol.build()

mf = dft.RKS(mol)
mf.xc = 'pbe'
mf.kernel()

sigma = SIGMA(mf)
sigma.kernel(nw=20, w_scale=2.5)
print(f"RPA:   E_corr={sigma.e_corr_rpa:.10f}  E_tot={sigma.e_tot_rpa:.10f}")
print(f"SIGMA: E_corr={sigma.e_corr:.10f}  E_tot={sigma.e_tot:.10f}")
