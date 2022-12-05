#!/usr/bin/env python3

from pyscf import gto, dft
from pyscf.gw.usigma import USIGMA

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    [7 , (0. , 0. , 0.129649)],
    [1 , (0. , 0. ,-0.907543)]]
mol.basis = {"N": 'aug-cc-pwCVTZ', "H": 'aug-cc-pVTZ'}
mol.build()

mf = dft.UKS(mol)
mf.xc = 'pbe'
mf.spin = 2
mf.kernel()

usigma = USIGMA(mf)
usigma.kernel(nw=20, w_scale=2.5)
print(f"RPA:   E_corr={usigma.e_corr_rpa:.10f}  E_tot={usigma.e_tot_rpa:.10f}")
print(f"SIGMA: E_corr={usigma.e_corr:.10f}  E_tot={usigma.e_tot:.10f}")