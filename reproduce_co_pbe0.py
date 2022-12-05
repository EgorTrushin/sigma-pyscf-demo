#!/usr/bin/env python3

from pyscf import gto, dft
from pyscf.gw.sigma import SIGMA
from aug_cc_pwcvqz_mp2fit import AUXBASIS

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    [6 , (0. , 0. ,-0.646514)],
    [8 , (0. , 0. , 0.484886)]]
mol.basis = 'aug-cc-pwCVQZ'
mol.build()

mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis=AUXBASIS).run()

sigma = SIGMA(mf)
sigma.kernel(nw=50, w_scale=2.5)
print(f"RPA:   E_corr={sigma.e_corr_rpa:.10f}  E_tot={sigma.e_tot_rpa:.10f}")
print(f"SIGMA: E_corr={sigma.e_corr:.10f}  E_tot={sigma.e_tot:.10f}")

# Molpro values for similar setup:
# RPA:   E_corr=-0.7481530934  E_tot=-113.5255931397
# SIGMA: E_corr=-0.5943104782  E_tot=-113.3717505245