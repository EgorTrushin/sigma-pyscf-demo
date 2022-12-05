#!/usr/bin/env python3

from pyscf import gto, dft
from pyscf.gw.usigma import USIGMA
from aug_cc_pwcvqz_mp2fit import AUXBASIS

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    [7 , (0. , 0. , 0.129649)],
    [1 , (0. , 0. ,-0.907543)]]
mol.basis = {"N": 'aug-cc-pwCVQZ', "H": 'aug-cc-pVQZ'}
mol.spin = 2
mol.build()

mf = dft.UKS(mol, xc='pbe').density_fit(auxbasis=AUXBASIS).run()

usigma = USIGMA(mf)
usigma.kernel(nw=50, w_scale=2.5)
print(f"RPA:   E_corr={usigma.e_corr_rpa:.10f}  E_tot={usigma.e_tot_rpa:.10f}")
print(f"SIGMA: E_corr={usigma.e_corr:.10f}  E_tot={usigma.e_tot:.10f}")

# Molpro values for similar setup:
# RPA:   E_corr=-0.3903602822  E_tot=-55.3670281989
# SIGMA: E_corr=-0.2954264966  E_tot=-55.2720944134