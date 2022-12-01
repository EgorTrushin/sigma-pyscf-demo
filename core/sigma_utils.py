#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Egor Trushin <e.v.trushin@gmail.com>
#

"""
Auxiliary stuff used in spin-restricted and spin-unrestricted
sigma-functional (sigma.py and usigma.py).
"""

def get_spline_coeffs(logger, rpa):
    assert rpa._scf.xc in ["pbe", "pbe0", "b3lyp", "tpss"]
    if rpa.param is None and rpa._scf.xc in ["pbe", "pbe0"]:
        param_name = rpa._scf.xc + "_s2"
    elif rpa.param is None and rpa._scf.xc in ["b3lyp", "tpss"]:
        param_name = rpa._scf.xc + "_w1"
    else:
        param_name = rpa._scf.xc + "_" + rpa.param
    assert param_name in params.keys()
    logger.debug(rpa, "  Parametrization for sigma-functional: %s", param_name)
    return params[param_name]["x"], params[param_name]["c"]


def intervalnum(x, s):
    """Determine to which interval s belongs in x.

    Args:
        x: List of non-negative real numbers in increasing order.
        s: Positive real number.

    Returns:
        inum: The number of interval.
    """

    assert s > 0.  # verify that s is positive
    assert all(i > -1e-24 for i in x)  # verify that all x-values are positive
    assert x == sorted(x)  # verify that x-values are in increasing order

    # find interval
    inum = -1
    if s > x[-1]:
        inum = len(x)
    for i in range(0, len(x)-1):
        if (s > x[i] and (s <= x[i+1])):
            inum = i+1

    assert inum != -1  # verify that an interval was determined

    return inum


def cspline_integr(c, x, s):
    """Integrate analytically cubic spline representation of sigma-functional
       'correction' from 0 to s.

    First interval of spline is treated as linear.
    Last interval of spline is treated as a constant.

    Args:
        c: Coefficients of spline
        x: Ordinates of spline. Have to be non-negative and increasingly order
        s: Sigma-value for which one integrate. Has to be positive.

    Returns:
        integral: resulting integral
    """
    m = intervalnum(x, s)  # determine to which interval s belongs

    # evaluate integral
    integral = 0.
    if m == 1:
        integral = 0.5*c[1][0]*s
    if m > 1 and m < len(x):
        h = s-x[m-1]
        integral = 0.5*c[1][0]*x[1]**2/s + (c[0][m-1]*h + c[1][m-1]/2.*h**2 + c[2][m-1]/3.*h**3 + c[3][m-1]/4.*h**4)/s
        for i in range(2, m):
            h = x[i]-x[i-1]
            integral += (c[0][i-1]*h + c[1][i-1]/2.*h**2 + c[2][i-1]/3.*h**3 + c[3][i-1]/4.*h**4)/s
    if m == len(x):
        integral = 0.5*c[1][0]*x[1]**2/s
        for i in range(2, m):
            h = x[i]-x[i-1]
            integral += (c[0][i-1]*h + c[1][i-1]/2.*h**2 + c[2][i-1]/3.*h**3 + c[3][i-1]/4.*h**4)/s
        integral += c[0][-1]*(1.-x[-1]/s)

    return integral*s


# Parametrization P1 from JCP 154, 014104 (2021) designed to be applied
# on top of PBE reference calculation
pbe_p1 = {
    "x": [0.0e+00, 1.0e-04, 1.0e-03, 1.0e-02, 1.0e-2*10.0**(1./2.), 1.0e-01, 1.0e-1*10.0**(1./4.), 1.0e-1*10.0**(2./4.), 1.0e-1*10.0**(3./4.), 1.0e+00, 10.0**(1./3.), 10.0**(2./3.), 10.0],
    "c": [[0., -5.885072220000e-05, -6.644982900000e-04, -1.180579200000e-02, -1.570844509420e-02, 1.760985050000e-01, 9.814551470161e-02, -3.628324496584e-02, -4.247178370737e-02, -4.94407477000e-02, -9.490347223423e-03, 4.773528498951e-03, 0.], [-5.885072220000e-01, -5.885072220000e-01, -7.606856690986e-01, -1.281859889831e+00, 1.086808331529e+00, 1.709528474395e+00, -1.974002322512e+00, -3.286910860598e-01, 7.951658597067e-02, -3.924864826146e-02, 5.362175675285e-02, -1.285966486385e-02, 0.], [0., -9.013901433489e+01, -1.011703713302e+02, 4.326212458218e+01, 6.628292899688e+01, -5.717580074689e+01, 9.846644284656e+00, 2.041457714737e+00, -3.828425884461e-01, 1.114776159963e-01, -3.103096464721e-02, 4.301048534742e-03, 0.], [0., -4.085687776044e+03, 5.349351700459e+03, 3.548851108681e+02, -6.018511409108e+02, 2.870539130593e+02, -1.879864267456e+01, -3.283444048247e+00, 3.764883708805e-01, -4.114815729697e-02, 4.735266512417e-03, -3.858229788514e-04, 0e0]]}

pbe0_w1 = {
    "x": [0.0, 1e-05, 0.0001, 0.000215443469, 0.000464158883, 0.001, 0.00177827941, 0.00316227766, 0.00562341325, 0.01, 0.0177827941, 0.0316227766, 0.0562341325, 0.1, 0.177827941, 0.316227766, 0.562341325, 1.0, 1.77827941, 2.61015722, 10.0, 21.5443469],
    "c": [[0.0, 0.0, -0.00016359410359, -0.000679644651516, -0.00206305566905, -0.00538487736264, -0.010345840735, -0.0168414630497, -0.0204773827189, -0.0136547635644, -0.00615544718661, -0.073486462515, 0.00824090490068, 0.117940542727, 0.149367614907, -0.0403837739134, -0.00566051777937, 0.05523743, 0.0392254426171, 0.0899646367376, 0.0, 0.0], [0.0, 0.0, -2.54025083646, -4.89179427394, -5.8250050236, -6.28017499439, -5.4839506684, -2.362249928, 0.0, 1.21782248423, 0.0, 0.0, 2.89449672543, 0.745869971141, 0.0, 0.0, 0.140199554463, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.0, -32365.3994429, -29782.3525233, -4334.59468098, -1245.91667249, -1385.76248969, -541.888483103, 118.846515312, 790.30737233, 58.4731002077, -1054.5455328, 287.170551613, 22.4985882, -3.60197342952, -29.7190646015, 1.15011706043, 0.313110259185, -0.0793040582103, 0.219961027743, -0.00494222698634, -0.0, 0.0], [0.0, 135206628.118, 113172424.28, 6589950.31995, 1021685.83769, 1625201.30715, 804274.604886, 97804.1865691, -99191.1668821, -11710.5524995, 50797.0552129, -6185.93746028, -716.622677854, -10.1918364203, 143.155598169, -2.34387763193, -0.72092745192, 0.0679310944898, -0.176276712036, 0.000445857657461, 0.0, 0.0]]}

b3lyp_w1 = {
    "x": [0.0, 1e-05, 0.0001, 0.000215443469, 0.000464158883, 0.001, 0.00177827941, 0.00316227766, 0.00562341325, 0.01, 0.0177827941, 0.0316227766, 0.0562341325, 0.1, 0.177827941, 0.316227766, 0.562341325, 1.0, 1.77827941, 2.61015722, 10.0, 21.5443469],
    "c": [[0.0, 0.0, -0.000137219924008, -0.000564174402221, -0.0016952600717, -0.00445126004995, -0.00881950690944, -0.0152445457422, -0.0209493408452, -0.0208768163366, -0.0208621661135, -0.082562362571, 0.0390526593565, 0.152691916364, 0.120069723833, -0.011429667572, -0.034394942292, 0.04620382, 0.0326800355187, 0.0610967959278, 0.0, 0.0], [0.0, 0.0, -2.12272836088, -4.0286897778, -4.79128762008, -5.35339840091, -5.12692821021, -3.19156353876, 0.0, 0.0036521914177, 0.0, 0.0, 3.50609352998, 0.0, -0.561370960807, -0.184046397209, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.0, -27236.3234004, -24436.2951908, -3194.26908947, -921.864998465, -1290.49711906, -348.053147786, -231.897043984, 10.5243818063, -0.21293440767, -966.35504796, 459.875658562, 17.7625626878, -8.94412192522, -11.1534506587, 0.358199252047, 1.26234639741, -0.0669805044232, 0.12319036463, -0.00335636584066, -0.0, 0.0], [0.0, 114395467.127, 93444521.6758, 4452735.54761, 494366.005567, 1230056.32087, 504454.621957, 238450.768157, -1539.57721034, -1.85862973587, 46548.9532693, -10527.565066, -880.711825842, 45.7216655248, 60.2920760359, 0.0425445558964, -1.92287806232, 0.0573748566925, -0.0987247270764, 0.000302790910946, 0.0, 0.0]]}

tpss_w1 = {
    "x": [0.0, 1e-05, 0.0001, 0.000215443469, 0.000464158883, 0.001, 0.00177827941, 0.00316227766, 0.00562341325, 0.01, 0.0177827941, 0.0316227766, 0.0562341325, 0.1, 0.177827941, 0.316227766, 0.562341325, 1.0, 1.77827941, 3.16227766, 5.62341325, 15.3992653, 31.6227766],
    "c": [[0.0, 0.0, 0.000146798731127, 0.000574878906062, 0.00154055131096, 0.00314892637363, 0.00361072227263, 0.000735868971744, -0.00901096557066, -0.0307032486139, -0.0682643373448, -0.0891052789694, 0.0994712453012, 0.2053601, 0.0513852204369, 0.0632811675468, -0.0890738739423, -0.069038095, 0.00272741297389, -0.0472452045191, 0.0265078785377, 0.0, 0.0], [0.0, 0.0, 2.22980820374, 3.78277406131, 3.43943397868, 1.03341235903, 0.0, -2.64796787242, -4.35727350647, -4.89652548865, -2.41365453601, 0.0, 3.86531297928, 0.0, 0.0, 0.0, 0.0, 0.0593195958219, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.0, 29594.2537092, 24964.7722458, 2585.03390026, 2038.82849848, -368.453827753, -2589.35185354, -905.147145166, -287.494126008, -291.900950983, 22.3814018522, 776.926436706, -10.7917068055, -76.2605697829, 1.86315590548, -7.54582976226, 0.178264013546, 0.203001913958, -0.0782676457237, 0.0365283750212, -0.000832121965427, -0.0, 0.0], [0.0, -127455039.668, -105325341.933, -8779138.7367, -5329835.04783, -253083.484333, 786472.503057, 151119.29245, 34408.4990927, 38667.4503442, 3122.22024991, -18918.0804017, -508.269491627, 653.240715379, -8.97475077723, 20.439967612, -0.16831193652, -0.206533675729, 0.0377012257656, -0.00989472100322, 5.67467647949e-05, 0.0, 0.0]]}

pbe_w1 = {
    "x": [0.0, 1e-05, 0.0001, 0.000215443469, 0.000464158883, 0.001, 0.00177827941, 0.00316227766, 0.00562341325, 0.01, 0.0177827941, 0.0316227766, 0.0562341325, 0.1, 0.177827941, 0.316227766, 0.562341325, 1.0, 1.77827941, 3.16227766, 5.62341325, 15.3992653, 31.6227766],
    "c": [[0.0, 0.0, 1.4684179582e-05, 3.5594722238e-05, -3.71192689381e-05, -0.000641360939061, -0.00255216217181, -0.00670235407381, -0.0134545320809, -0.0242715093069, -0.0461787575633, -0.0896767885692, 0.0342837410852, 0.224951854545, 0.124376029194, 0.0564118372586, -0.101942856761, -0.01613392, -0.0127751725427, -0.0153560580635, -0.0290083965652, 0.0, 0.0], [0.0, 0.0, 0.171306505377, 0.0, -0.433243270713, -1.51104029702, -2.67495140333, -2.87737928116, -2.6131720056, -2.61618455996, -2.95465031755, 0.0, 4.70381335331, 0.0, -0.742860478947, -0.550118664559, 0.0, 0.00927399359423, 0.0, -0.00266751515726, 0.0, 0.0, 0.0], [-0.0, 3535.17941509, 1739.24027457, -1784.4946853, -1876.35021369, -2143.76854534, -555.499257099, 55.8142973786, 97.7615610264, -33.0823157893, -254.297665006, 422.827139998, 83.672922579, -40.268216606, 4.06521833899, -3.37253431809, 1.32275869041, -0.00719686076381, -0.0021148126071, -0.00459400974873, 0.000910616967308, -0.0, 0.0], [0.0, -19136863.7588, -14328456.1334, 2448669.6867, 1083212.16548, 1195819.24119, 232354.680061, -579.247398297, -14944.0243023, 971.187372896, 17391.2044223, -8864.88584944, -2093.12731864, 304.053296034, -16.2278426521, 12.1628021074, -1.99876259469, 0.00106118734554, 0.000554487076904, 0.0013912107569, -6.2099751009e-05, 0.0, 0.0]]}

pbe0_s1 = {
    "x": [0.0, 1e-05, 0.0001, 0.001, 0.00215443469, 0.00464158883, 0.01, 0.0146779926762, 0.0215443469003, 0.0316227766017, 0.0464158883361, 0.0681292069058, 0.1, 0.158489319246, 0.251188643151, 0.398107170553, 0.63095734448, 1.0, 2.61015721568, 10.0, 21.5443469],
    "c": [[0.0, 0.0, -0.000149500660756, -0.00353017276233, -0.0109810247734, -0.0231246943777, -0.0268999962858, -0.000634751994007, 0.011879289247, -0.0473431931326, -0.0817589390539, 0.0125726011069, 0.108028492092, 0.193548206759, 0.0358395561305, -0.0497714974829, 0.0341059348835, 0.0341050720155, 0.0785549033229, 0.0, 0.0], [0.0, 0.0, -2.08376539581, -4.69755869285, -5.65503803415, -1.35502867642, 0.0, 2.84340701746, 0.0, -3.42695931351, 0.0, 3.58739081268, 2.0336880613, 0.0, -0.901387663218, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.0, -32217.6662524, -2670.90835643, -3735.3206735, -797.121299, 111.299540119, 2992.84621116, -31.9333485618, -1409.10103454, -8.48330431187, 435.025012278, -7.00327539634, 5.45486142353, -45.3346282407, 0.37192102791, 4.64101795796, -1.90069531714e-05, 0.051434533666, -0.00431543078188, -0.0, 0.0], [0.0, 152897717.268, 902815.532735, 1917604.93084, 445372.471512, 1883.62654331, -383203.258784, -17002.7418959, 81962.9330224, 5602.28610945, -10820.3002413, -363.378668069, -260.332257619, 291.068208088, 12.2322834276, -13.287565647, 3.43356030115e-05, -0.0212958640167, 0.000389311916174, 0.0, 0.0]]}

pbe0_s2 = {
    "x": [0.0, 1e-05, 0.0001, 0.001, 0.00215443469, 0.00464158883, 0.01, 0.0146779926762, 0.0215443469003, 0.0316227766017, 0.0464158883361, 0.0681292069058, 0.1, 0.158489319246, 0.251188643151, 0.398107170553, 0.63095734448, 1.0, 2.61015721568, 10.0, 21.5443469],
    "c": [[0.0, 0.0, -4.31405252048e-05, -0.00182874853131, -0.00852003132762, -0.0218177403992, -0.0305777654735, -0.00870882903969, 0.0137878988102, -0.028435200744, -0.0798812002431, -0.00334010771574, 0.0934182748715, 0.204960802253, 0.0213204380281, -0.0401220283037, 0.0321629738336, 0.0321618301891, 0.0808763912948, 0.0, 0.0], [0.0, 0.0, -0.661870777583, -2.8975291259, -5.58979946652, -2.6776570454, 0.0, 3.89592612611, 0.0, -3.82296397421, 0.0, 3.27772498106, 2.3963372431, 0.0, -0.726304793204, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.0, -8623.85254713, -1923.06222883, -5200.47462362, -877.473657666, 84.1408344046, 2165.16760964, 296.702212913, -867.733655494, -188.41005538, 336.084151111, 4.89746728744, 15.8746877181, -56.2764882273, 1.34759277149, 3.99959778866, -2.51917983154e-05, 0.056369409276, -0.00444296223097, -0.0, 0.0], [0.0, 36642908.679, 504466.528222, 2329809.23705, 392124.287301, 20617.3887726, -249217.659838, -56351.9876566, 44853.0826095, 14314.0667434, -8001.44415404, -391.685311241, -414.433988077, 376.550449117, 5.10124747789, -11.4511339236, 4.55083767664e-05, -0.0233390912502, 0.00040081702779, 0.0, 0.0]]}

pbe_s1 = {
    "x": [0.0, 1e-05, 0.0001, 0.001, 0.00215443469, 0.00464158883, 0.01, 0.0146779926762, 0.0215443469003, 0.0316227766017, 0.0464158883361, 0.0681292069058, 0.1, 0.158489319246, 0.251188643151, 0.398107170553, 0.63095734448, 1.0, 2.37137370566, 5.62341325, 15.3992652606, 31.6227766],
    "c": [[0.0, 0.0, -4.93740326815e-05, -0.00136110637329, -0.00506905111755, -0.012741122293, -0.0220144968504, -0.0239939034695, -0.043638641629, -0.117890214262, -0.141123921668, 0.086552487674, 0.179390274565, 0.269368658116, 0.0785040456996, 0.0490248637276, -0.111571911794, -0.0197712184164, -0.0197716870218, -0.0372253617253, 0.0, 0.0], [0.0, 0.0, -0.709484897949, -1.97447407686, -3.15478745349, -2.29603163128, -0.670801534786, -0.704199644986, -4.00987325224, -2.69982990241, 0.0, 4.72814414167, 2.07638470052, 0.0, -0.389846972557, -0.298496119087, 0.0, 0.0, -6.01781536636e-07, 0.0, 0.0, 0.0], [-0.0, -10403.5132381, -1087.77473624, -2193.28637518, -260.711341283, 13.2509852177, 165.970301474, -460.909893146, -1129.39707971, 46.50350675, 1230.97490767, -87.6616265219, 7.90484996078, -62.4281400584, 3.24152775194, -6.32212496608, 2.0221533297, -3.08693235932e-07, -0.00495067060383, 0.00116855980641, -0.0, 0.0], [0.0, 47866151.6427, 285187.385316, 971371.823345, 116156.741398, 17219.1903906, -24161.3612898, 21379.0845631, 79006.3233314, 2016.6788876, -34451.921437, 963.471669433, -292.417702205, 433.842720035, -13.298246809, 19.9358142858, -3.65297127483, 4.34041376596e-08, 0.00101490424907, -7.96902275213e-05, 0.0, 0.0]]}

pbe_s2 = {
    "x": [0.0, 1e-05, 0.0001, 0.001, 0.00215443469, 0.00464158883, 0.01, 0.0146779926762, 0.0215443469003, 0.0316227766017, 0.0464158883361, 0.0681292069058, 0.1, 0.158489319246, 0.251188643151, 0.398107170553, 0.63095734448, 1.0, 2.37137370566, 5.62341325, 15.3992652606, 31.6227766],
    "c": [[0.0, 0.0, -0.000156157535801, -0.0036519900327, -0.0108302033233, -0.0203436953346, -0.0214330355346, 0.000109617244934, 0.00813969827075, -0.0701367130014, -0.162002361715, 0.0337288711362, 0.140348429629, 0.271234417677, 0.078073275124, 0.0436066976238, -0.106097689688, -0.0133141637069, -0.0133143525246, -0.0430994711278, 0.0, 0.0], [0.0, 0.0, -2.17211651544, -4.73638379726, -4.87821808504, -0.433631413905, 0.0, 1.93813387881, 0.0, -6.95060290528, 0.0, 5.02541925806, 2.73498669354, 0.0, -0.448708826169, -0.332102918195, 0.0, 0.0, -2.42488141082e-07, 0.0, 0.0, 0.0], [-0.0, -33701.4964214, -2857.9535128, -3727.23918347, -516.689374427, 48.0322803175, 2538.94893657, -53.5684993409, -1622.23464755, -319.667723139, 1014.01359817, -86.2770702569, 21.2578002151, -62.5949163782, 3.57838438707, -5.43078279308, 2.04380282001, -1.2437692788e-07, -0.0084489217348, 0.00135295689023, -0.0, 0.0], [0.0, 160253203.309, 1061748.57663, 2116943.19531, 377995.031873, -941.771032564, -332306.990184, -8501.76312292, 84497.8829514, 24993.3770676, -27580.3550484, 1053.08484492, -508.788318128, 432.758845019, -14.4367801061, 17.5904487062, -3.69208055752, 1.74842962107e-08, 0.00173203285755, -9.22652326542e-05, 0.0, 0.0]]}

params = {
    "pbe_p1": pbe_p1,
    "pbe0_w1": pbe0_w1,
    "b3lyp_w1": b3lyp_w1,
    "tpss_w1": tpss_w1,
    "pbe_w1": pbe_w1,
    "pbe0_s1": pbe0_s1,
    "pbe0_s2": pbe0_s2,
    "pbe_s1": pbe_s1,
    "pbe_s2": pbe_s2,
}