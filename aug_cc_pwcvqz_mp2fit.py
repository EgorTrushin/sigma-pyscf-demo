#!/usr/bin/env python3
from pyscf import gto

AUXBASIS = {
"H": gto.parse('''
H    S
     21.88353363             1.0000000
H    S
      4.30891017             1.0000000
H    S
      1.18248348             1.0000000
H    S
      0.43311133             1.0000000
H    S
      0.22763540             1.0000000
H    S
      0.99710780880E-01      1.0000000
H    P
      3.22865328             1.0000000
H    P
      1.22856824             1.0000000
H    P
      0.51629315             1.0000000
H    P
      0.22450816             1.0000000
H    P
      0.94444585167E-01      1.0000000
H    D
      2.77818064             1.0000000
H    D
      1.01771006             1.0000000
H    D
      0.52759834             1.0000000
H    D
      0.30650358869          1.0000000
H    F
      1.89643296             1.0000000
H    F
      0.67193094             1.0000000
H    F
      0.44795390000          1.0000000
H    G
      2.13088265             1.0000000
H    G
      0.59562780346          1.0000000
'''),
"B": gto.parse('''
#BASIS SET: (13s,12p,10d,8f,5g,2h) -> [13s,12p,10d,8f,5g,2h]
B    S
    562.5900000              1.0000000
B    S
    157.11491613             1.0000000000
B    S
     75.1804000              1.0000000
B    S
     32.881671170            1.0000000000
B    S
     17.6344000              1.0000000
B    S
      7.6392597400           1.0000000000
B    S
      5.7840700              1.0000000
B    S
      4.9011042100           1.0000000000
B    S
      1.6576127200           1.0000000000
B    S
      0.78359815000          1.0000000000
B    S
      0.32156487000          1.0000000000
B    S
      0.17201663000          1.0000000000
B    S
      0.51789335392E-01      1.0000000000
B    P
    159.9830000              1.0000000
B    P
     83.0330000              1.0000000
B    P
     39.926216980            1.0000000000
B    P
     19.7059000              1.0000000
B    P
     10.022342580            1.0000000000
B    P
      5.7914000              1.0000000
B    P
      3.7955219000           1.0000000000
B    P
      1.6873830300           1.0000000000
B    P
      0.73744111000          1.0000000000
B    P
      0.33659453000          1.0000000000
B    P
      0.15849230000          1.0000000000
B    P
      0.58922032879E-01      1.0000000000
B    D
     50.4872000              1.0000000
B    D
     21.4342000              1.0000000
B    D
     14.831087490            1.0000000000
B    D
      5.9024600              1.0000000
B    D
      4.0532478900           1.0000000000
B    D
      1.3431361900           1.0000000000
B    D
      0.68046983000          1.0000000000
B    D
      0.47013147000          1.0000000000
B    D
      0.24468129000          1.0000000000
B    D
      0.86437718884E-01      1.0000000000
B    F
     15.6076000              1.0000000
B    F
      6.5427600              1.0000000
B    F
      4.2977569300           1.0000000000
B    F
      1.7571088500           1.0000000000
B    F
      1.0907308900           1.0000000000
B    F
      0.47390324000          1.0000000000
B    F
      0.21274953000          1.0000000000
B    F
      0.75619629641E-01      1.0000000000
B    G
      4.1394300              1.0000000
B    G
      1.7049461000           1.0000000000
B    G
      0.87447217000          1.0000000000
B    G
      0.47441651000          1.0000000000
B    G
      0.20431353873          1.0000000000
B    H
      0.96144483000          1.0000000000
B    H
      0.42845246936          1.0000000000
'''),
"C": gto.parse('''
#BASIS SET: (13s,12p,10d,8f,5g,2h) -> [13s,12p,10d,8f,5g,2h]
C    S
    944.9180000              1.0000000
C    S
    221.30970816             1.0000000000
C    S
     70.5017000              1.0000000
C    S
     47.118565820            1.0000000000
C    S
     27.9213000              1.0000000
C    S
      9.1715267600           1.0000000000
C    S
      6.1910700              1.0000000
C    S
      4.1791682300           1.0000000000
C    S
      1.6392093800           1.0000000000
C    S
      0.87237134000          1.0000000000
C    S
      0.48964562000          1.0000000000
C    S
      0.24926145000          1.0000000000
C    S
      0.91582248080E-01      1.0000000000
C    P
    200.1470000              1.0000000
C    P
     93.4590000              1.0000000
C    P
     49.862264980            1.0000000000
C    P
     27.8690000              1.0000000
C    P
     12.325982010            1.0000000000
C    P
      6.5454900              1.0000000
C    P
      3.5906423100           1.0000000000
C    P
      1.6884440200           1.0000000000
C    P
      1.0533818300           1.0000000000
C    P
      0.51716817000          1.0000000000
C    P
      0.24545173000          1.0000000000
C    P
      0.10457210217          1.0000000000
C    D
     54.7128000              1.0000000
C    D
     25.2257000              1.0000000
C    D
     16.935433640            1.0000000000
C    D
      5.6402869100           1.0000000000
C    D
      3.2922300              1.0000000
C    D
      2.2042152900           1.0000000000
C    D
      1.0310124000           1.0000000000
C    D
      0.59945554000          1.0000000000
C    D
      0.35214502000          1.0000000000
C    D
      0.14052014554          1.0000000000
C    F
     21.9219000              1.0000000
C    F
     10.2631000              1.0000000
C    F
      5.4484111500           1.0000000000
C    F
      2.2630139600           1.0000000000
C    F
      1.7389784400           1.0000000000
C    F
      0.72979487000          1.0000000000
C    F
      0.34863256000          1.0000000000
C    F
      0.15883684421          1.0000000000
C    G
      7.3920500              1.0000000
C    G
      2.8127761500           1.0000000000
C    G
      1.3495575800           1.0000000000
C    G
      0.71325674000          1.0000000000
C    G
      0.32435913973          1.0000000000
C    H
      1.4869361100           1.0000000000
C    H
      0.65934829301          1.0000000000
'''),
"N": gto.parse('''
N    S
    927.7870000              1.0000000
N    S
    284.72976833             1.0000000000
N    S
    101.8900000              1.0000000
N    S
     61.812130160            1.0000000000
N    S
     40.7825000              1.0000000
N    S
     14.133378780            1.0000000000
N    S
      7.5049700              1.0000000
N    S
      5.1587107600           1.0000000000
N    S
      2.0549150500           1.0000000000
N    S
      1.1370573000           1.0000000000
N    S
      0.59481528000          1.0000000000
N    S
      0.35860680000          1.0000000000
N    S
      0.89848936374E-01      1.0000000000
N    P
    280.7020000              1.0000000
N    P
    129.6310000              1.0000000
N    P
     66.749955880            1.0000000000
N    P
     36.2049000              1.0000000
N    P
     16.924786440            1.0000000000
N    P
      9.0846600              1.0000000
N    P
      5.0303492800           1.0000000000
N    P
      2.8212238000           1.0000000000
N    P
      1.7003362000           1.0000000000
N    P
      0.69507352000          1.0000000000
N    P
      0.34953700000          1.0000000000
N    P
      0.11449794090          1.0000000000
N    D
     98.4186000              1.0000000
N    D
     39.5811000              1.0000000
N    D
     25.136207520            1.0000000000
N    D
      9.1488300              1.0000000
N    D
      8.3552240100           1.0000000000
N    D
      3.3625287000           1.0000000000
N    D
      1.4989770800           1.0000000000
N    D
      0.74093179000          1.0000000000
N    D
      0.46948829000          1.0000000000
N    D
      0.15762013479          1.0000000000
N    F
     31.3497000              1.0000000
N    F
     14.6463000              1.0000000
N    F
      8.4797650900           1.0000000000
N    F
      3.6173570000           1.0000000000
N    F
      2.4953599400           1.0000000000
N    F
      1.0331339900           1.0000000000
N    F
      0.48153351000          1.0000000000
N    F
      0.16353389200          1.0000000000
N    G
     10.7654000              1.0000000
N    G
      4.1850847000           1.0000000000
N    G
      1.9255348800           1.0000000000
N    G
      0.97077363000          1.0000000000
N    G
      0.42656181902          1.0000000000
N    H
      2.1281915200           1.0000000000
N    H
      0.78532726327          1.0000000000
'''),
"O": gto.parse('''
#BASIS SET: (13s,12p,10d,8f,5g,2h) -> [13s,12p,10d,8f,5g,2h]
O    S
   1360.9500000              1.0000000
O    S
    353.83035678             1.0000000000
O    S
    149.9520000              1.0000000
O    S
     75.357261710            1.0000000000
O    S
     44.3906000              1.0000000
O    S
     21.117406900            1.0000000000
O    S
      7.9723700              1.0000000
O    S
      5.2252556200           1.0000000000
O    S
      2.5880879700           1.0000000000
O    S
      1.5253313500           1.0000000000
O    S
      0.71853188000          1.0000000000
O    S
      0.31850176000          1.0000000000
O    S
      0.95281907341E-01      1.0000000000
O    P
    351.4270000              1.0000000
O    P
    147.0530000              1.0000000
O    P
     88.557170360            1.0000000000
O    P
     45.7078000              1.0000000
O    P
     22.294394600            1.0000000000
O    P
     12.0992000              1.0000000
O    P
      6.5663064600           1.0000000000
O    P
      3.3419619800           1.0000000000
O    P
      1.8538951800           1.0000000000
O    P
      0.86798989000          1.0000000000
O    P
      0.47805185000          1.0000000000
O    P
      0.18989077313          1.0000000000
O    D
     80.6501000              1.0000000
O    D
     35.0853000              1.0000000
O    D
     25.394489460            1.0000000000
O    D
     14.8091000              1.0000000
O    D
      8.6360887100           1.0000000000
O    D
      3.9106678000           1.0000000000
O    D
      2.1402819600           1.0000000000
O    D
      0.98946206000          1.0000000000
O    D
      0.42309083000          1.0000000000
O    D
      0.13954687713          1.0000000000
O    F
     40.5353000              1.0000000
O    F
     18.4997000              1.0000000
O    F
     10.502388090            1.0000000000
O    F
      4.4495593000           1.0000000000
O    F
      2.9580200500           1.0000000000
O    F
      1.3208390600           1.0000000000
O    F
      0.73872352000          1.0000000000
O    F
      0.28801462388          1.0000000000
O    G
     15.2068000              1.0000000
O    G
      4.9476228900           1.0000000000
O    G
      2.5401950000           1.0000000000
O    G
      1.2595085000           1.0000000000
O    G
      0.53719929557          1.0000000000
O    H
      2.7655405000           1.0000000000
O    H
      1.0512431837           1.0000000000
'''),
"F": gto.parse('''
#BASIS SET: (13s,12p,10d,8f,5g,2h) -> [13s,12p,10d,8f,5g,2h]
F    S
   2403.0400000              1.0000000
F    S
    431.67695820             1.0000000000
F    S
    199.3190000              1.0000000
F    S
     93.461929760            1.0000000000
F    S
     59.5466000              1.0000000
F    S
     27.620303050            1.0000000000
F    S
     10.5292000              1.0000000
F    S
      6.3566143800           1.0000000000
F    S
      3.5892924800           1.0000000000
F    S
      1.9037103100           1.0000000000
F    S
      0.92836019000          1.0000000000
F    S
      0.38830933000          1.0000000000
F    S
      0.12498303738          1.0000000000
F    P
    405.7360000              1.0000000
F    P
    172.9200000              1.0000000
F    P
     81.008225940            1.0000000000
F    P
     62.6788000              1.0000000
F    P
     23.473144240            1.0000000000
F    P
     13.8606000              1.0000000
F    P
      6.7911063800           1.0000000000
F    P
      4.3827062600           1.0000000000
F    P
      2.2145213700           1.0000000000
F    P
      1.0404852700           1.0000000000
F    P
      0.59915550000          1.0000000000
F    P
      0.25173678581          1.0000000000
F    D
    107.6660000              1.0000000
F    D
     56.8439000              1.0000000
F    D
     22.819797250            1.0000000000
F    D
     16.4713000              1.0000000
F    D
      7.2976092000           1.0000000000
F    D
      4.5135499800           1.0000000000
F    D
      2.4059320400           1.0000000000
F    D
      1.2526711700           1.0000000000
F    D
      0.49245361000          1.0000000000
F    D
      0.18946697252          1.0000000000
F    F
     52.8086000              1.0000000
F    F
     23.8725000              1.0000000
F    F
     13.812186020            1.0000000000
F    F
      5.9724044900           1.0000000000
F    F
      4.0340302300           1.0000000000
F    F
      1.8225724300           1.0000000000
F    F
      1.0194384600           1.0000000000
F    F
      0.40090866618          1.0000000000
F    G
     20.4005000              1.0000000
F    G
      6.4904183700           1.0000000000
F    G
      3.3578833600           1.0000000000
F    G
      1.7436062500           1.0000000000
F    G
      0.76588330370          1.0000000000
F    H
      3.5694932400           1.0000000000
F    H
      1.3684048531           1.0000000000
'''),
"Al": gto.parse('''
Al    S
    692.8470000              1.0000000
Al    S
    303.58760145             1.0000000000
Al    S
    168.1190000              1.0000000
Al    S
    105.81456731             1.0000000000
Al    S
     58.0437000              1.0000000
Al    S
     40.419439839            1.0000000000
Al    S
     22.7118000              1.0000000
Al    S
     12.761800201            1.0000000000
Al    S
      5.5489787481           1.0000000000
Al    S
      3.2330855231           1.0000000000
Al    S
      1.0654955345           1.0000000000
Al    S
      0.47598894597          1.0000000000
Al    S
      0.29170881381          1.0000000000
Al    S
      0.19442125941          1.0000000000
Al    S
      0.91560708801E-01      1.0000000000
Al    S
      0.50402562621E-01      1.0000000000
Al    P
    401.3940000              1.0000000
Al    P
    162.86385663             1.0000000000
Al    P
     54.816189488            1.0000000000
Al    P
     30.0031000              1.0000000
Al    P
     14.9635000              1.0000000
Al    P
      8.9642578065           1.0000000000
Al    P
      6.0063219144           1.0000000000
Al    P
      3.5032800              1.0000000
Al    P
      2.0433407907           1.0000000000
Al    P
      0.93022135217          1.0000000000
Al    P
      0.44281831986          1.0000000000
Al    P
      0.22043212270          1.0000000000
Al    P
      0.11156272819          1.0000000000
Al    P
      0.57260901927E-01      1.0000000000
Al    D
     99.0865000              1.0000000
Al    D
     47.076946028            1.0000000000
Al    D
     25.1171000              1.0000000
Al    D
     13.634175691            1.0000000000
Al    D
      8.0000000              1.0000000
Al    D
      4.5653210637           1.0000000000
Al    D
      2.5914772201           1.0000000000
Al    D
      1.3131555622           1.0000000000
Al    D
      0.57916888089          1.0000000000
Al    D
      0.29959972951          1.0000000000
Al    D
      0.14992270614          1.0000000000
Al    D
      0.67857277719E-01      1.0000000000
Al    F
     20.6507000              1.0000000
Al    F
      7.3610672305           1.0000000000
Al    F
      4.9105400              1.0000000
Al    F
      3.2758038625           1.0000000000
Al    F
      2.0617300              1.0000000
Al    F
      1.2976173270           1.0000000000
Al    F
      0.52986045044          1.0000000000
Al    F
      0.28997998755          1.0000000000
Al    F
      0.16935910681          1.0000000000
Al    F
      0.72278385730E-01      1.0000000000
Al    G
     11.5400000              1.0000000
Al    G
      3.8169900              1.0000000
Al    G
      1.7549118072           1.0000000000
Al    G
      1.0910300              1.0000000
Al    G
      0.51785875173          1.0000000000
Al    G
      0.24524831749          1.0000000000
Al    G
      0.95874073730E-01      1.0000000000
Al    H
      3.8754900              1.0000000
Al    H
      1.7962300              1.0000000
Al    H
      0.50872988782          1.0000000000
Al    H
      0.21239023993          1.0000000000
'''),
"Si": gto.parse('''
#BASIS SET: (16s,14p,12d,10f,7g,4h) -> [16s,14p,12d,10f,7g,4h]
Si    S
    906.0290000              1.0000000
Si    S
    256.83087219             1.0000000000
Si    S
    163.2060000              1.0000000
Si    S
     87.861910726            1.0000000000
Si    S
     50.4526000              1.0000000
Si    S
     28.960868256            1.0000000000
Si    S
     15.6417000              1.0000000
Si    S
      8.4480674825           1.0000000000
Si    S
      4.0922528981           1.0000000000
Si    S
      2.5762997817           1.0000000000
Si    S
      1.1718308263           1.0000000000
Si    S
      0.52104495957          1.0000000000
Si    S
      0.34473520623          1.0000000000
Si    S
      0.19193698213          1.0000000000
Si    S
      0.64564888423E-01      1.0000000000
Si    S
      0.16677550295E-01      1.0000000000
Si    P
    454.2540000              1.0000000
Si    P
    155.15721864             1.0000000000
Si    P
     59.208838431            1.0000000000
Si    P
     36.1693000              1.0000000
Si    P
     18.0463000              1.0000000
Si    P
     10.061048425            1.0000000000
Si    P
      6.4069645477           1.0000000000
Si    P
      4.1461900              1.0000000
Si    P
      2.6831546491           1.0000000000
Si    P
      1.1982271028           1.0000000000
Si    P
      0.59486169772          1.0000000000
Si    P
      0.29543772161          1.0000000000
Si    P
      0.16702303132          1.0000000000
Si    P
      0.59070317986E-01      1.0000000000
Si    D
    122.8710000              1.0000000
Si    D
     36.633145616            1.0000000000
Si    D
     21.2466000              1.0000000
Si    D
     12.322663421            1.0000000000
Si    D
      5.1597306967           1.0000000000
Si    D
      2.3179344789           1.0000000000
Si    D
      1.4585400              1.0000000
Si    D
      0.91778122169          1.0000000000
Si    D
      0.55722960454          1.0000000000
Si    D
      0.30259020135          1.0000000000
Si    D
      0.17369977325          1.0000000000
Si    D
      0.73689775491E-01      1.0000000000
Si    F
     26.6987000              1.0000000
Si    F
      8.9160731813           1.0000000000
Si    F
      5.9436300              1.0000000
Si    F
      3.9621389114           1.0000000000
Si    F
      1.8790600476           1.0000000000
Si    F
      1.1526800              1.0000000
Si    F
      0.70709129617          1.0000000000
Si    F
      0.40315442038          1.0000000000
Si    F
      0.21501555354          1.0000000000
Si    F
      0.81438465235E-01      1.0000000000
Si    G
     14.4380000              1.0000000
Si    G
      4.7851200              1.0000000
Si    G
      2.3109706984           1.0000000000
Si    G
      1.3390800              1.0000000
Si    G
      0.70626472074          1.0000000000
Si    G
      0.35792288122          1.0000000000
Si    G
      0.15257444636          1.0000000000
Si    H
      4.7385600              1.0000000
Si    H
      2.2121100              1.0000000
Si    H
      0.66825252207          1.0000000000
Si    H
      0.32191215847          1.0000000000
'''),
"P": gto.parse('''
#BASIS SET: (16s,14p,12d,10f,7g,4h) -> [16s,14p,12d,10f,7g,4h]
P    S
   1059.6000000              1.0000000
P    S
    353.20068154             1.0000000000
P    S
    204.7470000              1.0000000
P    S
    112.77544132             1.0000000000
P    S
     52.5158000              1.0000000
P    S
     37.728319048            1.0000000000
P    S
     20.4263000              1.0000000
P    S
     11.058986610            1.0000000000
P    S
      6.4565998609           1.0000000000
P    S
      4.3043992402           1.0000000000
P    S
      2.3471981685           1.0000000000
P    S
      0.67827314440          1.0000000000
P    S
      0.44084149158          1.0000000000
P    S
      0.29389433280          1.0000000000
P    S
      0.15934159216          1.0000000000
P    S
      0.42294735379E-01      1.0000000000
P    P
    371.8510000              1.0000000
P    P
    106.90913122             1.0000000000
P    P
     48.777858489            1.0000000000
P    P
     31.2387000              1.0000000
P    P
     18.0000000              1.0000000
P    P
     12.849534763            1.0000000000
P    P
      9.1006700              1.0000000
P    P
      6.4455431274           1.0000000000
P    P
      3.3387274725           1.0000000000
P    P
      1.6194853377           1.0000000000
P    P
      0.70637210538          1.0000000000
P    P
      0.39581598161          1.0000000000
P    P
      0.21865173454          1.0000000000
P    P
      0.76865401233E-01      1.0000000000
P    D
    106.6520000              1.0000000
P    D
     39.396766859            1.0000000000
P    D
     19.5339000              1.0000000
P    D
     12.501867481            1.0000000000
P    D
      6.1522798130           1.0000000000
P    D
      2.6952873395           1.0000000000
P    D
      1.8000000              1.0000000
P    D
      1.2021068578           1.0000000000
P    D
      0.62782760882          1.0000000000
P    D
      0.37399375168          1.0000000000
P    D
      0.20042557937          1.0000000000
P    D
      0.73662273208E-01      1.0000000000
P    F
     34.2326000              1.0000000
P    F
     10.645240710            1.0000000000
P    F
      7.2000000              1.0000000
P    F
      4.8706854590           1.0000000000
P    F
      3.3229900              1.0000000
P    F
      2.2670917997           1.0000000000
P    F
      0.95282944287          1.0000000000
P    F
      0.51941752050          1.0000000000
P    F
      0.27674131065          1.0000000000
P    F
      0.10463534348          1.0000000000
P    G
     17.7952000              1.0000000
P    G
      5.9865100              1.0000000
P    G
      2.9314734121           1.0000000000
P    G
      1.7597500              1.0000000
P    G
      0.93612719454          1.0000000000
P    G
      0.47911263123          1.0000000000
P    G
      0.19049975552          1.0000000000
P    H
      5.8083600              1.0000000
P    H
      2.6371300              1.0000000
P    H
      0.87573835694          1.0000000000
P    H
      0.37232681641          1.0000000000
'''),
"S": gto.parse('''
#BASIS SET: (16s,14p,12d,10f,7g,4h) -> [16s,14p,12d,10f,7g,4h]
S    S
   1200.3500000              1.0000000
S    S
    400.11571327             1.0000000000
S    S
    216.8300000              1.0000000
S    S
    123.17295244             1.0000000000
S    S
     67.0796000              1.0000000
S    S
     38.356308104            1.0000000000
S    S
     21.8972000              1.0000000
S    S
     12.500841994            1.0000000000
S    S
      6.9609096853           1.0000000000
S    S
      4.1130474075           1.0000000000
S    S
      2.2219991533           1.0000000000
S    S
      0.97055633892          1.0000000000
S    S
      0.56340003868          1.0000000000
S    S
      0.37518396762          1.0000000000
S    S
      0.15064894488          1.0000000000
S    S
      0.67460366397E-01      1.0000000000
S    P
    414.1360000              1.0000000
S    P
    180.41447200             1.0000000000
S    P
     88.518564917            1.0000000000
S    P
     39.5416000              1.0000000
S    P
     22.0000000              1.0000000
S    P
     13.432333757            1.0000000000
S    P
      8.7220800              1.0000000
S    P
      5.6635507463           1.0000000000
S    P
      2.9325035426           1.0000000000
S    P
      1.5581785596           1.0000000000
S    P
      0.88619682119          1.0000000000
S    P
      0.49461777704          1.0000000000
S    P
      0.26727533119          1.0000000000
S    P
      0.10311650574          1.0000000000
S    D
    120.2110000              1.0000000
S    D
     41.816785927            1.0000000000
S    D
     22.6502000              1.0000000
S    D
     16.090062515            1.0000000000
S    D
      7.4729381128           1.0000000000
S    D
      3.5878749040           1.0000000000
S    D
      2.4230100              1.0000000
S    D
      1.6363363949           1.0000000000
S    D
      0.94416078632          1.0000000000
S    D
      0.45026614895          1.0000000000
S    D
      0.23213678108          1.0000000000
S    D
      0.94135084627E-01      1.0000000000
S    F
     40.2243000              1.0000000
S    F
     12.276131605            1.0000000000
S    F
      7.8538500              1.0000000
S    F
      5.0246237407           1.0000000000
S    F
      3.4313500              1.0000000
S    F
      2.3432991679           1.0000000000
S    F
      1.2647947272           1.0000000000
S    F
      0.73122191804          1.0000000000
S    F
      0.36086017232          1.0000000000
S    F
      0.14009368577          1.0000000000
S    G
     21.4007000              1.0000000
S    G
      7.2042100              1.0000000
S    G
      3.4634661735           1.0000000000
S    G
      2.0273700              1.0000000
S    G
      1.1867413008           1.0000000000
S    G
      0.57655948880          1.0000000000
S    G
      0.24870171738          1.0000000000
S    H
      6.8985400              1.0000000
S    H
      3.1529900              1.0000000
S    H
      1.0179355003           1.0000000000
S    H
      0.44918798276          1.0000000000
'''),
"Cl": gto.parse('''
#BASIS SET: (16s,14p,12d,10f,7g,4h) -> [16s,14p,12d,10f,7g,4h]
Cl    S
   1321.7700000              1.0000000
Cl    S
    440.58858250             1.0000000000
Cl    S
    237.6620000              1.0000000
Cl    S
    138.90122145             1.0000000000
Cl    S
     78.1835000              1.0000000
Cl    S
     44.007194578            1.0000000000
Cl    S
     24.1958000              1.0000000
Cl    S
     13.303183011            1.0000000000
Cl    S
      7.4772614324           1.0000000000
Cl    S
      4.0701375516           1.0000000000
Cl    S
      1.9453625871           1.0000000000
Cl    S
      1.0787159912           1.0000000000
Cl    S
      0.71219619422          1.0000000000
Cl    S
      0.33242674816          1.0000000000
Cl    S
      0.19533039522          1.0000000000
Cl    S
      0.81858283108E-01      1.0000000000
Cl    P
    504.9080000              1.0000000
Cl    P
    205.03402167             1.0000000000
Cl    P
     94.734596899            1.0000000000
Cl    P
     46.3595000              1.0000000
Cl    P
     25.0000000              1.0000000
Cl    P
     15.606217957            1.0000000000
Cl    P
     10.1592000              1.0000000
Cl    P
      6.6132857053           1.0000000000
Cl    P
      3.1369184059           1.0000000000
Cl    P
      1.7482117297           1.0000000000
Cl    P
      1.0304024543           1.0000000000
Cl    P
      0.56281342100          1.0000000000
Cl    P
      0.32789396039          1.0000000000
Cl    P
      0.13405622026          1.0000000000
Cl    D
    225.1760000              1.0000000
Cl    D
     62.987707930            1.0000000000
Cl    D
     35.1896000              1.0000000
Cl    D
     19.659559624            1.0000000000
Cl    D
      8.4292880072           1.0000000000
Cl    D
      4.2698173981           1.0000000000
Cl    D
      2.8311800              1.0000000
Cl    D
      1.8772608920           1.0000000000
Cl    D
      1.1195334342           1.0000000000
Cl    D
      0.54269498500          1.0000000000
Cl    D
      0.26783494586          1.0000000000
Cl    D
      0.10462641692          1.0000000000
Cl    F
     48.1427000              1.0000000
Cl    F
     15.169300936            1.0000000000
Cl    F
      9.5118200              1.0000000
Cl    F
      5.9643244460           1.0000000000
Cl    F
      4.0774100              1.0000000
Cl    F
      2.7874498788           1.0000000000
Cl    F
      1.6887788806           1.0000000000
Cl    F
      0.94776135270          1.0000000000
Cl    F
      0.43653876396          1.0000000000
Cl    F
      0.18900391211          1.0000000000
Cl    G
     25.5945000              1.0000000
Cl    G
      8.8047800              1.0000000
Cl    G
      4.2424168679           1.0000000000
Cl    G
      2.5731200              1.0000000
Cl    G
      1.5009282424           1.0000000000
Cl    G
      0.72255390768          1.0000000000
Cl    G
      0.36371026234          1.0000000000
Cl    H
      8.2536600              1.0000000
Cl    H
      3.7528700              1.0000000
Cl    H
      1.2422554159           1.0000000000
Cl    H
      0.57063298029          1.0000000000
''')
}