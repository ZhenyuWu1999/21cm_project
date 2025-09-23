#This file contains the halo profile data extracted from Latif+ papers using PlotDigitizer.

#PlotDigitizer, 3.3.9, 2025, https://plotdigitizer.com
# @misc{
# url = {https://plotdigitizer.com},
# title = {PlotDigitizer: Version 3.3.9},
# year = {2025}
# }

import numpy as np
from collections import defaultdict
from physical_constants import Omega_b, Omega_m
import matplotlib.pyplot as plt
import os
import copy

PC_TO_CM = 3.085677581e18
MSUN_G   = 1.98847e33

# ---- basic container：halo -> quantity -> j21 -> {"r":..., "y":..., "units":...}
Latif2019_data = defaultdict(lambda: defaultdict(dict))


def add_profile(halo, j21, quantity, r, y, r_unit="pc", y_unit=None):
    Latif2019_data[halo][quantity][float(j21)] = {
        "r": np.asarray(r, float),
        "y": np.asarray(y, float),
        "units": {"r": r_unit, "y": y_unit},
    }

def set_scalar(halo, j21, name, value):
    Latif2019_data[halo][float(j21)][name] = float(value)

#Latif & Sadegh 2019 https://doi.org/10.1093/mnras/stz2812

#1. halo 6
#1.0 collapse properties
set_scalar("halo6", 0.1, "z_collapse", 25.4)
set_scalar("halo6", 0.1, "M_collapse_Msun", 7.6e5)
set_scalar("halo6", 1.0, "z_collapse", 24.5)
set_scalar("halo6", 1.0, "M_collapse_Msun", 1.2e6)
set_scalar("halo6", 10.0, "z_collapse", 11.5)
set_scalar("halo6", 10.0, "M_collapse_Msun", 5.7e7)
set_scalar("halo6", 100.0, "z_collapse", 11.0)
set_scalar("halo6", 100.0, "M_collapse_Msun", 6.3e7)
set_scalar("halo6", 1000.0, "z_collapse", 10.6)
set_scalar("halo6", 1000.0, "M_collapse_Msun", 6.7e7)

#1.1 M_enc profile
#(Menc, r) the same for J21=0.1 and 1.0
r01 = [0.00035, 0.00069, 0.00404, 0.0121, 0.10329, 1.0899, 7.85423, 45.46763, 379.38141, 1051.29983]
Menc01 = [0.01992, 0.30115, 16.70967, 69.69781, 320.09786, 2598.68239, 11766.59897, 34768.52715, 329098.2253, 2426517.68699]
add_profile("halo6", 0.1, "M_enc",  r01, Menc01, r_unit="pc", y_unit="Msun")


add_profile("halo6", 1.0, "M_enc",  r01, Menc01, r_unit="pc", y_unit="Msun")

r10 = [0.00056, 0.00164, 0.00452, 0.01914, 0.10766, 0.68431, 3.62174, 22.02055, 91.32361, 407.53673, 1484.40518]
Menc10 = [0.02091, 0.28719, 4.69984, 91.40065, 380.73462, 1755.5782, 9404.70814, 97885.65112, 725264.60559, 2583163.90617, 5338062.6527]
add_profile("halo6", 10.0, "M_enc",  r10, Menc10, r_unit="pc", y_unit="Msun")

r100 = [0.00079, 0.0023, 0.00635, 0.02173, 0.08802, 0.48188, 1.62388, 8.6568, 48.09039, 247.64147, 1305.93423]
Menc100 = [0.04954, 0.86036, 9.60501, 118.22416, 697.33893, 2384.36329, 9064.68081, 36120.78278, 686441.63339, 2760925.10654, 5135942.1426]
add_profile("halo6", 100.0, "M_enc",  r100, Menc100, r_unit="pc", y_unit="Msun")

r1e3 = [0.00045, 0.00102, 0.00299, 0.0093, 0.03864, 0.13931, 1.0459, 9.20296, 82.05128, 493.5039, 1848.25547]
Menc1e3 = [0.05776, 0.57917, 12.05993, 192.25975, 2386.47998, 14985.24882, 78619.98072, 361394.84373, 1494099.07065, 3944326.84803, 7615946.47632]
add_profile("halo6", 1000.0, "M_enc",  r1e3, Menc1e3, r_unit="pc", y_unit="Msun")


#1.2 density profile
r01 = [0.00029, 0.00078, 0.00258, 0.00619, 0.01679, 0.04151, 0.1117, 0.42741, 0.78844, 1.32157, 2.0675, 4.10838, 7.80126, 24.54502, 83.34281, 210.55258, 593.11402, 1172.9761]
rho01 = 1.0e-24 * np.array([6676036375.4725, 5567284274.42254, 2941752221.67314, 1004635218.58635, 84964879.48948, 8549438.59941, 953025.53732, 129167.03329, 34624.77271, 22766.28197, 3072.93953, 508.66154, 64.72145, 5.50033, 1.15906, 0.21366, 0.04147, 0.02405])
add_profile("halo6", 0.1, "density", r01, rho01, r_unit="pc", y_unit="g/cm^3")

r1 = [0.00029, 0.00078, 0.00258, 0.00718, 0.01919, 0.03995, 0.11464, 0.23425, 0.50577, 1.00734, 2.0675, 4.9583, 10.12032, 26.09247, 83.34281, 193.87067, 593.11402, 1172.9761]
rho1 = 1.0e-24*np.array([6676036375.4725, 5567284274.42254, 2941752221.67314, 1095622150.09972, 198226571.17579, 19775489.35551, 2148304.48494, 557040.58712, 35383.43833, 6297.38324, 3072.93953, 512.09462, 50.80204, 8.59027, 1.15906, 0.19954, 0.04147, 0.02405])
add_profile("halo6", 1.0, "density", r1, rho1, r_unit="pc", y_unit="g/cm^3")

r10 = [0.00051, 0.00143, 0.0031, 0.00847, 0.02277, 0.04529, 0.11058, 0.32619, 0.76996, 1.7908, 2.82628, 5.73591, 9.27555, 17.04728, 44.64756, 91.84813, 475.44081, 1579.44347, 25.41432, 1004.94017]
rho10 = 1.0e-24*np.array([1095212795.11404, 1091127647.88932, 1008396540.90551, 341556983.36877, 90031637.12634, 14180280.55537, 1613533.47786, 200077.82296, 38200.62571, 34289.98263, 8164.97848, 13296.88224, 8713.53436, 1155.65551, 147.2641, 5.27685, 0.20822, 0.16344, 138.61337, 0.0823])
add_profile("halo6", 10.0, "density", r10, rho10, r_unit="pc", y_unit="g/cm^3")

r100 = [0.00051, 0.00145, 0.00636, 0.03108, 0.8432, 0.11837, 0.27728, 1.5136, 2.82628, 5.29368, 9.55243, 112.70825, 475.44081, 1579.44347, 39.68564, 889.54949]
rho100 = 1.0e-24*np.array([1097717526.19304, 1096896621.25293, 656821109.84086, 142029271.05545, 63317.97653, 2467373.53578, 834868.35431, 53328.89635, 8091.95476, 2768.91762, 422.03676, 6.66972, 0.20429, 0.16031, 54.93343, 0.06053])
add_profile("halo6", 100.0, "density", r100, rho100, r_unit="pc", y_unit="g/cm^3")

r1e3 = [0.0006, 0.00201, 0.01221, 0.05365, 1.83971, 0.1469, 0.37665, 3.38012, 9.1015, 112.70825, 361.61643, 1360.91106, 39.68564, 820.71265, 1816.7673]
rho1e3 = 1.0e-24*np.array([6842183804.26088, 5095471065.99368, 1926704050.31779, 259565424.94541, 76140.32924, 129931609.87845, 3785100.43162, 45151.52889, 1950.26949, 6.66972, 0.25964, 0.13855, 54.93343, 0.06698, 0.06272])
add_profile("halo6", 1000.0, "density", r1e3, rho1e3, r_unit="pc", y_unit="g/cm^3")

#1.3 fH2 profile
r01 = [0.00029, 0.00076, 0.00154, 0.0036, 0.01057, 0.03824, 0.22964, 1.23441, 4.65921, 11.91119, 21.60988, 26.65703, 29.67299, 45.83186]
fH2_01 = [0.00536, 0.00565, 0.00459, 0.00312, 0.00196, 0.00133, 0.00104, 0.00087, 0.00055, 0.0002, 0.00007, 0.00003, 0.00002, 0.00001]
add_profile("halo6", 0.1, "fH2", r01, fH2_01, r_unit="pc", y_unit=None)

r1 = [0.00042, 0.00083, 0.00168, 0.0036, 0.01057, 0.03824, 0.22964, 0.99363, 3.36506, 6.06289, 8.21556, 10.29322, 13.72205, 15.50683]
fH2_1 = [0.00444, 0.00389, 0.00341, 0.00312, 0.00196, 0.00133, 0.00104, 0.00062, 0.00047, 0.00031, 0.00012, 0.00006, 0.00002, 0.00001]
add_profile("halo6", 1.0, "fH2", r1, fH2_1, r_unit="pc", y_unit=None)

r10 = [0.0005, 0.00193, 0.00542, 0.01464, 0.05303, 0.21793, 1.07813, 2.41321, 5.4761, 9.3343, 20.219, 42.07087, 65.86782, 79.97309]
fH2_10 = [0.00244, 0.0025, 0.00216, 0.00184, 0.00138, 0.00102, 0.0009, 0.00075, 0.00071, 0.0006, 0.0002, 0.00007, 0.00005, 0.00001]
add_profile("halo6", 10.0, "fH2", r10, fH2_10, r_unit="pc", y_unit=None)

r100 = [0.00091, 0.0025, 0.00578, 0.01524, 0.04354, 0.13427, 0.32564, 0.79233, 1.31568, 2.29682, 4.54791, 5.92353, 7.22422, 8.54987]
fH2_100 = [0.00196, 0.00197, 0.0018, 0.00134, 0.00085, 0.00052, 0.0005, 0.00039, 0.0005, 0.00031, 0.00015, 0.00008, 0.00004, 0.00001]
add_profile("halo6", 100.0, "fH2", r100, fH2_100, r_unit="pc", y_unit=None)

r1e3 = [2.14766, 2.2645, 2.88788, 3.95687, 4.9461, 5.10946]
fH2_1e3 = [0.00001, 0.00002, 0.00006, 0.00004, 0.00003, 0.00001]
add_profile("halo6", 1000.0, "fH2", r1e3, fH2_1e3, r_unit="pc", y_unit=None)

#2. halo 1
set_scalar("halo1", 0.1, "z_collapse", 24.5)
set_scalar("halo1", 0.1, "M_collapse_Msun", 8.8e5)
set_scalar("halo1", 1.0, "z_collapse", 23.9)
set_scalar("halo1", 1.0, "M_collapse_Msun", 8.0e5)
set_scalar("halo1", 5, "z_collapse", 15.4)
set_scalar("halo1", 5, "M_collapse_Msun", 7.7e6)
set_scalar("halo1", 10, "z_collapse", 15)
set_scalar("halo1", 10, "M_collapse_Msun", 9.3e6)
set_scalar("halo1", 50, "z_collapse", 12.6)
set_scalar("halo1", 50, "M_collapse_Msun", 1.5e7)
set_scalar("halo1", 100, "z_collapse", 11.7)
set_scalar("halo1", 100, "M_collapse_Msun", 1.8e7)
set_scalar("halo1", 1000, "z_collapse", 11.4)
set_scalar("halo1", 1000, "M_collapse_Msun", 1.9e7)

#2.1 M_enc profile

#2.2 density profile
r0 = [0.00065, 0.00132, 0.00344, 0.01056, 0.02454, 0.05032, 0.15356, 0.65231, 2.17313, 7.86232, 17.92803, 111.31718, 648.16059, 60.8456, 305.52455, 406.6636, 525.52314, 846.30535]
rho0 = 1.0e-24*np.array([3518135780.27434, 3358307287.24769, 1817815986.08733, 596974873.72292, 79816641.99075, 14468155.76692, 1185138.9588, 53815.26286, 6764.4595, 247.67061, 29.91538, 1.19222, 0.17514, 3.64173, 0.52895, 14.93481, 0.87424, 0.0599])
add_profile("halo1", 0.0, "density", r0, rho0, r_unit="pc", y_unit="g/cm^3")

r01 = [0.00034, 0.00127, 0.0032, 0.00674, 0.01694, 0.04054, 0.09879, 0.25853, 0.70681, 3.10899, 7.07326, 22.68678, 92.19933, 423.48338, 701.52518]
rho01 = 1.0e-24 * np.array([2876910403.77372, 2289087177.8256, 2282835188.42443, 1118060843.29187, 181950357.49772, 23809912.53676, 3168538.69654, 307388.37966, 26315.86303, 2327.18763, 163.30271, 12.06573, 1.74364, 0.97454, 0.1303])
add_profile("halo1", 0.1, "density", r01, rho01, r_unit="pc", y_unit="g/cm^3")

r1 = [0.00033, 0.00086, 0.00318, 0.00674, 0.01757, 0.03593, 0.09879, 0.25853, 0.70681, 3.10899, 7.07326, 22.68678, 92.19933, 423.48338, 701.52518]
rho1 =1.0e-24 * np.array([5404968643.62274, 3512641828.02525, 2677348085.5781, 1118060843.29187, 168536988.63247, 16666257.24442, 3168538.69654, 307388.37966, 26315.86303, 2327.18763, 163.30271, 12.06573, 1.74364, 0.97454, 0.1303])
add_profile("halo1", 1.0, "density", r1, rho1, r_unit="pc", y_unit="g/cm^3")

r5 = [0.00043246198102412666, 0.0016421900100564637, 0.006356727399716979, 0.02545564750584017, 0.09675351377944945, 0.3756625487879706, 0.5645354074640656, 0.7226755207462523, 1.1996168969384535, 2.4348336185338937, 5.321333228762902, 19.813218792892823, 112.35101706662647, 313.51895013443016, 808.7089589899316]
rho5 = 1.0e-24*np.array([1519793368.1251152, 1570724581.363684, 929572849.2847463, 98813668.47427365, 3606434.4225863963, 235015.6817276271, 3780627.572245374, 14505928.53943961, 230911.69653707076, 1599.5518666220332, 377.63230908395997, 14.284996980955503, 6.9094072447067205, 0.6679662135727504, 0.1653153922768664])
add_profile("halo1", 5.0, "density", r5, rho5, r_unit="pc", y_unit="g/cm^3")

r10 = [0.00048230635070109886, 0.0016421900100564637, 0.006356727399716979, 0.015544691887878997, 0.042853200977141966, 0.09576208885577311, 0.2889577135438804, 0.6144569506852565, 1.2049620281730158, 2.872381660202119, 8.820838524856457, 22.498733393966585, 112.35101706662647, 313.51895013443016, 808.7089589899316]
rho10 =1.0e-24*np.array([2226636679.3744407, 1570724581.363684, 929572849.2847463, 333628602.60461193, 34139683.57738917, 2814897.544783489, 574903.9448900474, 161508.05385836613, 18952.83851578146, 1512.8956673206972, 254.11776584540104, 18.364374807450503, 6.9094072447067205, 0.6679662135727504, 0.1653153922768664])
add_profile("halo1", 10.0, "density", r10, rho10, r_unit="pc", y_unit="g/cm^3")

r50 = [0.00048230635070109886, 0.0019395667532673823, 0.006356727399716979, 0.015544691887878997, 0.042853200977141966, 0.09576208885577311, 0.26666425448375003, 0.5110736539966564, 1.2049620281730158, 3.2724083825169683, 8.820838524856457, 29.010979367918175, 112.35101706662647, 313.51895013443016, 808.7089589899316]
rho50 = 1.0e-24*np.array([2226636679.3744407, 1743875674.4978206, 929572849.2847463, 333628602.60461193, 34139683.57738917, 2814897.544783489, 354445.5206094803, 69212.20025332393, 18952.83851578146, 3441.234588050047, 254.11776584540104, 49.536796541928545, 6.9094072447067205, 0.6679662135727504, 0.1653153922768664])
add_profile("halo1", 50.0, "density", r50, rho50, r_unit="pc", y_unit="g/cm^3") 

r100 = [0.00044, 0.00108, 0.00315, 0.00713, 0.01694, 0.0442, 0.13708, 0.39575, 1.29814, 4.18251, 10.63536, 22.81536, 161.27474, 701.52518, 68.67869]
rho100 = 1.0e-24 * np.array([1692380846.95792, 1198585016.41695, 739217327.27461, 673577867.06067, 181950357.49772, 27213792.18295, 12683243.26118, 3129170.22792, 132130.93791, 10331.62066, 1445.40306, 194.92133, 2.44378, 0.1303, 21.97938])
add_profile("halo1", 100.0, "density", r100, rho100, r_unit="pc", y_unit="g/cm^3")

r1000 = [0.00046, 0.00101, 0.00273, 0.00695, 0.01709, 0.05119, 0.13653, 0.39575, 1.29814, 4.18251, 10.63536, 22.81536, 161.27474, 701.52518, 68.67869]
rho1000 = 1.0e-24 * np.array([10008794781.14461, 8705731925.21613, 6084247081.14796, 2725901839.66557, 974019544.24272, 221377412.21515, 28253908.96922, 3129170.22792, 132130.93791, 10331.62066, 1445.40306, 194.92133, 2.44378, 0.1303, 21.97938])
add_profile("halo1", 1000.0, "density", r1000, rho1000, r_unit="pc", y_unit="g/cm^3")


#2.3 fH2 profile
r0 = [0.00034, 0.00106, 0.0069, 0.02827, 0.16428, 0.84514, 3.83637, 10.33738, 22.98916, 48.29276, 203.35932, 420.68913, 108.95213, 718.95561]
fH2_0 = [0.00319, 0.00303, 0.00231, 0.00172, 0.0012, 0.00091, 0.00064, 0.00027, 0.00013, 0.00008, 0.00003, 0.00004, 0.00005, 0.00002]
add_profile("halo1", 0.0, "fH2", r0, fH2_0, r_unit="pc", y_unit=None)

r01 = [0.0005, 0.00126, 0.00538, 0.02827, 0.16428, 0.84514, 3.83637, 10.33738, 18.58832, 26.10842, 34.6112, 67.00282]
fH2_01 = [0.0047, 0.00457, 0.00293, 0.00172, 0.0012, 0.00091, 0.00064, 0.00027, 0.00011, 0.00005, 0.00002, 0.00001]
add_profile("halo1", 0.1, "fH2", r01, fH2_01, r_unit="pc", y_unit=None)

r1 = [0.00052, 0.00181, 0.00538, 0.02827, 0.14517, 0.72707, 2.42141, 6.78575, 11.75897, 15.13466, 18.14938, 21.30909]
fH2_1 = [0.0049, 0.00377, 0.00293, 0.00172, 0.00118, 0.00081, 0.00055, 0.00033, 0.00012, 0.00005, 0.00002, 0.00001]
add_profile("halo1", 1.0, "fH2", r1, fH2_1, r_unit="pc", y_unit=None)

r5 = [0.0004259313185670526, 0.001781979519906398, 0.008337973634367963, 0.032533208411717626, 0.09196130011149084, 0.26641438975941195, 0.753425512147508, 2.02128638582587, 4.408469949003642, 7.11222710447131, 10.952162590337483]
fH2_5 = [0.0025798035219728433, 0.002726940442761194, 0.0022937804201509926, 0.0012494570657738211, 0.0008790724567173884, 0.0007558836102784575, 0.0006588162258881097, 0.00033378657622362126, 0.00011010232793485985, 0.00004342548791638736, 0.000013909966999458965]
add_profile("halo1", 5.0, "fH2", r5, fH2_5, r_unit="pc", y_unit=None)

r10 = [0.0004259313185670526, 0.001781979519906398, 0.008337973634367963, 0.028572018763353055, 0.08663723831370905, 0.26641438975941195, 0.753425512147508, 2.02128638582587, 6.8633020185882065, 12.101289048093445, 17.52303792588523]
fH2_10 = [0.0025798035219728433, 0.002726940442761194, 0.0022937804201509926, 0.0016096665374605208, 0.0010868115206115558, 0.0007558836102784575, 0.0006588162258881097, 0.00033378657622362126, 0.0001682927654365427, 0.00004720885828812877, 0.000015167069933237267]
add_profile("halo1", 10.0, "fH2", r10, fH2_10, r_unit="pc", y_unit=None)

r50 = [0.0004259313185670526, 0.001311725896316336, 0.00489318823087793, 0.011438479588721824, 0.04835809915590955, 0.13980362541767202, 0.35205329310918504, 0.979337288155285, 4.064052301267865, 8.335248419978672, 17.52303792588523]
fH2_50 = [0.0025798035219728433, 0.0027142099607280925, 0.0021724150541818097, 0.001333503252025404, 0.000868494963459788, 0.0005773901284339241, 0.00035082845640770007, 0.0002489501603186649, 0.00016512883414599767, 0.000050407135944191675, 0.000015167069933237267]
add_profile("halo1", 50.0, "fH2", r50, fH2_50, r_unit="pc", y_unit=None)

r100 = [0.00042, 0.0015, 0.00466, 0.02041, 0.05138, 0.15228, 0.81539, 2.40737, 5.34509, 8.9421, 12.9866, 16.54554]
fH2_100 = [0.00266, 0.00196, 0.00162, 0.0013, 0.00076, 0.00035, 0.0001, 0.00019, 0.00023, 0.00007, 0.00003, 0.00001]
add_profile("halo1", 100.0, "fH2", r100, fH2_100, r_unit="pc", y_unit=None)

r1000 = [0.00042, 0.00156, 0.00536, 0.02264, 0.14541, 0.49829, 0.67808, 1.06397, 1.80078, 2.25608, 3.25278, 4.53198]
fH2_1000 = [0.00266, 0.00275, 0.00265, 0.00214, 0.00158, 0.00085, 0.00039, 0.00016, 0.00007, 0.00003, 0.00002, 0.00001]
add_profile("halo1", 1000.0, "fH2", r1000, fH2_1000, r_unit="pc", y_unit=None)

Latif2019_data_copy = copy.deepcopy(Latif2019_data)

# ------------------------------------------------------------
#                      Data analysis routines
# ------------------------------------------------------------


def _get_raw_xy(halo: str, j21: float, quantity: str):
    """
    Fetch raw (r, y) from Latif2019_data and make it well-behaved:
    - remove non-finite points
    - sort by radius ascending
    - drop duplicate radii (keep first)
    Returns (r_sorted_unique, y_aligned).
    """
    block = Latif2019_data[halo][quantity][float(j21)]
    r = np.asarray(block["r"], float)
    y = np.asarray(block["y"], float)

    # remove NaN/Inf
    m = np.isfinite(r) & np.isfinite(y)
    r, y = r[m], y[m]

    # sort ascending
    order = np.argsort(r)
    r, y = r[order], y[order]

    # drop duplicates in r
    r_unique, idx = np.unique(r, return_index=True)
    y_unique = y[idx]
    return r_unique, y_unique
"""
def _interp_log_clamped(x, y, x_eval, *, floor=None, ceil=None):
    
    # Log–log interpolation with clamped extrapolation.

    # Behavior:
    # - If x>0 and y>0, interpolate in log10-space; otherwise fall back to linear.
    # - Extrapolation is handled by clamping x_eval to [x_min, x_max] (i.e., use endpoint values).
    # - Optionally enforce floor/ceil on the result.
   
    x = np.asarray(x, float); y = np.asarray(y, float)
    x_eval = np.asarray(x_eval, float)

    # clamp evaluation points to data range (endpoint extrapolation)
    x_clip = np.clip(x_eval, x[0], x[-1])

    # choose log or linear interpolation
    if np.all(x > 0) and np.all(y > 0) and np.all(x_clip > 0):
        yy = np.interp(np.log10(x_clip), np.log10(x), np.log10(y))
        y_eval = 10.0**yy
    else:
        y_eval = np.interp(x_clip, x, y)

    # apply optional bounds
    if floor is not None:
        y_eval = np.maximum(y_eval, floor)
    if ceil is not None:
        y_eval = np.minimum(y_eval, ceil)
    return y_eval
"""

def _interp_log_clamped(x, y, x_eval, *, floor=None, ceil=None):
    """
    Interpolate y(x) at x_eval.

    Behavior:
    - Clean input (finite-only, sort by x ascending, drop duplicate x).
    - Prefer log–log interpolation if x>0 and y>0; otherwise fall back to linear.
    - Extrapolation:
        * if `floor` is not None: outside-range values are set to `floor`
        * else: endpoint-constant (clamped to y(xmin)/y(xmax))
    - Finally apply optional floor/ceil bounds to the entire result.
    """

    x = np.asarray(x, float); y = np.asarray(y, float)
    x_eval = np.asarray(x_eval, float)

    # --- clean: keep finite, sort, unique ---
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        raise ValueError("Empty input arrays after removing non-finite values.")
    order = np.argsort(x)
    x, y = x[order], y[order]
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]
    x, y = xu, yu

    # degenerate cases
    if x.size == 1:
        y_out = np.full_like(x_eval, y[0], dtype=float)
        if floor is not None:
            y_out = np.maximum(y_out, floor)
        if ceil is not None:
            y_out = np.minimum(y_out, ceil)
        return y_out

    xmin, xmax = x[0], x[-1]
    inside = (x_eval >= xmin) & (x_eval <= xmax)
    left   = (x_eval <  xmin)
    right  = (x_eval >  xmax)

    # decide log vs linear space (ensure positivity if using log)
    use_log = np.all(x > 0)
    y_for_interp = y


    if use_log:
        if floor is not None and floor > 0:
            y_for_interp = np.maximum(y_for_interp, floor)
        use_log = use_log and np.all(y_for_interp > 0)

    # interpolate on inside points
    if use_log:
        xi = np.log10(x); yi = np.log10(y_for_interp)
        y_inside = 10.0**np.interp(np.log10(x_eval[inside]), xi, yi)
    else:
        y_inside = np.interp(x_eval[inside], x, y)

    # assemble output
    y_out = np.empty_like(x_eval, dtype=float)
    y_out[inside] = y_inside

    # outside-range handling
    if floor is not None:
        # For fH2-like usage: directly set to floor outside the data range
        
        #debug test: reset floor = 0.5 * last y value
        
        floor = 0.5 * y[-1]
        y_out[left]  = floor
        y_out[right] = floor
    else:
        # For other quantities: endpoint-constant (clamped)
        y_out[left]  = y[0]
        y_out[right] = y[-1]

    # global bounds
    if floor is not None:
        y_out = np.maximum(y_out, floor)
    if ceil is not None:
        y_out = np.minimum(y_out, ceil)

    return y_out


def _build_r_grid(r_min_pc, r_max_pc, n_samples):
    """Log-spaced radius grid in pc."""
    return np.logspace(np.log10(r_min_pc), np.log10(r_max_pc), int(n_samples))

# ------------------------------
# 1) Mass-weighted H2 fraction
# ------------------------------
def mass_weighted_fH2(halo: str,
                      j21: float,
                      r_min_pc: float = 1e-6,
                      r_max_pc: float = 1e3,
                      n_samples: int = 1000,
                      fH2_floor: float = 5e-6):
    """
    Compute mass-weighted <f_H2> and total gas mass over [r_min_pc, r_max_pc].

        <f_H2> = ∫ 4π r^2 ρ(r) f_H2(r) dr / ∫ 4π r^2 ρ(r) dr

    Assumptions:
    - density is in g/cm^3
    - radius is in pc
    - f_H2 is clamped to at least fH2_floor (useful for strong LW where f_H2 → very small)
    - extrapolation is endpoint-constant

    Returns dict with:
      r_pc, density, fH2, fH2_avg, Mgas_Msun
    """
    # fetch density and fH2 profiles
    r_rho, rho = _get_raw_xy(halo, j21, "density")
    r_fh2, fh2 = _get_raw_xy(halo, j21, "fH2")

    # evaluation grid
    r_pc = _build_r_grid(r_min_pc, r_max_pc, n_samples)

    # interpolate (rho positive; fH2 with a lower floor to avoid log(0))
    rho_eval = _interp_log_clamped(r_rho, rho, r_pc)
    fH2_eval = _interp_log_clamped(r_fh2, np.maximum(fh2, fH2_floor), r_pc, floor=fH2_floor)

    # integrate
    r_cm = r_pc * PC_TO_CM
    dMdr = 4.0 * np.pi * (r_cm**2) * rho_eval            # mass shell derivative
    M_g   = np.trapezoid(dMdr, r_cm)                     # total gas mass [g]
    fH2_avg = np.trapezoid(fH2_eval * dMdr, r_cm) / M_g

    return {
        "r_pc": r_pc,
        "density": rho_eval,
        "fH2": fH2_eval,
        "fH2_avg": fH2_avg,
        "Mgas_Msun": M_g / MSUN_G,
    }

# ------------------------------
# 2) Total gas mass from density
# ------------------------------
def total_gas_mass(halo: str,
                   j21: float,
                   r_min_pc: float = 1e-6,
                   r_max_pc: float = 1e3,
                   n_samples: int = 1000):
    """
    Compute total gas mass by integrating density only:

        M_gas = ∫ 4π r^2 ρ(r) dr   (returned in Msun)

    Extrapolation is endpoint-constant; interpolation prefers log–log when possible.
    """
    r_rho, rho = _get_raw_xy(halo, j21, "density")
    r_pc = _build_r_grid(r_min_pc, r_max_pc, n_samples)
    rho_eval = _interp_log_clamped(r_rho, rho, r_pc)

    r_cm = r_pc * PC_TO_CM
    dMdr = 4.0 * np.pi * (r_cm**2) * rho_eval
    M_g   = np.trapezoid(dMdr, r_cm)
    return M_g / MSUN_G

#plot and compare profile for different j21

def plot_profiles(halo: str, quantity: str, j21_list: list, r_min_pc=1e-3, r_max_pc=1e3, n_samples=1000):
    outputdir = "/home/zwu/21cm_project/unified_model/Analytic_results/Latif/"
    color_map_halo6 = {
        0.1: "green",
        1.0: "blue",
        10.0: "cyan",
        100.0: "magenta",
        1000.0: "red",
    }
    color_map_halo1 = {
        0.0: "green",
        0.1: "blue",
        1.0: "blue",
        5.0: "cyan",
        10.0: "magenta",
        50.0: "red",
        100.0: "green",
        1000.0: "red"
    }
    linestyle_halo1 = {
        0.0: "dotted",
        0.1: "dashed",
        1.0: "solid",
        5.0: "solid",
        10.0: "solid",
        50.0: "solid",
        100.0: "solid",
        1000.0: "dotted"
    }

    plt.figure(figsize=(8,6))
    for j21 in j21_list:
        r, y = _get_raw_xy(halo, j21, quantity)
        r_eval = _build_r_grid(r_min_pc, r_max_pc, n_samples)
        if quantity == "fH2":
            y_eval = _interp_log_clamped(r, y, r_eval, floor=5e-6)
        else:
            y_eval = _interp_log_clamped(r, y, r_eval)

        if halo == "halo6":
            color = color_map_halo6.get(j21, "black")
            plt.plot(r_eval, y_eval, label=f"J21={j21}", color=color)
        elif halo == "halo1":
            color = color_map_halo1.get(j21, "black")
            linestyle = linestyle_halo1.get(j21, "solid")
            plt.plot(r_eval, y_eval, label=f"J21={j21}", color=color, linestyle=linestyle)
        
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f"Radius (pc)")
    unit = Latif2019_data[halo][quantity][float(j21_list[0])]['units']['y']
    if unit:
        plt.ylabel(f"{quantity} ({unit})")
    else:
        plt.ylabel(f"{quantity} (dimensionless)")
    plt.title(f"{quantity} profile for {halo}")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.xscale('log')
    filename = os.path.join(outputdir, f"{halo}_{quantity}_profile.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot to {filename}")

if __name__ == "__main__":
    halo_name = "halo6"
    j21_val  = 1000.0
    res = mass_weighted_fH2(halo_name, j21_val, r_min_pc=1e-3, r_max_pc=1e3, n_samples=1000, fH2_floor=5e-6)
    Mgas_Msun = res['Mgas_Msun']
    print(f"fH2_avg: {res['fH2_avg']}, Mgas_Msun: {Mgas_Msun:.3e}")
    #print z_collapse and M_collapse
    M_collapse_Msun = Latif2019_data[halo_name][j21_val]['M_collapse_Msun']
    print(f"z_collapse: {Latif2019_data[halo_name][j21_val]['z_collapse']}, M_collapse_Msun: {M_collapse_Msun:.3e}")
    print("Mgas/M_collapse:", Mgas_Msun / M_collapse_Msun)
    print("Omega_b/Omega_m:", Omega_b / Omega_m)

    #plot profiles
    j21_list_halo6 = [0.1, 1.0, 10.0, 100.0, 1000.0]
    plot_profiles(halo_name, "fH2", j21_list_halo6, r_min_pc=1e-3, r_max_pc=1e3, n_samples=1000)