import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('k', metavar='k', type=int, nargs='+',
                    help='an integer for the accumulator')
args = parser.parse_args()
print(args.k)

import numpy as np
import matplotlib.pyplot as plt
import yt
yt.enable_parallelism()
import glob
from yt.visualization.api import get_multi_plot
import matplotlib.colorbar as cb
from matplotlib.colors import LogNorm

path = '/scratch/kaurov/rei/shuffle10/'


def _IonizedHydrogen(field, data):
    return data["RT_HVAR_HII"] / ( data["RT_HVAR_HII"] + data["RT_HVAR_HI"] )

yt.add_field("IonizedHydrogen", function=_IonizedHydrogen, )#units=r"\frac{\rho_{HII}}{rho_H}")

for j in args.k:
    pathij = path + "OUT_shuffle_%04i_%04i"%(4,j)
    print(pathij)
    snaps = glob.glob(pathij+"/*/*.art")
    snaps.sort()
    snaps_n = np.array([np.float(snaps[i][-10:-4]) for i in range(len(snaps))])
    # a_list = np.zeros(len(snaps))
    a_list = snaps_n.copy()
    ion_frac_M = np.zeros(len(a_list))
    ion_frac_V = np.zeros(len(a_list))
    for k in range(len(snaps)):
        ds = yt.load(snaps[k])
        # all_data_level_2 = ds.covering_grid(level=2, left_edge=[0, 0.0, 0.0],
        #                                     dims=ds.domain_dimensions * 2 ** 2)
        # temp_i = all_data_level_2['IonizedHydrogen']
        # temp_d = all_data_level_2['density']
        # np.savez("uni_%04i_%04i_%1.4f.npz" % (4, j, a_list[k]), i = temp_i, d = temp_d)
        ad = ds.all_data()
        average_value1 = ad.quantities.weighted_average_quantity('IonizedHydrogen', 'cell_mass')
        ion_frac_M[k] = np.float(average_value1)
        average_value2 = ad.quantities.weighted_average_quantity('IonizedHydrogen', 'cell_volume')
        ion_frac_V[k] = np.float(average_value2)
        print(a_list[k], average_value1, average_value2)
        np.savez("glob_hist_%04i_%04i.npz"%(4,j), a_list=a_list, ion_frac_M=ion_frac_M, ion_frac_V=ion_frac_V)



