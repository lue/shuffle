import numpy as np
import matplotlib.pyplot as plt
import yt
import glob
from yt.visualization.api import get_multi_plot
import matplotlib.colorbar as cb
from matplotlib.colors import LogNorm

path = '/scratch/kaurov/rei/shuffle10/'


def _IonizedHydrogen(field, data):
    return data["RT_HVAR_HII"] / ( data["RT_HVAR_HII"] + data["RT_HVAR_HI"] )

yt.add_field("IonizedHydrogen", function=_IonizedHydrogen, )#units=r"\frac{\rho_{HII}}{rho_H}")

for j in range(8):
    pathij = path + "OUT_shuffle_%04i_%04i"%(4,j)
    print(pathij)
    snaps = glob.glob(pathij+"/*/*.art")
    snaps.sort()
    snaps_n = np.array([np.float(snaps[i][-10:-4]) for i in range(len(snaps))])
    data_i = np.zeros([800, 800, len(snaps)])
    data_d = np.zeros([800, 800, len(snaps)])
    # a_list = np.zeros(len(snaps))
    a_list = snaps_n.copy()
    ion_frac_M = np.zeros(len(a_list))
    ion_frac_V = np.zeros(len(a_list))
    for k in range(len(snaps)):
        ds = yt.load(snaps[k])
        ad = ds.all_data()
        average_value1 = ad.quantities.weighted_average_quantity('IonizedHydrogen', 'density')
        ion_frac_M[k] = average_value1
        average_value2 = ad.quantities.weighted_average_quantity('IonizedHydrogen', 'cell_volume')
        ion_frac_V[k] = average_value2
        print(a_list[k], average_value1, average_value2)
    np.savez("slices_%04i_%04i.npz"%(4,j), a_list=a_list, ion_frac_M=ion_frac_M, ion_frac_V=ion_frac_V)


# for k in range(15,25):

k=0

for j in range(8):
    pathij = path + "OUT_shuffle_%04i_%04i"%(4,j)
    print(pathij)
    snaps = glob.glob(pathij+"/*/*.art")
    snaps.sort()
    snaps_n = np.array([np.float(snaps[i][-10:-4]) for i in range(len(snaps))])
    data_i = np.zeros([800, 800, len(snaps)])
    data_d = np.zeros([800, 800, len(snaps)])
    # a_list = np.zeros(len(snaps))
    a_list = snaps_n.copy()
    for k in range(len(snaps)):
        ds = yt.load(snaps[k])
        slc = yt.SlicePlot(ds, 'x', "IonizedHydrogen")
        slc_frb = slc.data_source.to_frb((10, "Mpccm/h"), 800)
        data_i[:,:,k] = np.array(slc_frb['IonizedHydrogen'])
        data_d[:,:,k] = np.array(slc_frb['density'])
        # snaps_ds_t.append((snaps_n[k]))
        # snaps_ds.append(ds)
        # Create density slices in all three axes.
        # yt.SlicePlot(ds, 'x', "density").save()
        # yt.SlicePlot(ds, 'x', "IonizedHydrogen").save()
    np.savez("slices_%04i_%04i.npz"%(4,j), a_list=a_list, data_i=data_i, data_d=data_d)



a_list_all = np.linspace(1./(1+9), 1./(1+6),20)
for j in range(8):
    ff = np.load("slices_%04i_%04i.npz" % (4, j))
    a_list = ff['a_list']
    data_i = ff['data_i']
    data_d = ff['data_d']
    res_i = np.zeros([data_i.shape[0],data_i.shape[1],len(a_list_all)])
    res_d = np.zeros([data_i.shape[0],data_i.shape[1],len(a_list_all)])
    for a1 in range(data_i.shape[0]):
        for a2 in range(data_i.shape[1]):
            res_i[a1,a2,:] = np.interp(a_list_all, a_list, data_i[a1,a2,:], left=-1, right=-1)
            res_d[a1,a2,:] = np.interp(a_list_all, a_list, data_d[a1,a2,:], left=-1, right=-1)
    np.savez("slices_%04i_%04i_m.npz" % (4, j), a_list=a_list_all, data_i=res_i, data_d=res_d)




data = []
for j in range(8):
    ff = np.load("slices_%04i_%04i_m.npz" % (4, j))
    a_list = ff['a_list']
    data_i = ff['data_i']
    data_d = ff['data_d']
    data.append([a_list,data_i,data_d])


fig = plt.figure(figsize=(10,5),dpi=300)
plt.clf()
for i in range(1,9):
    ax = plt.subplot(2,4,i)
    plt.imshow(data[i-1][1][:,:,5], vmin=0, vmax=1)#, norm=LogNorm())
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


plt.tight_layout()
plt.show()
# plt.savefig('%i.png'%k)


def ssum(data, j):
    N = data[0][1].shape[0]
    temp = np.zeros([N,N])
    nn=0
    for i in range(len(data)):
        if (data[i][1][0, 0, j])>=0:
            nn+=1
            temp+=data[i][1][:, :, j]
        else:
            print(i)
    return temp/nn, nn

fig = plt.figure(figsize=(6,5),dpi=150)
plt.clf()
temp, nn = ssum(data,11)
plt.contourf(temp, levels=np.linspace(0, 1, nn+1))#, norm=LogNorm())

plt.gca().xaxis.set_ticks([])
plt.gca().yaxis.set_ticks([])
# plt.axis('equal')

plt.gca().set_aspect('equal')
plt.gca().autoscale(tight=True)

# plt.gca().yaxis.set_visible(False)

plt.xlabel(r'$10\;h^{-1}\mathrm{Mpc}$')

plt.colorbar()

# plt.tight_layout()
plt.show()
