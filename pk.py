import numpy as np
import matplotlib.pyplot as plt
import glob


def readifrit(path, nvar=0, moden=2, skipmoden=2):
    import struct
    import numpy as np
    openfile=open(path, "rb")
    dump = np.fromfile(openfile, dtype='i', count=moden)
    N1,N2,N3=np.fromfile(openfile, dtype='i', count=3)
    print(N1,N2,N3)
    dump = np.fromfile(openfile, dtype='i', count=moden)
    data = np.zeros([N1, N2, N3])
    j = 0
    for i in range(nvar):
        openfile.seek(4*skipmoden, 1)
        for j in range(4):
            openfile.seek(N1*N2*N3, 1)
        openfile.seek(4*moden, 1)
    openfile.seek(4*skipmoden, 1)
    data[:, :, :] = np.reshape(np.fromfile(openfile, dtype='f4', count=N1 * N2 * N3), [N1, N2, N3])
    openfile.close()
    return N1,N2,N3,data


def downsample(data,f=2):
    data_new=(data[::2, ::2, ::2]+data[1::2, ::2, ::2]+
              data[::2, 1::2, ::2]+data[::2, ::2, 1::2]+
              data[1::2, 1::2, ::2]+data[::2, 1::2, 1::2]+
              data[1::2, ::2, 1::2]+data[1::2, 1::2, 1::2])
    return data_new

def pk(data, box_size, k_list_phys, mode=0, usefftw=False):
    N = data.shape[0]
    k_list = k_list_phys*box_size/N
    data=np.fft.rfftn(data)
    kx, ky, kz = np.mgrid[:N, :N, :(N/2+1)]
    kx[kx > N/2-1] = kx[kx > N/2-1]-N
    ky[ky > N/2-1] = ky[ky > N/2-1]-N
    kz[kz > N/2-1] = kz[kz > N/2-1]-N
    k=2.0*np.pi*np.sqrt(kx**2+ky**2+kz**2)/N
    if mode == 1:
        kf = 2.0*np.pi/N
        res = np.zeros(len(k_list)-1)
        for i in range(len(k_list)-1):
            if np.sum((k >= k_list[i]) & (k < k_list[i+1]))>0:
                res[i] = np.mean(np.abs(data[(k >= k_list[i]) & (k < k_list[i+1])])**2)
        return res*box_size**3 / N**6
    if mode==0:
        kf=2.0*np.pi/N
        h1, dump = np.histogram(k.flat,weights=np.abs(data.flat)**2,bins=k_list)
        h2, dump = np.histogram(k.flat,bins=k_list)
        h2[h2==0] = 1.0
        res = h1/h2
        return res*box_size**3/N**6
    if mode==2:
        kf=2.0*np.pi/N
        res=np.zeros(len(k_list))
        for i in range(len(k_list)):
            res[i]=np.mean(np.abs(data[(k>=k_list[i]-kf) & (k<k_list[i]+kf)])**2)
        return res*box_size**3/N**6
    if mode==2:
        return k,data


#####

machine = 'ias'
mode = 32

#####
i=0
# for s in [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]:
for s in [8, 16, 32, 64, 128]:
    i+=1
    if machine=='ias':
        files = glob.glob('/home/kaurov/scratch/rei/shuffle10/xLOG_xshuffle_%04i_%04i/rei.log'%(s,101))
    if machine=='prs':
        files = glob.glob('/home/alex.kaurov/scratch/shuffle/jobs/xOUT_xshuffle_%04i_%04i/*.bin'%(s,101))
    files.sort()
    d = np.genfromtxt(files[0])
    # plt.imshow(np.log10(d1[:,:,0]), vmax=2,vmin=-1)
    # plt.imshow((d1[:,:,0]), vmax=1,vmin=0)
    plt.plot(1./d[::-1,0]-1., d[::-1,8])

plt.xlim([5,12])
plt.show()

#####

a = 0.1000
res_all = []
z_list = np.linspace(5,12,100)
for i in np.arange(0,24):
    print(i)
    temp = []
    # files = glob.glob('/scratch/kaurov/rei/shuffle10/xOUT_xshuffle_%04i_%04i/*.bin'%(4,i))
    if machine=='ias':
        files = glob.glob('/home/kaurov/scratch/rei/shuffle10/xLOG_xshuffle_%04i_%04i/rei.log'%(mode,i))
    if machine == 'prs':
        files = glob.glob('/home/alex.kaurov/scratch/shuffle/jobs/xLOG_xshuffle_%04i_%04i/rei.log'%(mode,i))
    if len(files)>0:
        d = np.genfromtxt(files[0])
        temp = np.interp(z_list, 1./d[::-1,0]-1., d[::-1,8], left=np.nan)
        res_all.append(temp)

res_all = np.array(res_all)

res_mean = np.nanmean(res_all, 0)

ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
plt.fill_between(z_list, np.nanmin(res_all, 0), np.nanmax(res_all, 0), alpha=0.5)
plt.plot(z_list, res_mean, 'k')
plt.ylabel(r'$f_\mathrm{HI}$')
plt.xlim([5,10])
plt.setp(ax1.get_xticklabels() , visible=False)
ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
for i in range(res_all.shape[0]):
    plt.plot(z_list, res_all[i,:] - res_mean)

plt.xlim([5,10])
plt.xlabel(r'$z$')
plt.ylabel(r'$\mathrm{\Delta} f_\mathrm{HI}$')
plt.tight_layout()
plt.subplots_adjust(hspace=0.03)
plt.show()

###


import glob

a = 0.1000
res_all = []
z_list = np.linspace(5,20,100)
for i in np.arange(0,24):
    print(i)
    temp = []
    # files = glob.glob('/scratch/kaurov/rei/shuffle10/xOUT_xshuffle_%04i_%04i/*.bin'%(2,i))
    if machine=='ias':
        files = glob.glob('/home/kaurov/scratch/rei/shuffle10/xLOG_xshuffle_%04i_%04i/rei.log'%(mode,i))
    if machine=='prs':
        files = glob.glob('/home/alex.kaurov/scratch/shuffle/jobs/xLOG_xshuffle_%04i_%04i/rei.log'%(mode,i))
    if len(files)>0:
        d = np.genfromtxt(files[0])
        temp = np.interp(z_list, 1./d[::-1,0]-1., d[::-1,25], left=np.nan)
        res_all.append(temp)

res_all = np.array(res_all)*1000

res_mean = np.nanmean(res_all, 0)


ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
plt.fill_between(z_list, np.nanmin(res_all, 0), np.nanmax(res_all, 0), alpha=0.5)
plt.plot(z_list, res_mean, 'k')
plt.ylabel(r'$T_b\;\mathrm{[mK]}$')
plt.xlim([5,16])
# plt.yscale('log')
plt.setp(ax1.get_xticklabels() , visible=False)
ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)

from matplotlib import cm

for i, val in enumerate(cm.jet(np.linspace(0,1,res_all.shape[0]))):
    plt.plot(z_list, res_all[i,:] - res_mean, color=val)

plt.xlim([5,16])
plt.xlabel(r'$z$')
plt.ylabel(r'$\mathrm{\Delta} T_b\;\mathrm{[mK]}$')
plt.tight_layout()
plt.subplots_adjust(hspace=0.03)
plt.show()



###

z = 7.
a = 1./(1.+z)
print(z,a)
res_all = []
for i in np.concatenate([np.arange(0,24),[101]]):
    print(i)
    temp = []
    # files = glob.glob('/scratch/kaurov/rei/shuffle10/xOUT_xshuffle_%04i_%04i/*.bin'%(2,i))
    if machine=='ias':
        files = glob.glob('/home/kaurov/scratch/rei/shuffle10/xOUT_xshuffle_%04i_%04i/*.bin'%(mode,i))
    if machine=='prs':
        files = glob.glob('/home/alex.kaurov/scratch/shuffle/jobs/xOUT_xshuffle_%04i_%04i/*.bin'%(mode,i))
    files.sort()
    print(files)
    snaps_n = np.array([np.float(files[i][-10:-4]) for i in range(len(files))])
    # print(snaps_n[-1])
    # a_list = np.zeros(len(snaps))
    xn = np.where(snaps_n>=a)[0]
    if len(xn)>0:
        xn = xn[0]
        d1 = readifrit(files[xn])
        d2 = readifrit(files[xn-1])
        r = (snaps_n[xn]-a) / (snaps_n[xn] - snaps_n[xn-1])
        res = d1[3]*(1.-r) + r*d2[3]
        res_all.append(res)
    else:
        print('no data')


plt.figure(2)
N = len(res_all)
plt.contourf(np.array(res_all).sum(0)[:, :, 0]/N, levels=np.linspace(0, 1, N+1), cmap='gray')#, norm=LogNorm())
plt.gca().xaxis.set_ticks([])
plt.gca().yaxis.set_ticks([])
# plt.axis('equal')

plt.gca().set_aspect('equal')
plt.gca().autoscale(tight=True)

# plt.gca().yaxis.set_visible(False)

plt.xlabel(r'$10\;h^{-1}\mathrm{Mpc}$')

plt.colorbar()
plt.show()


# from mayavi.mlab import *
# contour3d(np.array(res_all).sum(0))

######


z = 7.
a = 1./(1.+z)
nvar=4
print(z,a)
res_all = []
res_all2 = []
for i in np.arange(0,24):
    print(i)
    temp = []
    # files = glob.glob('/scratch/kaurov/rei/shuffle10/xOUT_xshuffle_%04i_%04i/*.bin'%(2,i))
    if machine=='ias':
        files = glob.glob('/home/kaurov/scratch/rei/shuffle10/xOUT_xshuffle_%04i_%04i/*.bin'%(mode,i))
    if machine=='prs':
        files = glob.glob('/home/alex.kaurov/scratch/shuffle/jobs/xOUT_xshuffle_%04i_%04i/*.bin'%(mode,i))
    print(files)
    files.sort()
    snaps_n = np.array([np.float(files[i][-10:-4]) for i in range(len(files))])
    # a_list = np.zeros(len(snaps))
    xn = np.where(snaps_n>=a)[0]
    if len(xn)>0:
        xn = xn[0]
        d1 = readifrit(files[xn],nvar=nvar)
        d2 = readifrit(files[xn-1],nvar=nvar)
        r = (snaps_n[xn]-a) / (snaps_n[xn] - snaps_n[xn-1])
        res = d1[3]*(1.-r) + r*d2[3]
        res_all.append(res.mean())
        print(res.mean())
        if machine == 'ias':
            files = glob.glob('/home/kaurov/scratch/rei/shuffle10/xLOG_xshuffle_%04i_%04i/rei.log' % (mode, i))
        d = np.genfromtxt(files[0])
        temp = np.interp(z, 1. / d[::-1, 0] - 1., d[::-1, 25], left=np.nan)
        res_all2.append(temp)
    else:
        print('no data')


######


z = 7.
a = 1./(1.+z)
nvar=4
print(z,a)
res_all = []
res_all2 = []
for mode in [8,16,32,64,128]:
    for i in [0,1,2,3,4,5,6,7,101]:#np.arange(0,24):
        print(i)
        temp = []
        # files = glob.glob('/scratch/kaurov/rei/shuffle10/xOUT_xshuffle_%04i_%04i/*.bin'%(2,i))
        if machine=='ias':
            files = glob.glob('/home/kaurov/scratch/rei/shuffle10/xOUT_xshuffle_%04i_%04i/*.bin'%(mode,i))
        if machine=='prs':
            files = glob.glob('/home/alex.kaurov/scratch/shuffle/jobs/xOUT_xshuffle_%04i_%04i/*.bin'%(mode,i))
        print(files)
        files.sort()
        snaps_n = np.array([np.float(files[i][-10:-4]) for i in range(len(files))])
        # a_list = np.zeros(len(snaps))
        # xn = np.where(snaps_n>=a)[0]
        # if len(xn)>0:
        res_all = []
        res_all2 = []
        for xn in range(len(snaps_n)):
            if len(glob.glob(files[xn]+'.8dat'))==0:
                d1 = readifrit(files[xn],nvar=nvar)
                res = d1[3]
                res_all.append(res[:128,:128,:128].mean())
                temp = []
                for a1 in [0,1]:
                    for a2 in [0,1]:
                        for a3 in [0,1]:
                            temp.append(res[a1*128:(a1+1)*128,
                                        a2 * 128:(a2 + 1) * 128,
                                        a3 * 128:(a3 + 1) * 128].mean())
                np.savetxt(files[xn]+'.8dat', temp)
            # print(res.mean())
            # if machine == 'ias':
            #     files = glob.glob('/home/kaurov/scratch/rei/shuffle10/xLOG_xshuffle_%04i_%04i/rei.log' % (mode, i))
            # d = np.genfromtxt(files[0])
            # temp = np.interp(z, 1. / d[::-1, 0] - 1., d[::-1, 25], left=np.nan)
            # res_all2.append(temp)
        # plt.plot(1./snaps_n-1, res_all)


print(z,a)
res_all = []
res_all2 = []
mode=64

for iii in range(6):
    mode = [8,16,32,64,128][iii]
    plt.subplot(2,3,iii+1)
    for i in [0,1,2,3,4,5,6,7,101]:#np.arange(0,24):
        print(i)
        temp = []
        # files = glob.glob('/scratch/kaurov/rei/shuffle10/xOUT_xshuffle_%04i_%04i/*.bin'%(2,i))
        if machine=='ias':
            files = glob.glob('/home/kaurov/scratch/rei/shuffle10/xOUT_xshuffle_%04i_%04i/*.bin.8dat'%(mode,i))
        if machine=='prs':
            files = glob.glob('/home/alex.kaurov/scratch/shuffle/jobs/xOUT_xshuffle_%04i_%04i/*.bin.8dat'%(mode,i))
        # print(files)
        files.sort()
        if len(files)>1:
            snaps_n = np.array([np.float(files[i][-15:-9]) for i in range(len(files))])
            # a_list = np.zeros(len(snaps))
            # xn = np.where(snaps_n>=a)[0]
            # if len(xn)>0:
            res_all = []
            for xn in range(len(snaps_n)):
                temp = np.genfromtxt(files[xn])
                res_all.append(temp)
            res_all = np.array(res_all)
        plt.plot(1. / snaps_n - 1, res_all[:,4])
    plt.ylim([-0.024,0.024])
    plt.xlim([6,15])

plt.show()