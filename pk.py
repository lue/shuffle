import numpy as np
import matplotlib.pyplot as plt


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


import glob

a = 0.125
res_all = []
for i in np.arange(8,24):
    print(i)
    temp = []
    # files = glob.glob('/scratch/kaurov/rei/shuffle10/xOUT_xshuffle_%04i_%04i/*.bin'%(4,i))
    files = glob.glob('/home/alex.kaurov/scratch/shuffle/jobs/xLOG_xshuffle_%04i_%04i/rei.log'%(4,i))
    d = np.genfromtxt(files[0])
    plt.plot(1./d[:,0]-1., d[:,8])

plt.xlim([6,10])
plt.show()


z = 7.2
a = 1./(1.+z)
print(z,a)
res_all = []
for i in np.arange(8,24):
    print(i)
    temp = []
    # files = glob.glob('/scratch/kaurov/rei/shuffle10/xOUT_xshuffle_%04i_%04i/*.bin'%(4,i))
    files = glob.glob('/home/alex.kaurov/scratch/shuffle/jobs/xOUT_xshuffle_%04i_%04i/*.bin'%(4,i))
    print(files)
    files.sort()
    snaps_n = np.array([np.float(files[i][-10:-4]) for i in range(len(files))])
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



# plt.figure(1)
# for i in range(len(res_all)):
#     plt.subplot(3,3,i+1)
#     plt.imshow(res_all[i][:,:,0])
#
# plt.show()
#
#
plt.figure(2)
plt.imshow(np.array(res_all).sum(0)[:, :, 0])
plt.colorbar()
plt.show()




# from mayavi.mlab import *
# contour3d(np.array(res_all).sum(0))