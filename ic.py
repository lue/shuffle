import numpy as np
from scipy.io import FortranFile
import matplotlib.pyplot as plt
import pexpect
import os

openfile=open('test.high.ref.amps', "rb")

x = np.fromfile(openfile, dtype='i', count=5)
print(x)
data = np.zeros([x[1],x[2],x[3]])
N = x[1]
for i in range(x[3]):
    print(np.fromfile(openfile, dtype='i', count=2))
    data[:,:,i]=np.reshape(np.fromfile(openfile, dtype='<f4', count=x[1]*x[2]),[x[1],x[2]])
#    print(np.fromfile(openfile, dtype='i', count=1))

openfile.close()
data = data-data.mean()

def shuffle(d, s, seed):
    N = d.shape[0]
    data = d.copy()
    np.random.seed(seed)
    print(N/s)
    for i in range(int(N/s)):
        # print(i)
        for j in range(int(N/s)):
            for k in range(int(N/s)):
                np.random.shuffle(data[i*s:(i+1)*s,j*s:(j+1)*s,k*s:(k+1)*s,])
    return data

def shuffle_fft(d,s,seed, saveamps=False):
    data = d.copy()
    N = data.shape[0]
    np.random.seed(seed)
    data_fft = np.fft.rfftn(data)
    freq1 = np.fft.fftfreq(N)
    freq2 = np.fft.rfftfreq(N)
    xv, yv, zv = np.meshgrid(freq1, freq1, freq2)
    vv = np.sqrt(xv**2+yv**2+zv**2)*N
    if saveamps:
        rand_field_1 = np.random.rand(N,N,int(N/2+1))
        data_fft[vv>s] *= np.exp(rand_field_1[vv>s]*1j*2.*np.pi)
    else:
        rand_field_1 = np.random.normal(loc=0, scale=1, size=(N, N, int(N / 2 + 1)))
        rand_field_2 = np.random.normal(loc=0, scale=1, size=(N, N, int(N / 2 + 1)))
        data_fft[vv > s] = np.sqrt(1.0*N**3)*(rand_field_1 + 1j*rand_field_2)[vv > s]
    return np.fft.irfftn(data_fft)



# data[:128,:,:]*=2
# data[:128,:,:]+=10

# data2 = shiffle_fft(data,16,1111)
# plt.subplot(121)
# plt.imshow(data[:,:,0])
# plt.subplot(122)
# plt.imshow(data2[:,:,0]); plt.show()


# f = FortranFile('test.high.mod.amps', 'w')
# f.write_record(np.array([N,N,N,0],dtype=np.int32))
# for i in range(N):
#     f.write_record(data[:,:,i].astype(np.float32))
# f.close()


# for s in [2,4,8,12,16,24,32,48,64,96,128]:
for s in [8,16,32,64,128]:
    print(s)
    for iii in range(0, 16):
        d = data.copy()
        # d = shuffle(data, s, s*10000+iii)
        d = shuffle_fft(d, 1.0*N/s, s*10000+iii)
        f = FortranFile('test.high.mod.amps', 'w')
        f.write_record(np.array([N,N,N,0],dtype=np.int32))
        for i in range(N):
            f.write_record(d[:,:,i].astype(np.float32))
        f.close()
        # os.system('source ~/.set_gcc_gic && ./bin/gic-low -m -d -b -save')
        os.system('source ~/.set_gcc_gic && ./bin/gic-high -multi -uni=2 -m -d -b')
        os.system('source ~/.set_gcc_gic && ./bin/gic2ifrit shuffle_mr2_M -den')
        # os.system('source ~/.set_gcc_gic && ./bin/gic2ifrit shuffle_mr2_D -den')
        # os.system('source ~/.set_gcc_gic && ./bin/gic2ifrit shuffle_lr_M -den')
        os.system('mkdir shuffle_%04i_%04i'%(s,iii))
        os.system('cp shuffle//* shuffle_%04i_%04i//'%(s,iii))
#        os.system('mkdir /scratch/kaurov/rei/shuffle10/LOG_%03i_%03i'%(s,i))
#        os.system('mkdir /scratch/kaurov/rei/shuffle10/OUT_%03i_%03i'%(s,i))
#        os.system('mkdir /scratch/kaurov/rei/shuffle10/LOG_%03i_%03i'%(s,i))


############
# Check
############


# def readifrit(path, nvar=0, moden=1, skipmoden=2):
#     import struct
#     import numpy as np
#     openfile=open(path, "rb")
#     dump = np.fromfile(openfile, dtype='<i', count=moden)
#     N1,N2,N3=np.fromfile(openfile, dtype='<i', count=3)
#     print(N1,N2,N3)
#     dump = np.fromfile(openfile, dtype='<i', count=moden)
#     data = np.zeros([N1, N2, N3])
#     j = 0
#     for i in range(nvar):
#         openfile.seek(4*skipmoden, 1)
#         for j in range(4):
#             openfile.seek(N1*N2*N3, 1)
#         openfile.seek(4*moden, 1)
#     openfile.seek(4*skipmoden, 1)
#     data[:, :, :] = np.reshape(np.fromfile(openfile, dtype='<f4', count=N1 * N2 * N3), [N1, N2, N3])
#     openfile.close()
#     return N1,N2,N3,data

# n1,n2,n3,d0 = readifrit('shuffle_0008_0000/shuffle_mr2_M.d.bin')
# n1,n2,n3,d1 = readifrit('shuffle_0008_0001/shuffle_mr2_M.d.bin')
#
# plt.subplot(121)
# plt.imshow(d0[:,:,0])
# plt.subplot(122)
# plt.imshow(d1[:,:,0]); plt.show()
#
#
#
#
d1 = np.fft.rfftn(data)
# d2 = np.fft.rfftn(shuffle(data, 8, 0))
d0 = np.fft.rfftn(shuffle_fft(data, 256/2, 0))
N=256
freq1 = np.fft.fftfreq(N)
freq2 = np.fft.rfftfreq(N)
xv, yv, zv = np.meshgrid(freq1, freq1, freq2)
vv = np.sqrt(xv ** 2 + yv ** 2 + zv ** 2) * N

plt.subplot(211)
plt.imshow(np.abs((d0-d1)[:,:,0]))
plt.subplot(212)
plt.imshow((vv[:,:,0]>(256./2)))
plt.show()
