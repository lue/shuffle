import glob
import os
import numpy as np

files =  glob.glob('xOUT_xshuffle_0004_0015/ifrit-mesh.*.bin')
files.sort(key=os.path.getmtime)
snaps_n = np.array([np.float(files[i][-10:-4]) for i in range(len(files))])
print("\n".join(files))
for j in np.where([np.sum(snaps_n[i]>snaps_n[i:])>0 for i in range(len(snaps_n))])[0]:
    print("rm "+files[j])