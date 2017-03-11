#!/usr/bin/python
import sys
sys.path.append("../build/lib.linux-x86_64-2.7/")
import lprecmods.lpTransform as lpTransform
import matplotlib.pyplot as plt
from numpy import *
import struct

N=512
Nproj=3*N/2
Nslices=1
filter_type='None'
pad=False
cor=N/2

fid = open('./data/f', 'rb')
f=float32(reshape(struct.unpack(N*N*'f',fid.read(N*N*4)),[Nslices,N,N]))

fid = open('./data/R', 'rb')
R=float32(reshape(struct.unpack(Nproj*N*'f',fid.read(Nproj*N*4)),[Nslices,N,Nproj]))

clpthandle=lpTransform.lpTransform(N,Nproj,Nslices,filter_type,pad)
clpthandle.precompute()
clpthandle.initcmem()

Rf=clpthandle.fwd(f)
frec=clpthandle.adj(R,cor);
Rrec=clpthandle.fwd(frec)


#dot product test
sum1= sum(ndarray.flatten(Rrec)*ndarray.flatten(R))
sum2= sum(ndarray.flatten(frec)*ndarray.flatten(frec))
print linalg.norm(sum1-sum2)/linalg.norm(sum2)

plt.subplot(2,3,1)
plt.imshow(f[0,:,:])
plt.colorbar()
plt.subplot(2,3,2)
plt.imshow(frec[0,:,:])
plt.colorbar()
plt.subplot(2,3,3)
plt.imshow(frec[0,:,:]-f[0,:,:])
plt.colorbar()
plt.subplot(2,3,4)
plt.imshow(Rrec[0,:,:])
plt.colorbar()
plt.subplot(2,3,5)
plt.imshow(Rf[0,:,:])
plt.colorbar()
plt.subplot(2,3,6)
plt.imshow(Rrec[0,:,:]-Rf[0,:,:])
plt.colorbar()

plt.show()
