#!/usr/bin/python
import sys
sys.path.append("../build/lib.linux-x86_64-2.7/")
import lprecmods.lpTransform as lpTransform
import matplotlib.pyplot as plt
from numpy import *
import struct

N=512
Nproj=3*N/2
Nslices=8
filter_type='None'
pad=True
cor=N/2

fid = open('./data/f', 'rb')
f=float32(reshape(struct.unpack(N*N*'f',fid.read(N*N*4)),[1,N,N]))
fa=zeros([Nslices,N,N],dtype=float32);
for k in range(0,Nslices):
	fa[k,:,:]=f*(k+1);

fid = open('./data/R', 'rb')
R=float32(reshape(struct.unpack(Nproj*N*'f',fid.read(Nproj*N*4)),[1,N,Nproj]))
Ra=zeros([Nslices,N,Nproj],dtype=float32);
for k in range(0,Nslices):
	Ra[k,:,:]=R;


clpthandle=lpTransform.lpTransform(N,Nproj,Nslices,filter_type,pad)
clpthandle.precompute()
clpthandle.initcmem()

Rf=clpthandle.fwd(fa)
frec=clpthandle.adj(Ra,cor);
Rrec=clpthandle.fwd(frec)


#dot product test
sum1= sum(ndarray.flatten(Rrec)*ndarray.flatten(Ra))
sum2= sum(ndarray.flatten(frec)*ndarray.flatten(frec))
print linalg.norm(sum1-sum2)/linalg.norm(sum2)

plt.subplot(2,3,1)
plt.imshow(fa[-1,:,:])
plt.colorbar()
plt.subplot(2,3,2)
plt.imshow(frec[-1,:,:])
plt.colorbar()
plt.subplot(2,3,3)
plt.imshow(frec[-1,:,:]-fa[-1,:,:])
plt.colorbar()
plt.subplot(2,3,4)
plt.imshow(Rrec[-1,:,:])
plt.colorbar()
plt.subplot(2,3,5)
plt.imshow(Rf[-1,:,:])
plt.colorbar()
plt.subplot(2,3,6)
plt.imshow(Rrec[-1,:,:]-Rf[-1,:,:])
plt.colorbar()

plt.show()
