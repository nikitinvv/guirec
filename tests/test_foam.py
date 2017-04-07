#!/usr/bin/python
import sys
sys.path.append("../build/lib.linux-x86_64-2.7/")
import lprecmods.lpTransform as lpTransform
import lprecmods.mresample as mresample

from lprecmods.timing import *
import matplotlib.pyplot as plt
from numpy import *
import struct

N=2016
Nproj=299
Nslices=4
filter_type='hamming'
pad=True
cor=N/2

fid = open('./data/Rfoam', 'rb')
R=float32(reshape(struct.unpack(Nproj*N*'f',fid.read(Nproj*N*4)),[1,N,Nproj]))
Ra=float32(zeros([Nslices,N,Nproj]));

for k in range(0,Nslices):
	Ra[k,:,:]=R[0,:,:];

p=4;q=1;
tic()
Ra=mresample.mresample(Ra,p,q,q/float32(p));
print Ra.shape
toc()
clpthandle=lpTransform.lpTransform(N,Nproj*p/q,Nslices,filter_type,pad)
clpthandle.precompute()
clpthandle.initcmem()

for k in range(0,3):
	tic()
	frec=clpthandle.adj(Ra,cor);
	toc()

fid=open('frec2','wb');
frec[1,:,:].tofile(fid);
fid.close();

plt.subplot(1,2,1)
plt.imshow(R[0,:,:])
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(frec[0,:,:])
plt.colorbar()
plt.show()
