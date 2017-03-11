#!/usr/bin/python
import sys
sys.path.append("../build/lib.linux-x86_64-2.7/")
import lprecmods.lpTransform as lpTransform
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

clpthandle=lpTransform.lpTransform(N,Nproj,Nslices,filter_type,pad)
clpthandle.precompute(1024,3072)#choose Ntheta,Nrho
clpthandle.initcmem()

for k in range(0,3):
	tic()
	frec=clpthandle.adj(Ra,cor);
	toc()

#fid=open('frec5','wb');
#frec[0,:,:].tofile(fid);
#fid.close();

plt.subplot(1,2,1)
plt.imshow(R[0,:,:])
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(frec[0,:,:])
plt.colorbar()
plt.show()
