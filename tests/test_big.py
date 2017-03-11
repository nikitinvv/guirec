#!/usr/bin/python
import sys
sys.path.append("../build/lib.linux-x86_64-2.7/")
import lprecmods.lpTransform as lpTransform
import matplotlib.pyplot as plt
from numpy import *
import struct
from timing import *

N=2016
Ntheta=299
Nslices=8
filter_type='hamming'
interp_type='cubic'
cor=1008
pad=True

fa=float32(random.random([Nslices,N,N]))
Ra=float32(random.random([Nslices,N,Ntheta]))

clpthandle=lpTransform.lpTransform(N,Ntheta,Nslices,filter_type,pad)
#tic()
#for k in range(0,30):
#	Rf=clpthandle.fwd(fa)
#toc()
#for k in range(0,30):
clpthandle=lpTransform.lpTransform(N,Ntheta,8,filter_type,pad)
clpthandle.precompute()
clpthandle.initcmem()
tic()
frec=clpthandle.adj(Ra,cor);
toc()
