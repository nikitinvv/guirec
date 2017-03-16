from numpy import *
import lpTransform 
import fileParser 
from timing import *
import matplotlib.pyplot as plt

class processing:
	def __init__(self,N,Nproj,filter_type,pad,fname):
		self.Nslices=8
		self.clpthandle=lpTransform.lpTransform(N,Nproj,self.Nslices,filter_type,pad)
		self.clpthandle.precompute_adj()
		self.clpthandle.initcmem_adj()

		self.fphandle=fileParser.fileParser(fname)
	
	def rec(self,idslice,center,amp,recfname):
		Ni=size(idslice);
		for k in range(0,int(ceil(Ni/float32(self.Nslices)))):
			print k
			print idslice[range(k*self.Nslices,min((k+1)*self.Nslices,Ni))]
			R=self.fphandle.readh5Corrected(idslice[range(k*self.Nslices,min((k+1)*self.Nslices,Ni))])
			R=float32(R[0:-1,:,:])
			R=R.swapaxes(0,1).swapaxes(1,2)
			f=self.clpthandle.adj(R,int(center))
			if (amp):
				f=f.clip(min=0,max=amp)/amp
			self.fphandle.writeTiff16(f,idslice[range(k*self.Nslices,min((k+1)*self.Nslices,Ni))],recfname)
	
	
	

	
