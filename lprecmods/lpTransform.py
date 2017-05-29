import initsgl 
import initsfwd 
import initsadj 
import lpRgpu
from numpy import *
class lpTransform:
	def __init__(self,N,Nproj,Nslices,filter_type,pad):
		self.N=N;self.Ns=N;self.Nslices=Nslices;self.filter_type=filter_type;	
		#size after zero padding in the angle direction (for nondense sampling rate)
		self.osangles=int(max(round(3.0*N/2.0/Nproj),1))
		self.Nproj=self.osangles*Nproj
		#size after zero padding in radial direction
		if (pad): 
			self.Npad=self.Nspad=3*N/2
		else: 
			self.Npad=self.Nspad=N

	def precompute(self,Ntheta=0,Nrho=0):		
		#precompute parameters for the lp method
		Pgl=initsgl.create_gl(self.Npad,self.Nproj,self.Nspad,self.Nslices,'Pgl',Ntheta,Nrho)
		initsfwd.create_fwd(Pgl,'Pfwd')
		initsadj.create_adj(Pgl,'Padj',self.filter_type)

	def initcmem(self):
		#init memory in C, read data from files
		self.clphandle=lpRgpu.lpRgpu('Pgl')
		self.clphandle.initFwd('Pfwd')
		self.clphandle.initAdj('Padj')	

	def precompute_adj(self,Ntheta=0,Nrho=0):		
		#precompute parameters for the lp method
		Pgl=initsgl.create_gl(self.Npad,self.Nproj,self.Nspad,self.Nslices,'Pgl',Ntheta,Nrho)
		initsadj.create_adj(Pgl,'Padj',self.filter_type)

	def initcmem_adj(self):
		#init memory in C, read data from files
		self.clphandle=lpRgpu.lpRgpu('Pgl')
		self.clphandle.initAdj('Padj')


	def fwd(self,f):
		Ros=zeros([f.shape[0],self.Ns,self.Nproj],dtype='float32')
		self.clphandle.execFwdMany(Ros,f)
		R=Ros[:,:,0::self.osangles]
		return R;


	def adj(self,R,cor):
		Ros=zeros([R.shape[0],self.Ns,self.Nproj],dtype='float32')
		Ros[:,:,0::self.osangles]=R*self.osangles
		f=zeros([R.shape[0],self.N,self.N],dtype='float32')
		self.clphandle.execAdjMany(f,Ros,cor)
		return f;


	



