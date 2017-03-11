import initsgl 
import initsfwd 
import initsadj 
import lpRgpu
from numpy import *
class lpTransform:
	def __init__(self,N,Nproj,Nslices,filter_type,pad):
		self.N=N;self.Nproj=Nproj;self.Ns=N;self.Nslices=Nslices;self.filter_type=filter_type;	
		if (pad): 
			self.Npad=self.Nspad=3*N/2
		else: 
			self.Npad=self.Nspad=N

	def precompute(self,Ntheta=0,Nrho=0):		
		#precompute parameters for the lp method
		Pgl=initsgl.create_gl(self.Npad,self.Nproj,self.Nspad,self.Nslices,'Pgl',Ntheta,Nrho);
		initsfwd.create_fwd(Pgl,'Pfwd');
		initsadj.create_adj(Pgl,'Padj',self.filter_type);

	def initcmem(self):
		#init memory in C, read data from files
		self.clphandle=lpRgpu.lpRgpu('Pgl');	
		self.clphandle.initFwd('Pfwd');
		self.clphandle.initAdj('Padj');	

	def fwd(self,f):
		R=zeros([f.shape[0],self.Ns,self.Nproj],dtype='float32');
		self.clphandle.execFwdMany(R,f);
		return R;


	def adj(self,R,cor):
		f=zeros([R.shape[0],self.N,self.N],dtype='float32');
		self.clphandle.execAdjMany(f,R,cor);
		return f;


	



