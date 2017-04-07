from numpy import *
from lpRgpu import lpRgpu

class Pgl:
	def __init__(self, Nspan,N,proj,s,thsp,rhosp,aR,beta,add,B3com,am,g,Nslices):
		self.Nspan=Nspan;
		self.N=N;
		self.Ns=size(s);
		self.Nproj=size(proj);
		self.Ntheta=size(thsp);
		self.Nrho=size(rhosp);

		self.proj=proj;
		self.s=s;
		self.thsp=thsp;
		self.rhosp=rhosp;

		self.aR=aR;
		self.beta=beta;
		self.add=add;
		self.B3com=B3com;
		self.am=am;
		self.g=g;
		self.Nslices=Nslices;	

def create_gl(N,Nproj,Ns,Nslices,Pgl_file,Ntheta,Nrho):
	"initialize global parameters for the log-polar-based method and save to file"
	Nspan=3;
	th=linspace(0,pi,Nproj+1);th=th[0:-1];
	s=linspace(-1,1,Ns);add=0;
	Pgl=precompute_gl(N,th,s,add,1,Nslices,Nspan,Ntheta,Nrho);
	savePglparams(Pgl,Pgl_file);
	print ("Global parameters for the log-polar RT were saved to the file %s\n" % Pgl_file);
	return Pgl;

def precompute_gl(N,proj,s,add,radius,Nslices,Nspan,Ntheta,Nrho):

	beta=pi/Nspan;
	#log-polar space
	(Nrho,Ntheta,dtheta,drho,aR,am,g)=getparameters(beta,proj[1]-proj[0],s[1]-s[0],N,size(proj),size(s),Ntheta,Nrho);
	thsp=arange(-Ntheta/2,Ntheta/2)*dtheta;
	rhosp=arange(-Nrho,0)*drho;
	proj=proj-beta/2;

	#compensation for cubic interpolation
	B3th=splineB3(thsp,radius);B3th=fft.fft(fft.ifftshift(B3th));
	B3rho=splineB3(rhosp,radius);B3rho=(fft.fft(fft.ifftshift(B3rho)));
	B3com=array(transpose(matrix(B3rho))*B3th);

	Pgl0=Pgl(Nspan,N,proj,s,thsp,rhosp,aR,beta,add,B3com,am,g,Nslices);
	return Pgl0;

def getparameters(beta,dtheta,ds,N,Nproj,Ns,Ntheta,Nrho):
	aR=sin(beta/2)/(1+sin(beta/2));
	am=(cos(beta/2)-sin(beta/2))/(1+sin(beta/2));
	g=osg(aR,beta/2);#wrapping
	if(Ntheta==0):
		#recommendation
		Ntheta=N;
	if (Nrho==0):
		#recommendation
		Nrho=2*N;

	if mod(Ntheta,2)!=0: 
		Ntheta=Ntheta+1;
	dtheta=(2*beta)/Ntheta;
	drho=(g-log(am))/Nrho;
	return (Nrho,Ntheta,dtheta,drho,aR,am,g);

def osg(aR,theta):
	t=linspace(-pi/2,pi/2,100000);
	w=aR*cos(t)+(1-aR)+1j*aR*sin(t);
	g=max(log(abs(w))+log(cos(theta-arctan2(imag(w),real(w)))));	
	return g;

def splineB3(x2,r):
	sizex=size(x2);
	x2=x2-(x2[-1]+x2[0])/2;
	stepx=x2[1]-x2[0];
	ri=int32(ceil(2*r)); 

	r=r*stepx;
	x2c=x2[int32(ceil((sizex+1)/2.0))-1];
	x=x2[range(int32(ceil((sizex+1)/2.0)-ri-1),int32(ceil((sizex+1)/2.0)+ri))];
	d=abs(x-x2c)/r;
	B3=x*0;
	for ix in range(-ri,ri+1):
		id=ix+ri;
		if d[id]<1: #use the first polynomial  
			B3[id]=(3*d[id]**3-6*d[id]**2+4)/6;  
		else:
			if(d[id]<2):
				B3[id]=(-d[id]**3+6*d[id]**2-12*d[id]+8)/6;
	B3f=x2*0;
	B3f[range(int32(ceil((sizex+1)/2.0)-ri-1),int32(ceil((sizex+1)/2.0)+ri))]=B3;
	return B3f;

def savePglparams(P,P_file):
	fid = open(P_file, 'wb');
	fid.write(int32(P.N));
	fid.write(int32(P.Ntheta));
	fid.write(int32(P.Nrho));
	fid.write(int32(P.Nspan));
	fid.write(int32(P.Nproj));
	fid.write(int32(P.Ns));
	fid.write(int32(P.add));
	rho=transpose(tile(P.rhosp,[size(P.thsp),1]));
	fid.write(float32(ravel(rho)));
	fid.write(int32(P.Nslices));
	fid.close();
