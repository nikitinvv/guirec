from numpy import *

class Pfwd:
	def __init__(self,fZgpu,lp2C1,lp2C2,p2lp1,p2lp2,cids,pids):
		self.fZgpu=fZgpu;
		self.lp2C1=lp2C1;
		self.lp2C2=lp2C2;
		self.p2lp1=p2lp1;
		self.p2lp2=p2lp2;
		self.cids=cids;
		self.pids=pids;

def create_fwd(Pgl,Pfwd_file):
	"initialize parameters for fwd transform and save to file"
	Pfwd=precompute_fwd(Pgl);
	savePfwdparams(Pfwd,Pfwd_file);
	print ("Parameters for the forward transform were saved to the file %s\n" % Pfwd_file);
	return Pfwd;

def precompute_fwd(P):
	#convolution function
	fZ=fft.fftshift(fzeta_loop_weights(P.Ntheta,P.Nrho,2*P.beta,P.g-log(P.am),0,4));

	#(lp2C1,lp2C2), transformed log-polar to Cartesian coordinates
	texprho=transpose(matrix(exp(P.rhosp)))
	lp2C1=[None]*P.Nspan; lp2C2=[None]*P.Nspan;
	for k in range(0,P.Nspan):
		lp2C1[k]=ndarray.flatten(array((texprho*cos(P.thsp)-(1-P.aR))*cos((k)*P.beta+P.beta/2)-texprho*sin(P.thsp)*sin((k)*P.beta+P.beta/2))/P.aR);
		lp2C2[k]=ndarray.flatten(array((texprho*cos(P.thsp)-(1-P.aR))*sin((k)*P.beta+P.beta/2)+texprho*sin(P.thsp)*cos((k)*P.beta+P.beta/2))/P.aR);
		cids=where((lp2C1[k]**2+lp2C2[k]**2)<=1);
		lp2C1[k]=lp2C1[k][cids];
		lp2C2[k]=lp2C2[k][cids];

	#pids, index in polar grids after splitting by spans
	pids=[None]*P.Nspan;
	[th0,s0]=meshgrid(P.proj,P.s);th0=ndarray.flatten(th0);s0=ndarray.flatten(s0);
	for k in range(0,P.Nspan):
		pids[k]=ndarray.flatten(array(where((th0>=k*P.beta-P.beta/2) & (th0<k*P.beta+P.beta/2))));

	#(p2lp1,p2lp2), transformed polar to log-polar coordinates
	p2lp1=[None]*P.Nspan;p2lp2=[None]*P.Nspan;
	for k in range(0,P.Nspan):
		th00=th0[pids[k]]-k*P.beta;s00=s0[pids[k]];
		p2lp1[k]=th00;
		p2lp2[k]=log(s00*P.aR+(1-P.aR)*cos(th00));

	#adapt for gpu interp 
	for k in range(0,P.Nspan):
		lp2C1[k]=(lp2C1[k]+1)/2*(P.N-1)
		lp2C2[k]=(lp2C2[k]+1)/2*(P.N-1)
		p2lp1[k]=(p2lp1[k]-P.thsp[0])/(P.thsp[-1]-P.thsp[0])*(P.Ntheta-1)
		p2lp2[k]=(p2lp2[k]-P.rhosp[0])/(P.rhosp[-1]-P.rhosp[0])*(P.Nrho-1)

	const=P.N*P.N/P.Nproj*pi/2/P.aR
	fZgpu=fZ[:,arange(0,P.Ntheta/2+1)]/(P.B3com[:,arange(0,P.Ntheta/2+1)])*const

	Pfwd0=Pfwd(fZgpu,lp2C1,lp2C2,p2lp1,p2lp2,cids,pids)
	return Pfwd0;

def fzeta_loop_weights(Ntheta,Nrho,betas,rhos,a,osthlarge):
	krho=arange(-Nrho/2,Nrho/2);
	Nthetalarge=osthlarge*Ntheta;
	thsplarge=arange(-Nthetalarge/2,Nthetalarge/2)/float32(Nthetalarge)*betas;
	fZ=array(zeros(shape=(Nrho,Nthetalarge)),dtype=complex);
	h=array(ones(Nthetalarge));
	# correcting=1+[-3 4 -1]/24;correcting(1)=2*(correcting(1)-0.5);
	#correcting=1+array([-23681,55688,-66109,57024,-31523,9976,-1375])/120960.0;correcting[0]=2*(correcting[0]-0.5);
	correcting=1+array([-216254335,679543284,-1412947389,2415881496,-3103579086,2939942400,-2023224114,984515304,-321455811,63253516,-5675265])/958003200.0;correcting[0]=2*(correcting[0]-0.5);
	h[0]=h[0]*(correcting[0]);
	for j in range(1,size(correcting)):
		h[j]=h[j]*correcting[j];
		h[-1-j+1]=h[-1-j+1]*(correcting[j]);
	for j in range(0,size(krho)):
		fcosa=pow(cos(thsplarge),(-2*pi*1j*krho[j]/rhos-1-a));
		fZ[j,:]=fft.fftshift(fft.fft(fft.fftshift(h*fcosa)));
	fZ=fZ[:,range(Nthetalarge/2-Ntheta/2,Nthetalarge/2+Ntheta/2)];
	fZ=fZ*(thsplarge[1]-thsplarge[0]);
	#put imag to 0 for the border
	fZ[0]=0;
	fZ[:,0]=0;
	return fZ;

def savePfwdparams(P,P_file):
	fid = open(P_file, 'wb');

	Nspan=shape(P.pids)[0]
	Npids=[None]*Nspan;
	for k in range(0,Nspan):
		Npids[k]=size(P.pids[k]);
	fid.write(int32(Npids));
	for k in range(0,Nspan):
	    fid.write(int32(P.pids[k]));

	Ncids=size(P.lp2C1[0]);
	fid.write(int32(Ncids));
	for k in range(0,Nspan): 
		fid.write(float32(P.lp2C1[k]));
	for k in range(0,Nspan): 
		fid.write(float32(P.lp2C2[k]));
	for k in range(0,Nspan): 
		fid.write(float32(P.p2lp1[k]));
	for k in range(0,Nspan): 
		fid.write(float32(P.p2lp2[k]));
	fid.write(int32(P.cids));	


	fZvec=ndarray.flatten(transpose(array([real(ndarray.flatten(P.fZgpu)),ndarray.flatten(imag(P.fZgpu))])));
	fid.write(float32(fZvec));
	fid.close();
