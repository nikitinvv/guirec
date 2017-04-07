from numpy import *
from scipy import signal

def upfirdn(s, h, p, q):
	t1=kron(s, r_[1, zeros(p-1)])
	t2=signal.convolve(h, t1)
	t3=t2[0::q]
	return t3

def mresample(x,p,q,N):
	bta=5
	fc = 1/2.0/float32(max(p,q));
	L = 2*N*max(p,q) + 1;
	ideal_filter=2*p*fc*sinc(2*fc*arange(-(L-1)/2,(L-1)/2+1))/float32(p);
	h = ideal_filter*kaiser(L,bta);
	h = p*h/sum(h);
	Lhalf = (L-1)/2;
	Lx = x.shape[2];
	# Need to delay output so that downsampling by q hits center tap of filter.
	nz = int(floor(q-mod(Lhalf,q)));
	h = r_[zeros(nz),h];
	Lhalf = Lhalf + nz;

	# Number of samples removed from beginning of output sequence 
	# to compensate for delay of linear phase filter:
	delay = int(floor(ceil(Lhalf)/float32(q)));

	# Need to zero-pad so output length is exactly ceil(Lx*p/q).
	nz1 = 0;
	while ceil( ((Lx-1)*float32(p)+size(h)+nz1)/float32(q) ) - delay < ceil(Lx*p/float32(q)):
	    nz1 = nz1+1;
	h = r_[h,zeros(nz1)];
	print h.shape
	Ly = int(ceil(Lx*p/float32(q)));#  % output length
	y=zeros([x.shape[0],x.shape[1],Ly],dtype='float32')
	#UPFIRDN
	for j in range(0,x.shape[0]):
		for k in range(0,x.shape[1]):
			y1 = upfirdn(x[j,k,:],h,p,q);
			y1=y1[delay:-1];
			y1=y1[0:Ly];
			y[j,k,:]=y1;
	return y
