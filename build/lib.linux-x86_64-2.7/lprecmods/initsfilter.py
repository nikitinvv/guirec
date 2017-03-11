from numpy import *

def take_filter(Ns,filter):
	os=4;d=0.5;
	Nse=os*Ns;
	t=arange(0,Nse/2+1)/float32(Nse);

	if (filter=='ramp'):
	        wfa=Nse*0.5*wint(12,t);#.*(t/(2*d)<=1);%compute the weigths
	elif (filter=='shepp-logan'):
		wfa = Nse*0.5*wint(12,t)*sinc(t/(2*d))*(t/d<=2);
	elif (filter=='cosine'):
        	wfa = Nse*0.5*wint(12,t)*cos(pi*t/(2*d))*(t/d<=1);
	elif (filter=='cosine2'):
	        wfa = Nse*0.5*wint(12,t)*(cos(pi*t/(2*d)))**2*(t/d<=1); 
	elif (filter=='hamming'):
		wfa = Nse*0.5*wint(12,t)*(.54 + .46 * cos(pi*t/d))*(t/d<=1);
	elif (filter=='hann'):
	        wfa=Nse*0.5*wint(12,t)*(1+cos(pi*t/d)) / 2.0*(t/d<=1);
	wfa=wfa*(wfa>=0);
	wfamid=2*wfa[0];
	tmp=wfa;
	wfa=append(flipud(tmp[1:]),wfamid);
	wfa=append(wfa, tmp[1:]);
	wfa=wfa[0:-1];
	wfa=float32(wfa);
	return wfa;

def wint(n,t):
	N=size(t);
	s=linspace(1e-40,1,n);
	iv=linalg.inv(exp(transpose(matrix(arange(0,n)))*log(s)));#Inverse vandermonde matrix
	u=diff(multiply(exp(transpose(matrix(arange(1,n+2)))*log(s)),transpose(tile(1.0/arange(1,n+2),[n,1])))); #integration over short intervals

	W1=iv*u[range(1,n+1),:];#x*pn(x) term
	W2=iv*u[range(0,n),:];#const*pn(x) term

	p=1./concatenate([range(1,n), [(n-1)]*(N-2*(n-1)-1),range(n-1,0,-1)]);#Compensate for overlapping short intervals
	w=float32(array([0]*N));
	for j in range(0,N-n+1):
		W=((t[j+n-1]-t[j])**2)*W1+(t[j+n-1]-t[j])*t[j]*W2;#Change coordinates, and constant and linear parts
		
		for k in range(0,n-1):
			w[j+arange(0,n)]=w[j+arange(0,n)]+transpose(W[:,k])*p[j+k];#% Add to weights

	wn=w;
	wn[range(N-40,N)]=(w[N-40])/(N-40)*arange(N-40,N);
	w=wn;
	return w
