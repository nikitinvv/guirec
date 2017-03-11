from numpy import *
import h5py
import sys
import libtiff

class fileParser:
	def __init__(self,h5name):
		self.h5fid=h5py.File(h5name,'r')

	def takePars(self):
		pars=self.h5fid.get('exchange/data').shape
		return pars
	
	def readh5Corrected(self,id):
		print 'read h5 + correction'
		data=(self.h5fid['exchange/data'][:,id,:]);
		if (size(id)==1):
			 data=resize(data,[size(data,0),1,size(data,1)]) #for consistency with 1 slice	
		data_dark=(self.h5fid['exchange/data_dark'][:,id,:])
		data_white=(self.h5fid['exchange/data_white'][:,id,:])
		
		dark=tile(mean(data_dark,0),[size(data,0),1,1])
		white=tile(mean(data_white,0),[size(data,0),1,1])
		data=(data-dark)/float32(white-dark)
		data=-log(data.clip(min=1e-31));#could be a problem
		return data
	
	def writeTiff16(self,f,ids,recfile):
		print 'write tiff 16'
		for k in range(0,size(ids)):
			name=recfile+str(ids[k])+'.tiff';	
			tif = libtiff.TIFF.open(name, mode='w')
			tif.write_image(uint16(f[k,:,:]*(2**16-1)));
	

