#include "lpRgpu.cuh"
#include "main_kernels.cuh"
#include "simple_kernels.cuh"
//init global parameters
lpRgpu::lpRgpu(char* params)
{  
	readGlobalParameters(params);

	err=cudaMalloc((void **)&drho, Ntheta*Nrho*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dfl, Nslices*Ntheta*Nrho*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dflc, Nslices*Ntheta_R2C*Nrho*sizeof(float2));if (err!=0) callErr(cudaGetErrorString(err));

	//init rho space
	err=cudaMemcpy(drho,rho,Ntheta*Nrho*sizeof(float),cudaMemcpyHostToDevice);if (err!=0) callErr(cudaGetErrorString(err));

	cudaChannelFormatDesc texf_desc = cudaCreateChannelDesc<float>();	
	cudaExtent volumeSize = make_cudaExtent(Ntheta,Nrho,Nslices); 
	
	err=cudaMalloc3DArray(&dfla, &texf_desc,volumeSize,cudaArrayLayered);if (err!=0) callErr(cudaGetErrorString(err));
	texfl.addressMode[0] = cudaAddressModeWrap;
	texfl.addressMode[1] = cudaAddressModeWrap;
	texfl.filterMode = cudaFilterModeLinear;
	texfl.normalized =true;
	cudaBindTextureToArray(texfl, dfla,texf_desc);  

	//fft plans for Nslices slices
	cufftResult res1,res2;
	int ffts[]={Nrho,Ntheta};
	int idist = Nrho*Ntheta;int odist = Nrho*(Ntheta/2+1);
	int inembed[] = {Nrho, Ntheta};int onembed[] = {Nrho, Ntheta/2+1};
	res1=cufftPlanMany(&plan_forward, 2, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_R2C, Nslices);if (res1!=0) {char errs[16];sprintf(errs,"fwd cufftPlanMany error %d",res1);callErr(errs);}
	res2=cufftPlanMany(&plan_inverse, 2, ffts, onembed, 1, odist, inembed, 1, idist, CUFFT_C2R, Nslices);if (res2!=0) {char errs[16];sprintf(errs,"inv cufftPlanMany error %d",res1);callErr(errs);}

	err=cudaMalloc((void **)&dR, Nslices*Nproj*Ns*sizeof(float));	if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&df, Nslices*N*N*sizeof(float));	if (err!=0) callErr(cudaGetErrorString(err));

	fwd_init=0;adj_init=0;
}
lpRgpu::~lpRgpu()
{
	delete[] rho;
	//free gpu memory
	cudaFree(drho);
	cudaFree(dfl);
	cudaFree(dflc);
	cudaUnbindTexture(texfl);
	cudaFreeArray(dfla);	
	cufftDestroy(plan_forward);
	cufftDestroy(plan_inverse);
	cudaFree(df);
	cudaFree(dR);

	//delete parameters for fwd and adj transform if they are initialized
	deleteFwd();
	deleteAdj();
}

//init parameters for forward (Radon) tranform 
void lpRgpu::initFwd(char* params)
{
	//delete if exist
	deleteFwd();

	fgs=new fwdgrids(Nspan);
	readFwdParameters(params);
	fgs->initgpu();

	err=cudaMalloc((void **)&dfZfwd, Ntheta_R2C*Nrho*sizeof(float2));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dtmpf, Nslices*N*N*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));//delete to do

	//copy Fourier transform of Z	
	err=cudaMemcpy(dfZfwd,fZfwd,Ntheta_R2C*Nrho*sizeof(float2),cudaMemcpyHostToDevice);if (err!=0) callErr(cudaGetErrorString(err));

	cudaChannelFormatDesc texf_desc = cudaCreateChannelDesc<float>();
	cudaExtent volumeSize = make_cudaExtent(N,N,Nslices); 
	err=cudaMalloc3DArray(&dfa, &texf_desc, volumeSize,cudaArrayLayered);if (err!=0) callErr(cudaGetErrorString(err));
	texf.addressMode[0] = cudaAddressModeWrap;
	texf.addressMode[1] = cudaAddressModeWrap;	
	texf.filterMode = cudaFilterModeLinear;
	texf.normalized = true;
	cudaBindTextureToArray(texf, dfa,texf_desc);

	//init result with zeros
	err=cudaMemset(dR, 0, Nslices*Nproj*Ns*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));

	fwd_init=1;
}
void lpRgpu::deleteFwd()
{
	if (!fwd_init) return;

	delete[] fZfwd;
	cudaFree(dtmpf);
	cudaUnbindTexture(texf);
	cudaFreeArray(dfa);
	cudaFree(dfZfwd);
	delete fgs;
	err=cudaGetLastError();if(err!=0) callErr(cudaGetErrorString(err));

	fwd_init=0;
}

//init parameters for adjoint tranform (back-projection)
void lpRgpu::initAdj(char* params)
{
	//delete if exist
	deleteAdj();

	ags=new adjgrids(Nspan);
	readAdjParameters(params);
	ags->initgpu();

	err=cudaMalloc((void **)&dfZadj, Ntheta_R2C*Nrho*sizeof(float2));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMalloc((void **)&dtmpR, Nslices*Ns*Nproj*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));//delete to do

	//copy Fourier transform of adj Z
	cudaMemcpy(dfZadj,fZadj,Ntheta_R2C*Nrho*sizeof(float2),cudaMemcpyHostToDevice);

	cudaChannelFormatDesc texf_desc = cudaCreateChannelDesc<float>();	
	cudaExtent volumeSize = make_cudaExtent(Nproj,Ns,Nslices); 
	err=cudaMalloc3DArray(&dRa, &texf_desc, volumeSize,cudaArrayLayered); if (err!=0) callErr(cudaGetErrorString(err));
	texR.addressMode[0] = cudaAddressModeWrap;
	texR.addressMode[1] = cudaAddressModeWrap;
	texR.filterMode = cudaFilterModeLinear;
	texR.normalized = true;
	cudaBindTextureToArray(texR, dRa,texf_desc);

	//init result with zeros
	err=cudaMemset(df, 0, Nslices*N*N*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));

	//init filter
	if (filter)
	{
		int osfilter=4;
		err=cudaMalloc((void **)&dfilter, Ns*osfilter*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
		err=cudaMemcpy(dfilter, filter,Ns*osfilter*sizeof(float),cudaMemcpyDefault);if (err!=0) callErr(cudaGetErrorString(err));
	
		cufftPlan1d(&plan_f_forward,Ns*osfilter,CUFFT_C2C,Nproj);
		cufftPlan1d(&plan_f_inverse,Ns*osfilter,CUFFT_C2C,Nproj);
		err=cudaMalloc((void **)&dRt, Nproj*Ns*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
		err=cudaMalloc((void **)&dRc, Nproj*Ns*osfilter*sizeof(float2));if (err!=0) callErr(cudaGetErrorString(err));
	}
	adj_init=1;
}
void lpRgpu::deleteAdj()
{
	if (!adj_init) return;

	delete[] fZadj;
	cudaFree(dtmpR);
	cudaFree(dfZadj);
	cudaUnbindTexture(texR);
	cudaFreeArray(dRa);
	delete ags;

	if(filter)
	{
		delete[] filter;
		cudaFree(dfilter);
		cudaFree(dRt);
		cudaFree(dRc);
		cufftDestroy(plan_f_forward);
		cufftDestroy(plan_f_inverse);
	}	
	
	adj_init=0;
}

cudaError_t copy3DDeviceToArray(cudaArray* dfa, float* df, cudaExtent ext)
{
	cudaMemcpy3DParms param = { 0 };
	param.srcPtr   = make_cudaPitchedPtr((void*)df, ext.width*sizeof(float), ext.width, ext.height);
	param.dstArray = dfa;
	param.kind = cudaMemcpyDeviceToDevice;
	param.extent = ext;
	return cudaMemcpy3D(&param);
}

cudaError_t copy3Dshifted(float *dst, int dstx,int dsty, cudaExtent dstext, float* src, int srcx, int srcy, cudaExtent srcext, cudaExtent copyext)
{
	cudaMemcpy3DParms param = { 0 };
	param.srcPtr = make_cudaPitchedPtr(&src[srcy*srcext.width+srcx], srcext.width*sizeof(float), srcext.width, srcext.height);
	param.dstPtr = make_cudaPitchedPtr(&dst[dsty*dstext.width+dstx], dstext.width*sizeof(float), dstext.width, dstext.height);
	param.kind = cudaMemcpyDefault;
	copyext.width*=sizeof(float);
	param.extent = copyext;
	return cudaMemcpy3D(&param);
}


//compute Radon transform for several slices
void lpRgpu::execFwdMany(float* R, int Nslices2_, int Ns_, int Nproj_, float* f, int Nslices1_, int N2_, int N1_)
{
	cudaMemset(df,0,N*N*Nslices*sizeof(float));
        err=copy3Dshifted(df,N/2-N1_/2,N/2-N2_/2,make_cudaExtent(N,N,Nslices),f,0,0,make_cudaExtent(N1_, N2_, Nslices1_),make_cudaExtent(N1_,N2_,Nslices1_)); if(err!=0) callErr(cudaGetErrorString(err));  	    execFwd();
        err=copy3Dshifted(R,0,0,make_cudaExtent(Nproj_,Ns_,Nslices2_),dR,0,Ns/2-Ns_/2,make_cudaExtent(Nproj, Ns, Nslices),make_cudaExtent(Nproj_,Ns_,Nslices2_)); if(err!=0) callErr(cudaGetErrorString(err));
}

//compute back-projection for several slices
void lpRgpu::execAdjMany(float* f, int Nslices1_, int N2_, int N1_, float* R, int Nslices2_, int Ns_, int Nproj_, int cor)
{
	cudaMemset(dR,0,Nproj*Ns*Nslices*sizeof(float));
	int shift=cor-Ns_/2;
        err=copy3Dshifted(dR,0,Ns/2-Ns_/2+shift,make_cudaExtent(Nproj, Ns, Nslices),R,0,0,make_cudaExtent(Nproj_,Ns_,Nslices2_),make_cudaExtent(Nproj_,Ns_,Nslices2_)); if(err!=0) callErr(cudaGetErrorString(err));   	   
	padding(Ns_,shift);
	applyFilter();
	execAdj();
        err=copy3Dshifted(f,0,0,make_cudaExtent(N1_, N2_, Nslices1_),df,N/2-N1_/2,N/2-N2_/2,make_cudaExtent(N,N,Nslices),make_cudaExtent(N1_,N2_,Nslices1_)); if(err!=0) callErr(cudaGetErrorString(err));  }

//padding
void lpRgpu::padding(int Ns_, int shift)
{
	uint GS31=(uint)ceil(Nproj/(float)MBS21);uint GS32=(uint)ceil(Ns/(float)MBS22);uint GS33=(uint)ceil(Nslices/(float)MBS33);
        dim3 dimBlock(MBS31,MBS32,MBS33);dim3 dimGrid(GS31,GS32,GS33);
        padker<<<dimGrid,dimBlock>>>(dR,Ns/2-Ns_/2+shift,Ns/2+Ns_/2+shift-1,Nproj,Ns,Nslices);
}

//prefilter to compensate amplitudes in cubic interpolation
void lpRgpu::prefilter2D(float *df, float* dtmpf, uint width, uint height)
{
	//transpose for optimal cache usage
	uint GS31=(uint)ceil(width/(float)MBS31);uint GS32=(uint)ceil(height/(float)MBS32);uint GS33=(uint)ceil(Nslices/(float)MBS33);
	dim3 dimBlock(MBS31,MBS32,MBS33);dim3 dimGrid(GS31,GS32,GS33);
	transpose<<<dimGrid,dimBlock>>>(dtmpf, df,width, height,Nslices);

	//compensate in samples for x direction
	uint GS41=(uint)ceil(height/(float)MBS41);uint GS42=(uint)ceil(Nslices/(float)MBS42); 
	dim3 dimBlock1(MBS41,MBS42);dim3 dimGrid1(GS41,GS42);
	SamplesToCoefficients2DY<<<dimGrid1, dimBlock1>>>(dtmpf,height*sizeof(float),height, width,Nslices);

	//transpose back
	GS31=(uint)ceil(height/(float)MBS31);GS32=(uint)ceil(width/(float)MBS32);GS33=(uint)ceil(Nslices/(float)MBS33);
	dim3 dimBlock2(MBS31,MBS32,MBS33);dim3 dimGrid2(GS31,GS32,GS33);
	transpose<<<dimGrid2,dimBlock2>>>(df,dtmpf,height, width,Nslices);

	//compensate in samples for y direction
	GS41=(uint)ceil(width/(float)MBS41);GS42=(uint)ceil(Nslices/(float)MBS42); 
	dim3 dimBlock3(MBS41,MBS42);dim3 dimGrid3(GS41,GS42);	
	SamplesToCoefficients2DY<<<dimGrid3, dimBlock3>>>(df,width*sizeof(float),width,height,Nslices);
}

//compute Radon transform in log-polar coordinates
void lpRgpu::execFwd()
{
	err=cudaMemset(dtmpf, 0, Nslices*N*N*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
	err=cudaMemset(dR, 0, Nslices*Nproj*Ns*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err));
	//compensation for cubic interpolation
	prefilter2D(df,dtmpf,N,N);

	//init gpu array with binded texture
	copy3DDeviceToArray(dfa,df,make_cudaExtent(N, N, Nslices));

	//CUDA block and grid sizes
	dim3 dimBlock(MBS31,MBS32,MBS33);
	uint GS31, GS32, GS33;

	for(int k=0;k<Nspan;k++)
	{
		err=cudaMemset(dfl, 0, Nslices*Ntheta*Nrho*sizeof(float));if (err!=0) callErr(cudaGetErrorString(err)); 

		//interp Cartesian to log-polar grid
		GS31=(uint)ceil(ceil(sqrtf((float)fgs->Ncidsfwd))/(float)MBS31);GS32=(uint)ceil(ceil(sqrtf((float)fgs->Ncidsfwd))/(float)MBS32);GS33=(uint)ceil(Nslices/(float)MBS33);dim3 dimGrid(GS31,GS32,GS33);
		interp<<<dimGrid, dimBlock>>>(0,dfl,fgs->dlp2C1[k],fgs->dlp2C2[k],MBS31*GS31,fgs->Ncidsfwd,N,N,Nslices,fgs->dcidsfwd,Ntheta*Nrho);
		
		//multiplication e^{\rho}
		GS31=(uint)ceil(Ntheta/(float)MBS31);GS32=(uint)ceil(Nrho/(float)MBS32);GS33=(uint)ceil(Nslices/(float)MBS33);dim3 dimGrid1(GS31,GS32,GS33);
		mulexp<<<dimGrid1, dimBlock>>>(dfl,drho,Ntheta,Nrho, Nslices);

		//forward FFT
		cufftExecR2C(plan_forward, (cufftReal*)dfl,(cufftComplex*)dflc);

		//multiplication by fZ
		GS31=(uint)ceil(Ntheta_R2C/(float)MBS31);GS32=(uint)ceil(Nrho/(float)MBS32);GS33=(uint)ceil(Nslices/(float)MBS33);dim3 dimGrid2(GS31,GS32,GS33);
		mul<<<dimGrid2, dimBlock>>>(1/(float)(Ntheta*Nrho),dflc,dfZfwd,Ntheta_R2C,Nrho,Nslices);

		//inverse FFT
		cufftExecC2R(plan_inverse,(cufftComplex*)dflc,(cufftReal*)dfl);

		//init gpu array with binded texture
		copy3DDeviceToArray(dfla,dfl,make_cudaExtent(Ntheta, Nrho, Nslices));

		//interp log-polar to polar grid
		GS31=(uint)ceil(ceil(sqrtf((float)fgs->Npids[k]))/(float)MBS31);GS32=(uint)ceil(ceil(sqrtf((float)fgs->Npids[k]))/(float)MBS32);GS33=(uint)ceil(Nslices/(float)MBS33);dim3 dimGrid3(GS31,GS32,GS33);
		interp<<<dimGrid3, dimBlock>>>(2,dR,fgs->dp2lp1[k],fgs->dp2lp2[k],MBS31*GS31,fgs->Npids[k],Ntheta,Nrho,Nslices,fgs->dpids[k],Nproj*Ns);
	}
}

//compute back-projection in log-polar coordinates
void lpRgpu::execAdj()
{
	cudaMemset(dtmpR, 0, Nslices*Nproj*Ns*sizeof(float)); 
	cudaMemset(df, 0, Nslices*N*N*sizeof(float)); 
	//compensation for cubic interpolation
	prefilter2D(dR,dtmpR,Nproj,Ns);
	//init gpu array with binded texture
	copy3DDeviceToArray(dRa,dR,make_cudaExtent(Nproj, Ns, Nslices));

	//CUDA block and grid sizes
	dim3 dimBlock(MBS31,MBS32,MBS33);
	uint GS31, GS32, GS33;
	for(int k=0;k<Nspan;k++)
	{   
		cudaMemset(dfl, 0, Nslices*Ntheta*Nrho*sizeof(float)); 
		//interp from polar to log-polar grid
		GS31=(uint)ceil(ceil(sqrt(ags->Nlpidsadj))/(float)MBS31); GS32=(uint)ceil(ceil(sqrt(ags->Nlpidsadj))/(float)MBS32);GS33=(uint)ceil(Nslices/(float)MBS33);dim3 dimGrid(GS31,GS32,GS33);
		interp<<<dimGrid, dimBlock>>>(1,dfl,ags->dlp2p1[k],ags->dlp2p2[k],MBS31*GS31,ags->Nlpidsadj,Nproj,Ns,Nslices,ags->dlpidsadj,Ntheta*Nrho);

		//interp from polar to log-polar grid additional points
		GS31=(uint)ceil(ceil(sqrt(ags->Nwids))/(float)MBS31); GS32=(uint)ceil(ceil(sqrt(ags->Nwids))/(float)MBS32);GS33=(uint)ceil(Nslices/(float)MBS33);dim3 dimGrid4(GS31,GS32,GS33);
		interp<<<dimGrid4, dimBlock>>>(1,dfl,ags->dlp2p1w[k],ags->dlp2p2w[k],MBS31*GS31,ags->Nwids,Nproj,Ns,Nslices,ags->dwids,Ntheta*Nrho);

		//Forward FFT
		cufftExecR2C(plan_forward, (cufftReal*)dfl,(cufftComplex*)dflc);

		//multiplication by adjoint fZ
		GS31=(uint)ceil(Ntheta_R2C/(float)MBS31); GS32=(uint)ceil(Nrho/(float)MBS32);GS33=(uint)ceil(Nslices/(float)MBS33);dim3 dimGrid2(GS31,GS32,GS33);
		mul<<<dimGrid2, dimBlock>>>(1/(float)(Ntheta*Nrho),dflc,dfZadj,Ntheta_R2C,Nrho,Nslices);

		//Inverse FFT
		cufftExecC2R(plan_inverse,(cufftComplex*)dflc,(cufftReal*)dfl);

		//init gpu array with binded texture
		copy3DDeviceToArray(dfla,dfl,make_cudaExtent(Ntheta, Nrho, Nslices));

		//interp from log-polar to Cartesian grid
		GS31=(uint)ceil(ceil(sqrt(ags->Ncidsadj))/(float)MBS31); GS32=(uint)ceil(ceil(sqrt(ags->Ncidsadj))/(float)MBS32);GS33=(uint)ceil(Nslices/(float)MBS33);dim3 dimGrid3(GS31,GS32,GS33);
		interp<<<dimGrid3, dimBlock>>>(2,df,ags->dC2lp1[k],ags->dC2lp2[k],MBS31*GS31,ags->Ncidsadj,Ntheta,Nrho,Nslices,ags->dcidsadj,N*N);
	}
}

//apply filter in frequency
void lpRgpu::applyFilter()
{
	if (!filter) return;
	
	dim3 dimBlock(MBS21,MBS22);
	uint GS21, GS22;
	int osfilter=4;
	for(int ij=0;ij<Nslices;ij++)
	{
		cudaMemset(dRc, 0, 2*Nproj*Ns*osfilter*sizeof(float));
		//transpose data
		GS21=ceil(Nproj/(float)MBS21);GS22=ceil(Ns/(float)MBS22);dim3 dimGrid(GS21,GS22);
		transpose<<<dimGrid, dimBlock>>>(dRt, &dR[Nproj*Ns*ij], Nproj, Ns,1);
		
		//copy to complex array
		GS21=ceil(Ns/(float)MBS21);GS22=ceil(Nproj/(float)MBS22);dim3 dimGrid1(GS21,GS22);	
		copyc<<<dimGrid1, dimBlock>>>(dRt,dRc,Ns,Nproj,osfilter);

		//fftshift 
		GS21=ceil(Ns*osfilter/(float)MBS21);GS22=ceil(Nproj/(float)MBS22);dim3 dimGrid2(GS21,GS22);
		fftshift<<<dimGrid2, dimBlock>>>(dRc,Ns*osfilter,Nproj);

		//forward fft	
		cufftExecC2C(plan_f_forward,dRc,dRc,CUFFT_FORWARD);
	
		//fftshift
		GS21=ceil(Ns*osfilter/(float)MBS21);GS22=ceil(Nproj/(float)MBS22);dim3 dimGrid3(GS21,GS22);
		fftshift<<<dimGrid3, dimBlock>>>(dRc,Ns*osfilter,Nproj);

		//mulfilter
		GS21=ceil(Ns*osfilter/(float)MBS21);GS22=ceil(Nproj/(float)MBS22);dim3 dimGrid4(GS21,GS22);
		mulfilter<<<dimGrid4, dimBlock>>>(dRc,dfilter,Ns*osfilter,Nproj);

		//fftshift
		GS21=ceil(Ns*osfilter/(float)MBS21);GS22=ceil(Nproj/(float)MBS22);dim3 dimGrid5(GS21,GS22);
		fftshift<<<dimGrid5, dimBlock>>>(dRc,Ns*osfilter,Nproj);
	
		//adjoint fft	
		cufftExecC2C(plan_f_inverse,dRc,dRc,CUFFT_INVERSE);
	
		//fftshift
		GS21=ceil(Ns*osfilter/(float)MBS21);GS22=ceil(Nproj/(float)MBS22);dim3 dimGrid6(GS21,GS22);
		fftshift<<<dimGrid6, dimBlock>>>(dRc,Ns*osfilter,Nproj);

		//copy from complex array
		GS21=ceil(Ns*osfilter/(float)MBS21);GS22=ceil(Nproj/(float)MBS22);dim3 dimGrid7(GS21,GS22);	
		copycback<<<dimGrid7, dimBlock>>>(dRt,dRc,Ns,Nproj,osfilter);

		//tranpose back
		GS21=ceil(Ns/(float)MBS21);GS22=ceil(Nproj/(float)MBS22);dim3 dimGrid8(GS21,GS22);
		transpose<<<dimGrid8, dimBlock>>>(&dR[Nproj*Ns*ij], dRt, Ns,Nproj,1);

		//mul const
		GS21=ceil(Nproj/(float)MBS21);GS22=ceil(Ns/(float)MBS22);dim3 dimGrid9(GS21,GS22);
		mulconst<<<dimGrid9, dimBlock>>>(&dR[Nproj*Ns*ij],1/(float)(osfilter*Ns), Nproj, Ns);
	}
}

