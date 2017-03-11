#include <cufft.h>
#include <stdio.h>
#include "config.cuh"
#include "gridStorage.cuh"

class lpRgpu{
	//global parameters
	int N;
	int Nspan;
	int Ntheta;int Nrho;
	int Ns;int Nproj;
	int Nslices;
	int add;
	int Ntheta_R2C;
	float* rho;

	//grids storages
	fwdgrids* fgs;
	adjgrids* ags;
	//gpu memory    
	float* drho;
	float* dfl;
	float2* dflc;
	cudaArray* dfla;
	cufftHandle plan_forward;
	cufftHandle plan_inverse;
	cufftHandle plan_f_forward;
	cufftHandle plan_f_inverse;
	//fwd
	float2* fZfwd;
	float2* dfZfwd;
	cudaArray* dfa;
	float* dR;
	float* dtmpf;

	//adj
	float2* fZadj;
	float2* dfZadj;
	float* dtmpR;
	cudaArray* dRa;
	float* df;

	//filter
	float* filter;
	float* dfilter;
	float* dRt;float2* dRc;
	
	cudaError_t err;
	
	bool fwd_init,adj_init;
	
public:
	lpRgpu(char* file_glparams);
	~lpRgpu();
	void printGlobalParameters();
	void printFwdParameters();
	void printAdjParameters();
	void readGlobalParameters(char* file_params);
	void readFwdParameters(char* file_params);
	void readAdjParameters(char* file_params);
	void deleteGlobalParameters();
	void printCurrentGPUMemory(const char* str = 0);
	void initFwd(char* params);
	void initAdj(char* params);

	void deleteFwd();
	void deleteAdj();

	void prefilter2D(float *df, float* dtmpf,uint width, uint height);
	void execFwd();
	void execAdj();
	void execFwdMany(float* R, int Nslices2_, int Ns_, int Nproj_, float* f, int Nslices1_, int N2_, int N1_);
	void execAdjMany(float* f, int Nslices1_, int N2_, int N1_, float* R, int Nslices2_, int Ns_, int Nproj_, int cor);
	void applyFilter();
	void padding(int Ns_, int shift);
};
