/*interface*/
%module lpRgpu

%{
#define SWIG_FILE_WITH_INIT
#include "lpRgpu.cuh"
%}

%include "numpy.i"

%init %{
import_array();
%}


class lpRgpu{
	//global parameters
	int N;
	int Nspan;
	int Ntheta;int Nrho;
	int Ns;int Nproj;
	int Nslices;int ni;
	int Ntheta_cut;
	int add;
	int Ntheta_R2C;
	float* rho;

	//grids storages
	glgrids* ggs;
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
	void printCurrentGPUMemory(const char* str = 0);
	void initFwd(char* params);
	void initAdj(char* params);

	void deleteFwd();
	void deleteAdj();

	void prefilter2D(float *df, float* dtmpf,uint width, uint height);
	void execFwd();
	void execAdj();

%apply (float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* R, int Nslices2_,int Ns_, int Nproj_)};
%apply (float* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* f, int Nslices1_, int N2_, int N1_)};
	void execFwdMany(float* R, int Nslices2_, int Ns_, int Nproj_, float* f, int Nslices1_, int N2_, int N1_);
%clear (float* R, int Nslices2_,int Ns_, int Nproj_);
%clear (float* f, int Nslices1_, int N2_, int N1_);

%apply (float* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* R, int Nslices2_,int Ns_, int Nproj_)};
%apply (float* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(float* f, int Nslices1_, int N2_, int N1_)};
	void execAdjMany(float* f, int Nslices1_, int N2_, int N1_, float* R, int Nslices2_, int Ns_, int Nproj_, int cor);
%clear (float* R, int Nslices2_,int Ns_, int Nproj_);
%clear (float* f, int Nslices1_, int N2_, int N1_);

	void applyFilter();
void padding(int Ns_, int shift);
};
