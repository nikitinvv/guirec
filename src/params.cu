#include"lpRgpu.cuh"
#include<stdio.h>

void lpRgpu::printGlobalParameters()
{
	printf("N %d\n",N);
	printf("(Ntheta,Nrho) (%d,%d)\n",Ntheta,Nrho);
	printf("Nspan %d\n",Nspan);
	printf("Nslices %d\n",Nslices);
	printf("Nproj %d\n",Nproj);
	printf("Ns %d\n",Ns);
	printf("add %d\n",add);
	printf("Ntheta_R2C %d\n",Ntheta_R2C);
	printf("rho ");
	for(int i=0;i<3;i++) printf("%f ",rho[i]);
	printf("\n");
}
void lpRgpu::printFwdParameters()
{
	printf("lp2C1[0][0] %f\n",fgs->lp2C1[0][0]);
	printf("lp2C2[0][0] %f\n",fgs->lp2C2[0][0]);
	printf("p2lp1[0][0] %f\n",fgs->p2lp1[0][0]);
	printf("p2lp2[0][0] %f\n",fgs->p2lp2[0][0]);
	printf("cids[0] %d\n",fgs->cidsfwd[0]);
	printf("Ncidsfwd %d\n",fgs->Ncidsfwd);
	printf("fZ (%f,%f)\n",fZfwd[0].x,fZfwd[0].y);
	printf("\n");
}
void lpRgpu::printAdjParameters()
{
	printf("C2lp1[0][0] %f\n",ags->C2lp1[0][0]);
	printf("C2lp2[0][0] %f\n",ags->C2lp2[0][0]);
	printf("lp2p1[0][0] %f\n",ags->lp2p1[0][0]);
	printf("lp2p2[0][0] %f\n",ags->lp2p2[0][0]);
	printf("lpids[0][0] %d\n",ags->lpidsadj[0]);
	printf("Ncidsadj %d\n",ags->Ncidsadj);
	printf("Nlpidsadj %d\n",ags->Nlpidsadj);
	printf("fZadj (%f,%f)\n",fZadj[0].x,fZadj[0].y);
	printf("\n");
}
void lpRgpu::readGlobalParameters(char* file_params)
{
	FILE * fid=fopen(file_params,"rb");
	fread(&(N),sizeof(int),1,fid);
	fread(&(Ntheta),sizeof(int),1,fid);
	fread(&(Nrho),sizeof(int),1,fid);
	fread(&(Nspan),sizeof(int),1,fid);
	fread(&(Nproj),sizeof(int),1,fid);
	fread(&(Ns),sizeof(int),1,fid);
	fread(&(add),sizeof(int),1,fid);

	rho=new float[Nrho*Ntheta];
	fread(rho,sizeof(float),Nrho*Ntheta,fid);
	fread(&(Nslices),sizeof(int),1,fid);

	Ntheta_R2C=(int)(Ntheta/2.0)+1;
	fclose(fid);
}
void lpRgpu::readFwdParameters(char* file_params)
{
	FILE * fid=fopen(file_params,"rb");
	
	fgs->Npids=new int[Nspan];
	fread(fgs->Npids,sizeof(int),Nspan,fid);
	
	fgs->pids=new int*[Nspan];

	for(int k=0;k<Nspan;k++) 
	{
		fgs->pids[k]=new int[fgs->Npids[k]];
		fread(fgs->pids[k],sizeof(int),fgs->Npids[k],fid);
	}

	fgs->lp2C1=new float*[Nspan];
	fgs->lp2C2=new float*[Nspan];

	fgs->p2lp1=new float*[Nspan];
	fgs->p2lp2=new float*[Nspan];

	fread(&fgs->Ncidsfwd,sizeof(int),1,fid);
	fgs->cidsfwd=new int[fgs->Ncidsfwd];

	for(int i=0;i<Nspan;i++)
	{
		fgs->lp2C1[i]=new float[fgs->Ncidsfwd];
		fgs->lp2C2[i]=new float[fgs->Ncidsfwd];

	}
	for(int i=0;i<Nspan;i++)
	{
		fgs->p2lp1[i]=new float[fgs->Npids[i]];
		fgs->p2lp2[i]=new float[fgs->Npids[i]];
	}

	for(int i=0;i<Nspan;i++) fread(fgs->lp2C1[i],sizeof(float),fgs->Ncidsfwd,fid);
	for(int i=0;i<Nspan;i++) fread(fgs->lp2C2[i],sizeof(float),fgs->Ncidsfwd,fid);
	for(int i=0;i<Nspan;i++) fread(fgs->p2lp1[i],sizeof(float),fgs->Npids[i],fid);
	for(int i=0;i<Nspan;i++) fread(fgs->p2lp2[i],sizeof(float),fgs->Npids[i],fid);
	fread(fgs->cidsfwd,sizeof(int),fgs->Ncidsfwd,fid);

	
	fZfwd=new float2[Ntheta_R2C*Nrho];
	fread(fZfwd,sizeof(float2),Ntheta_R2C*Nrho,fid);
	fclose(fid);
}
void lpRgpu::readAdjParameters(char* file_params)
{

	FILE * fid=fopen(file_params,"rb");
	ags->C2lp1=new float*[Nspan];
	ags->C2lp2=new float*[Nspan];

	ags->lp2p1=new float*[Nspan];
	ags->lp2p2=new float*[Nspan];
	ags->lp2p1w=new float*[Nspan];
	ags->lp2p2w=new float*[Nspan];
	fread(&ags->Ncidsadj,sizeof(int),1,fid);
	fread(&ags->Nlpidsadj,sizeof(int),1,fid);
	fread(&ags->Nwids,sizeof(int),1,fid);

	for(int i=0;i<Nspan;i++)
	{
		ags->C2lp1[i]=new float[ags->Ncidsadj];
		ags->C2lp2[i]=new float[ags->Ncidsadj];
	}
	for(int i=0;i<Nspan;i++)
	{
		ags->lp2p1[i]=new float[ags->Nlpidsadj];
		ags->lp2p2[i]=new float[ags->Nlpidsadj];
	}
	for(int i=0;i<Nspan;i++)
	{
		ags->lp2p1w[i]=new float[ags->Nwids];
		ags->lp2p2w[i]=new float[ags->Nwids];
	}
	//
	for(int i=0;i<Nspan;i++) fread(ags->C2lp1[i],sizeof(float),ags->Ncidsadj,fid);
	for(int i=0;i<Nspan;i++) fread(ags->C2lp2[i],sizeof(float),ags->Ncidsadj,fid);
	for(int i=0;i<Nspan;i++) fread(ags->lp2p1[i],sizeof(float),ags->Nlpidsadj,fid);
	for(int i=0;i<Nspan;i++) fread(ags->lp2p2[i],sizeof(float),ags->Nlpidsadj,fid);
	for(int i=0;i<Nspan;i++) fread(ags->lp2p1w[i],sizeof(float),ags->Nwids,fid);
	for(int i=0;i<Nspan;i++) fread(ags->lp2p2w[i],sizeof(float),ags->Nwids,fid);

	ags->cidsadj=new int[ags->Ncidsadj];
	fread(ags->cidsadj,sizeof(int),ags->Ncidsadj,fid);
	ags->lpidsadj=new int[ags->Nlpidsadj];
	fread(ags->lpidsadj,sizeof(int),ags->Nlpidsadj,fid);
	ags->wids=new int[ags->Nwids];
	fread(ags->wids,sizeof(int),ags->Nwids,fid);
	fZadj=new float2[Ntheta_R2C*Nrho];
	fread(fZadj,sizeof(float2),Ntheta_R2C*Nrho,fid);
	filter=new float[4*Ns];
	int els=fread(filter,sizeof(float),4*Ns,fid);
	if (els!=4*Ns) 
	{
		delete[] filter;
		filter=NULL;
	}
}
void lpRgpu::printCurrentGPUMemory(const char* str)
{
	size_t gpufree1,gputotal;
	cudaMemGetInfo(&gpufree1,&gputotal);
	if(str!=NULL)
		printf("%s gpufree=%.0fM,gputotal=%.0fM\n",str,gpufree1/(float)(1024*1024),gputotal/(float)(1024*1024));
	else
		printf("gpufree=%.0fM,gputotal=%.0fM\n",gpufree1/(float)(1024*1024),gputotal/(float)(1024*1024));
}
