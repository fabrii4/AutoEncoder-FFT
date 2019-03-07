#include <ctime>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include<iomanip>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cufft.h>
#include <cuComplex.h>

using namespace std;


/////////////////////////////////////////////////////////////
//       GPU FUNCTIONS
/////////////////////////////////////////////////////////////


//shift fft spectrum to display zero frequency to the center
__global__ void shift_magnitude(cufftReal *mag, cufftReal *mag_s, int dD, int Nx, int Ny)
{
   int Ntot=dD*Nx*Ny;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   int d=(int)idx/(Nx*Ny); 
   int i=(int)(idx-d*Nx*Ny)/Ny;
   int j=(int)(idx-d*Nx*Ny-i*Ny);
   int ind;
   if (idx < Ntot)
   {
      if(i<Nx/2 && j<Ny/2) ind=d*Nx*Ny+(Nx/2+i)*Ny+(Ny/2+j);
      if(i<Nx/2 && j>=Ny/2) ind=d*Nx*Ny+(Nx/2+i)*Ny+(j-Ny/2);
      if(i>=Nx/2 && j<Ny/2) ind=d*Nx*Ny+(i-Nx/2)*Ny+(Ny/2+j);
      if(i>=Nx/2 && j>=Ny/2) ind=d*Nx*Ny+(i-Nx/2)*Ny+(j-Ny/2);
      mag_s[idx]=mag[ind];
   }
}

/////////////////////////////////////////////////////////////

//compute fft spectrum with no shift
__global__ void magnitude(cufftComplex *data, cufftReal *mag, int dD, int Nx, int Ny)
{
   int Ntot=dD*Nx*Ny;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   int d=(int)idx/(Nx*Ny);
   int i=(int)(idx-d*Nx*Ny)/Ny;
   int j=(int)(idx-d*Nx*Ny-i*Ny);
   int Nyr=Ny/2+1;
   int indl=d*Nx*Nyr+i*Nyr+j;
   int indr=d*Nx*Nyr+(Nx-1-i)*Nyr+(2*Nyr-1-j);
   if(idx<Ntot)
   {
      if(j<Nyr) mag[idx]=sqrt(cuCabsf(data[indl])/Ntot);
      else mag[idx]=sqrt(cuCabsf(data[indr])/Ntot);
   }
}

/////////////////////////////////////////////////////////////

//Pooling (crop fft spectrum around zero frequency)
__global__ void pool(cufftComplex *data, int dD, int Nx, int Ny, int scale)
{
   float l=1./((float)scale*2);
   int Nyr=Ny/2+1;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < dD*Nx*Nyr)
   {
      int d=(int)idx/(Nx*Nyr);
      int i=(int)(idx-d*Nx*Nyr)/Nyr;
      int j=(int)(idx-d*Nx*Nyr-i*Nyr);
      //if(!((i<l*Nx || i>(1.-l)*Nx) && (j<l*Nyr || j>(1.-l)*Nyr)))
      if(!((i<l*Nx || i>=(1.-l)*Nx) && (j<l*Ny || j>=(1.-l)*Ny)))
         data[idx].x=data[idx].y=0;
   }
}

/////////////////////////////////////////////////////////////

//resize according to spectral pooling (for even Nx Ny Nxs Nys)
__global__ void resize(cufftComplex *freq_d, cufftComplex *freqs_d, int dM, int Nx, int Ny, int Nxs, int Nys, float l)
{
   int Nyr=Ny/2+1;
   int Nyrs=Nys/2+1;
   int ind;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < dM*Nxs*Nyrs)
   {
      int d=(int)idx/(Nxs*Nyrs);
      int i=(int)(idx-d*Nxs*Nyrs)/Nyrs;
      int j=(int)(idx-d*Nxs*Nyrs-i*Nyrs);
      if(Nxs<=Nx)
      {
         if(j<Nyrs-1)
         {
            if(i<Nxs/2) ind=d*Nx*Nyr+i*Nyr+j;
            if(i==Nxs/2) ind=d*Nx*Nyr+Nx/2*Nyr+j;
            if(i>Nxs/2) ind=d*Nx*Nyr+(i+Nx-Nxs)*Nyr+j;
            freqs_d[idx]=freq_d[ind];
         }
         if(j==Nyrs-1)
         {
            if(i<Nxs/2) ind=d*Nx*Nyr+i*Nyr+Nyr-1;
            if(i==Nxs/2) ind=d*Nx*Nyr+Nx/2*Nyr+Nyr-1;
            if(i>Nxs/2) ind=d*Nx*Nyr+(i+Nx-Nxs)*Nyr+Nyr-1;
            freqs_d[idx]=freq_d[ind];
         }
      }
      else
      {
         if(j<Nyr-1)
         {
            if(i<Nx/2)
            {
               ind=d*Nx*Nyr+i*Nyr+j;
               freqs_d[idx]=freq_d[ind];
            }
            else if(i>Nxs-Nx/2) 
            {
               ind=d*Nx*Nyr+(i-Nxs+Nx)*Nyr+j;
               freqs_d[idx]=freq_d[ind];
            }
            else if(i==Nxs/2) 
            {
               ind=d*Nx*Nyr+Nx/2*Nyr+j;
               freqs_d[idx]=freq_d[ind];
            }
         }
         if(j==Nyrs-1)
         {
            if(i<Nx/2)
            {
               ind=d*Nx*Nyr+i*Nyr+Nyr-1;
               freqs_d[idx]=freq_d[ind];
            }
            else if(i>Nxs-Nx/2) 
            {
               ind=d*Nx*Nyr+(i-Nxs+Nx)*Nyr+Nyr-1;
               freqs_d[idx]=freq_d[ind];
            }
            else if(i==Nxs/2) 
            {
               ind=d*Nx*Nyr+Nx/2*Nyr+Nyr-1;
               freqs_d[idx]=freq_d[ind];
            }
         }
      }
      //freqs_d[idx].x/=l;
      //freqs_d[idx].y/=l;
   }
}

/////////////////////////////////////////////////////////////

//convolution (out must be initialized to zero)
__global__ void conv_k(cufftComplex *in, cufftComplex *out, cufftComplex *c, cufftReal *b, int dM, int dD, int Nx, int Ny)
{
   int Nyr=Ny/2+1;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < dM*Nx*Nyr)
   {
      int m=(int)idx/(Nx*Nyr);
      int i=(int)(idx-m*Nx*Nyr)/Nyr;
      int j=(int)(idx-m*Nx*Nyr-i*Nyr);
      cufftComplex out_t;
      out_t.x=0; out_t.y=0;
      for(int d=0; d<dD; d++)
      {
         cufftComplex in_t=in[d*Nx*Nyr+i*Nyr+j];
         in_t.x/=dM;
         in_t.y/=dM;
         cufftComplex c_t=c[m*dD*Nx*Nyr+d*Nx*Nyr+i*Nyr+j];
         //out[m*Nx*Nyr+i*Nyr+j].x+=(in_t.x*c_t.x-in_t.y*c_t.y);
         //out[m*Nx*Nyr+i*Nyr+j].y+=(in_t.x*c_t.y+in_t.y*c_t.x);
         out_t.x+=(in_t.x*c_t.x-in_t.y*c_t.y);
         out_t.y+=(in_t.x*c_t.y+in_t.y*c_t.x);
         if(d==0 && i==0 && j==0)
            out_t.x+=b[m]*Nx*Ny;
            //out[m*Nx*Nyr+i*Nyr+j].x+=b[m]*Nx*Ny;
      }
      out[idx]=out_t;
   }
}

__global__ void conv_k1(cufftComplex *in, cufftComplex *out, cufftComplex *c, cufftReal *b, int dM, int dD, int Nx, int Ny)
{
   extern __shared__ cufftComplex in_s[];
   extern __shared__ cufftComplex c_s[];
   int Nyr=Ny/2+1;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx < dM*dD*Nx*Nyr)
   {
      if(idx < dD*Nx*Nyr)
         in_s[idx]=in[idx];
      c_s[idx]=c[idx];
      __syncthreads();
      if (idx < dM*Nx*Nyr)
      {
         int m=(int)idx/(Nx*Nyr);
         int i=(int)(idx-m*Nx*Nyr)/Nyr;
         int j=(int)(idx-m*Nx*Nyr-i*Nyr);
         cufftComplex out_t;
         out_t.x=0; out_t.y=0;
         for(int d=0; d<dD; d++)
         {
            cufftComplex in_t=in_s[d*Nx*Nyr+i*Nyr+j];
            in_t.x/=dM;
            in_t.y/=dM;
            cufftComplex c_t=c_s[m*dD*Nx*Nyr+d*Nx*Nyr+i*Nyr+j];
            //out[m*Nx*Nyr+i*Nyr+j].x+=(in_t.x*c_t.x-in_t.y*c_t.y);
            //out[m*Nx*Nyr+i*Nyr+j].y+=(in_t.x*c_t.y+in_t.y*c_t.x);
            out_t.x+=(in_t.x*c_t.x-in_t.y*c_t.y);
            out_t.y+=(in_t.x*c_t.y+in_t.y*c_t.x);
            if(d==0 && i==0 && j==0)
               out_t.x+=b[m]*Nx*Ny;
               //out[m*Nx*Nyr+i*Nyr+j].x+=b[m]*Nx*Ny;
         }
         out[idx]=out_t;
      }
   }
}

/////////////////////////////////////////////////////////////

//normalize input to convolution to output size
__global__ void normalize( cufftComplex* freq_d, int dM, int dD, int Nx, int Ny)
{
   int Nyr=Ny/2+1;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if (idx < dD*Nx*Nyr)
   {
      freq_d[idx].x/=dM;
      freq_d[idx].y/=dM;
   }
}

/////////////////////////////////////////////////////////////

//copy cufftComplex Ntot array to float 2*Ntot array
__global__ void copy_out(cufftComplex *cfreq_d, float *Cc_d, int dM, int dD, int Nx, int Ny)
{
   int Ntot=dM*dD*Nx*Ny*2;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   int m=(int)idx/(dD*Nx*Ny*2); 
   int d=(int)(idx-m*dD*Nx*Ny*2)/(Nx*Ny*2); 
   int i=(int)(idx-m*dD*Nx*Ny*2-d*Nx*Ny*2)/(Ny*2);
   int j=(int)(idx-m*dD*Nx*Ny*2-d*Nx*Ny*2-i*Ny*2)/2;
   if (idx < Ntot)
   {
      int ind=m*dD*Nx*Ny+d*Nx*Ny+i*Ny+j;
      cufftComplex cfreq_t=cfreq_d[ind];
      float cfreq_val;
      if(idx%2==0) cfreq_val=cfreq_t.x;
      else cfreq_val=cfreq_t.y;
      Cc_d[idx]=cfreq_val;
   }
}


/////////////////////////////////////////////////////////////

//copy float 2*Ntot array to cufftComplex Ntot array
__global__ void copy_in(float *Cc_d, cufftComplex *cfreq_d, int dM, int dD, int Nx, int Ny)
{
   int Ntot=dM*dD*Nx*Ny;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   int m=(int)idx/(dD*Nx*Ny); 
   int d=(int)(idx-m*dD*Nx*Ny)/(Nx*Ny); 
   int i=(int)(idx-m*dD*Nx*Ny-d*Nx*Ny)/Ny;
   int j=(int)(idx-m*dD*Nx*Ny-d*Nx*Ny-i*Ny);
   if (idx < Ntot)
   {
      cfreq_d[idx].x=Cc_d[m*dD*Nx*Ny*2+d*Nx*Ny*2+i*Ny*2+j*2+0];
      cfreq_d[idx].y=Cc_d[m*dD*Nx*Ny*2+d*Nx*Ny*2+i*Ny*2+j*2+1];
   }
}

/////////////////////////////////////////////////////////////

__device__ void adapt_rate(float& delR, float& delI, float delmax, float dDdXR, float dDdXI, cufftComplex& ddx, cufftComplex dx)
{
   delR=delI=delmax;
   float dddxR=dDdXR-ddx.x;
   float dddxI=dDdXI-ddx.y;
   if(dddxR!=0) delR=abs(dx.x/dddxR);
   if(dddxI!=0) delI=abs(dx.y/dddxI);
   if(delR>delmax) delR=delmax;
   if(delI>delmax) delI=delmax;
   ddx.x=dDdXR;
   ddx.y=dDdXI;
}

/////////////////////////////////////////////////////////////

__device__ void adapt_rateR(float& del, float delmax, float dDdX, cufftReal& ddx, cufftReal dx)
{
   del=delmax;
   float dddx=dDdX-ddx;
   if(dddx!=0) del=abs(dx/dddx);
   if(del>delmax) del=delmax;
   ddx=dDdX;
}

/////////////////////////////////////////////////////////////

__global__ void backprop_k(cufftComplex *freq_d, cufftComplex *ofreq_d, cufftComplex *cfreq_d, cufftComplex *ffreq_d, cufftReal *b_d, cufftReal *p_d, cufftComplex *cfreq1_d, cufftComplex *ffreq1_d, cufftReal *b1_d, cufftReal *p1_d, cufftComplex *dc_d, cufftComplex *df_d, cufftReal *db_d, cufftReal *dp_d, cufftComplex *ddc_d, cufftComplex *ddf_d, cufftReal *ddb_d, cufftReal *ddp_d, int dM, int dD, int Nx, int Ny)
{
   int Nyr=Ny/2+1;
   float norm=Nx*Ny;
   float n=norm*2*dM*dD*Nx*Ny;
   float delmax=0.5;
   float delR, delI;
   float alpha=0.9;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<dM*dD*Nx*Nyr)
   {
      int m=(int)idx/(dD*Nx*Nyr);
      int d=(int)(idx-m*dD*Nx*Nyr)/(Nx*Nyr);
      int i=(int)(idx-m*dD*Nx*Nyr-d*Nx*Nyr)/Nyr;
      int j=(int)(idx-m*dD*Nx*Nyr-d*Nx*Nyr-i*Nyr);
      float Norm=n;
      if(j>0 && j<Nyr-1) Norm/=2;
      float sumcRR=0, sumcRI=0, sumcIR=0, sumcII=0;
      float sumfR=0, sumfI=0;
      float sumb=0;
      for(int d1=0;d1<dD;d1++)
      {
         //c derivative sums over d
         int ind=d1*Nx*Nyr+i*Nyr+j;
         int indf=d1*dM*Nx*Nyr+m*Nx*Nyr+i*Nyr+j;
         cufftComplex ofreq=ofreq_d[ind];
         cufftComplex freq=freq_d[ind];
         cufftComplex ffreq=ffreq_d[indf];
         sumcRR+=(ofreq.x-freq.x)*ffreq.x;
         sumcRI+=(ofreq.x-freq.x)*ffreq.y;
         sumcIR+=(ofreq.y-freq.y)*ffreq.x;
         sumcII+=(ofreq.y-freq.y)*ffreq.y;
         //f derivative sums over d
         int indc=m*dD*Nx*Nyr+d1*Nx*Nyr+i*Nyr+j;
         cufftComplex cfreq=cfreq_d[indc];
         sumfR+=cfreq.x*freq.x-cfreq.y*freq.y;
         sumfI+=cfreq.x*freq.y+cfreq.y*freq.x;
         //b derivative sum over d
         if(i==0 && j==0)
            sumb+=(ofreq.x-freq.x)*ffreq.x+(ofreq.y-freq.y)*ffreq.y;
      }
      //c update
      int ind=d*Nx*Nyr+i*Nyr+j;
      cufftComplex freq=freq_d[ind];
      float dDR=sumcRR*freq.x-sumcRI*freq.y+sumcIR*freq.y+sumcII*freq.x;
      float dDI=-sumcRR*freq.y-sumcRI*freq.x+sumcIR*freq.x-sumcII*freq.y;
      float dDdCR=dDR/Norm;
      float dDdCI=dDI/Norm;
      cufftComplex cfreq=cfreq_d[idx];
      cufftComplex dc=dc_d[idx];
      adapt_rate(delR, delI, delmax, dDdCR, dDdCI, ddc_d[idx], dc);
      dc.x=(1-alpha)*delR*dDdCR/((10<abs(dDdCR))?abs(dDdCR):10)+alpha*dc.x;
      dc.y=(1-alpha)*delI*dDdCI/((10<abs(dDdCI))?abs(dDdCI):10)+alpha*dc.y;
      cfreq1_d[idx].x=cfreq.x-dc.x;
      cfreq1_d[idx].y=cfreq.y-dc.y;
      dc_d[idx]=dc;
      //f update
      int idxf=d*dM*Nx*Nyr+m*Nx*Nyr+i*Nyr+j;
      float b0=0;
      if(i==0 && j==0) 
         b0=b_d[m]*norm;
      cufftComplex ofreq=ofreq_d[ind];
      float diffR=ofreq.x-freq.x;
      float diffI=ofreq.y-freq.y;
      dDR=diffR*(sumfR+b0)+diffI*sumfI;
      dDI=-diffR*sumfI+diffI*(sumfR+b0);
      float dDdFR=dDR/Norm;
      float dDdFI=dDI/Norm;
      cufftComplex ffreq=ffreq_d[idxf];
      cufftComplex df=df_d[idxf];
      adapt_rate(delR, delI, delmax, dDdFR, dDdFI, ddf_d[idxf], df);
      df.x=(1-alpha)*delR*dDdFR/((10<abs(dDdFR))?abs(dDdFR):10)+alpha*df.x;
      df.y=(1-alpha)*delI*dDdFI/((10<abs(dDdFI))?abs(dDdFI):10)+alpha*df.y;
      ffreq1_d[idxf].x=ffreq.x-df.x;
      ffreq1_d[idxf].y=ffreq.y-df.y;
      df_d[idxf]=df;
      //b update
      if(i==0 && j==0 && d==0)
      {
         float dDdB=sumb*norm/Norm;
         cufftReal db=db_d[m];
         adapt_rateR(delR, delmax, dDdB, ddb_d[m], db);
         db=(1-alpha)*delR*dDdB/((10<abs(dDdB))?abs(dDdB):10)+alpha*db;
         b1_d[m]=b_d[m]-db;
         db_d[m]=db;
      }
      //p update
      if(i==0 && j==0 && m==0)
      {
         float dDdP=(ofreq.x-freq.x)*norm/Norm;
         cufftReal dp=dp_d[d];
         adapt_rateR(delR, delmax, dDdP, ddp_d[d], dp);
         dp=(1-alpha)*delR*dDdP/((10<abs(dDdP))?abs(dDdP):10)+alpha*dp;
         p1_d[d]=p_d[d]-dp;
         dp_d[d]=dp;
      }
   }
}

/////////////////////////////////////////////////////////////

__global__ void gradient_k(cufftComplex *freq_d, cufftComplex *ofreq_d, cufftComplex *cfreq_d, cufftComplex *ffreq_d, cufftReal *b_d, cufftReal *p_d, cufftComplex *dc_d, cufftComplex *df_d, cufftReal *db_d, cufftReal *dp_d, int dM, int dD, int Nx, int Ny)
{
   int Nyr=Ny/2+1;
   float norm=Nx*Ny;
   float n=norm*2*dM*dD*Nx*Ny;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<dM*dD*Nx*Nyr)
   {
      int m=(int)idx/(dD*Nx*Nyr);
      int d=(int)(idx-m*dD*Nx*Nyr)/(Nx*Nyr);
      int i=(int)(idx-m*dD*Nx*Nyr-d*Nx*Nyr)/Nyr;
      int j=(int)(idx-m*dD*Nx*Nyr-d*Nx*Nyr-i*Nyr);
      float Norm=n;
      //if(j>0 && (Ny%2==0)?(j<Nyr-1):(j<Nyr)) Norm/=2; 
      float sumcRR=0, sumcRI=0, sumcIR=0, sumcII=0;
      float sumfR=0, sumfI=0;
      float sumb=0;
      for(int d1=0;d1<dD;d1++)
      {
         //c derivative sums over d
         int ind=d1*Nx*Nyr+i*Nyr+j;
         int indf=d1*dM*Nx*Nyr+m*Nx*Nyr+i*Nyr+j;
         cufftComplex ofreq=ofreq_d[ind];
         cufftComplex freq=freq_d[ind];
         cufftComplex ffreq=ffreq_d[indf];
         sumcRR+=(ofreq.x-freq.x)*ffreq.x;
         sumcRI+=(ofreq.x-freq.x)*ffreq.y;
         sumcIR+=(ofreq.y-freq.y)*ffreq.x;
         sumcII+=(ofreq.y-freq.y)*ffreq.y;
         //f derivative sums over d
         int indc=m*dD*Nx*Nyr+d1*Nx*Nyr+i*Nyr+j;
         cufftComplex cfreq=cfreq_d[indc];
         sumfR+=cfreq.x*freq.x-cfreq.y*freq.y;
         sumfI+=cfreq.x*freq.y+cfreq.y*freq.x;
         //b derivative sum over d
         if(i==0 && j==0)
            sumb+=(ofreq.x-freq.x)*ffreq.x+(ofreq.y-freq.y)*ffreq.y;
      }
      //c update
      int ind=d*Nx*Nyr+i*Nyr+j;
      cufftComplex freq=freq_d[ind];
      float dDR=sumcRR*freq.x-sumcRI*freq.y+sumcIR*freq.y+sumcII*freq.x;
      float dDI=-sumcRR*freq.y-sumcRI*freq.x+sumcIR*freq.x-sumcII*freq.y;
      float dDdCR=dDR/Norm;
      float dDdCI=dDI/Norm;
      cufftComplex dc;
      dc.x=dDdCR;
      dc.y=dDdCI;
      dc_d[idx]=dc;
      //f update
      int idxf=d*dM*Nx*Nyr+m*Nx*Nyr+i*Nyr+j;
      float b0=0;
      if(i==0 && j==0) 
         b0=b_d[m]*norm;
      cufftComplex ofreq=ofreq_d[ind];
      float diffR=ofreq.x-freq.x;
      float diffI=ofreq.y-freq.y;
      dDR=diffR*(sumfR+b0)+diffI*sumfI;
      dDI=-diffR*sumfI+diffI*(sumfR+b0);
      float dDdFR=dDR/Norm;
      float dDdFI=dDI/Norm;
      cufftComplex df;
      df.x=dDdFR;
      df.y=dDdFI;
      df_d[idxf]=df;
      //b update
      if(i==0 && j==0 && d==0)
      {
         float dDdB=sumb*norm/Norm;
         db_d[m]=dDdB;
      }
      //p update
      if(i==0 && j==0 && m==0)
      {
         float dDdP=(ofreq.x-freq.x)*norm/Norm;
         dp_d[d]=dDdP;
      }
   }
}


/////////////////////////////////////////////////////////////

//calculate mse
__global__ void calc_mse(cufftComplex *freq_d, cufftComplex *ofreq_d, int dD, int Nx, int Ny, float *Cout)
{
   int Nyr=Ny/2+1;
   float norm=dD*Nx*Ny;
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   if(idx<dD*Nx*Nyr)
   {
      int d=(int)(idx)/(Nx*Nyr);
      int i=(int)(idx-d*Nx*Nyr)/Nyr;
      int j=(int)(idx-d*Nx*Nyr-i*Nyr);
      float fR=freq_d[idx].x;
      float fI=freq_d[idx].y;
      float ofR=ofreq_d[idx].x;
      float ofI=ofreq_d[idx].y;
      float n=norm;
      if(j>0 && j<Nyr-1) n/=2;
      Cout[idx]=((fR-ofR)*(fR-ofR)+(fI-ofI)*(fI-ofI))/n;
   }
}

/////////////////////////////////////////////////////////////

//__global__ void shrink_k<<<blocks,threads>>>(cufftReal *dc_d, cufftReal *dck_d, int dM, int dD, int Nx, int Ny, int Nk, int Nl)
//{
//   int idx = threadIdx.x + blockDim.x*blockIdx.x;
//   if(idx<dM*dD*Nx*Ny)
//   {
//      int m=(int)idx/(dD*Nx*Ny);
//      int d=(int)(idx-m*dD*Nx*Ny)/(Nx*Ny);
//      int i=(int)(idx-m*dD*Nx*Ny-d*Nx*Ny)/Ny;
//      int j=(int)(idx-m*dD*Nx*Ny-d*Nx*Ny-i*Ny);
//      if(i<Nk/2+1 && j<Nl/2+1)
//      {
//         int idk=m*dD*Nk*Nl+d*Nk*Nl+(i+Nk/2)*Nk+(j+Nl/2);
//         dck_d[idk]=dc_d[idx];
//      }
//      if(i>=Nx-Nk/2 && j<Nl/2+1)
//      {
//         int idk=m*dD*Nk*Nl+d*Nk*Nl+(i-Nx+Nk/2)*Nk+(j+Nl/2);
//         dck_d[idk]=dc_d[idx];
//      }
//      if(i<Nk/2+1 && j>=Ny-Nl/2)
//      {
//         int idk=m*dD*Nk*Nl+d*Nk*Nl+(i+Nk/2)*Nk+(j-Ny+Nl/2);
//         dck_d[idk]=dc_d[idx];
//      }
//      if(i>=Nx-Nk/2 && j>=Ny-Nl/2)
//      {
//         int idk=m*dD*Nk*Nl+d*Nk*Nl+(i-Nx+Nk/2)*Nk+(j-Ny+Nl/2);
//         dck_d[idk]=dc_d[idx];
//      }
//   }
//}

//shrink padded kernel to Nk*Nl size
__global__ void shrink_k(cufftReal *dc_d, cufftReal *dck_d, int dM, int dD, int Nx, int Ny, int Nk, int Nl)
{
   int idk = threadIdx.x + blockDim.x*blockIdx.x;
   if(idk<dM*dD*Nk*Nl)
   {
      int m=(int)idk/(dD*Nk*Nl);
      int d=(int)(idk-m*dD*Nk*Nl)/(Nk*Nl);
      int k=(int)(idk-m*dD*Nk*Nl-d*Nk*Nl)/Nl;
      int l=(int)(idk-m*dD*Nk*Nl-d*Nk*Nl-k*Nl);
      if(k>=Nk/2 && l>=Nl/2)
      {
         int idx=m*dD*Nx*Ny+d*Nx*Ny+(k-Nk/2)*Ny+(l-Nl/2);
         dck_d[idk]=dc_d[idx];
      }
      if(k<Nk/2 && l>=Nl/2)
      {
         int idx=m*dD*Nx*Ny+d*Nx*Ny+(k+Nx-Nk/2)*Ny+(l-Nl/2);
         dck_d[idk]=dc_d[idx];
      }
      if(k>=Nk/2 && l<Nl/2)
      {
         int idx=m*dD*Nx*Ny+d*Nx*Ny+(k-Nk/2)*Ny+(l+Ny-Nl/2);
         dck_d[idk]=dc_d[idx];
      }
      if(k<Nk/2 && l<Nl/2)
      {
         int idx=m*dD*Nx*Ny+d*Nx*Ny+(k+Nx-Nk/2)*Ny+(l+Ny-Nl/2);
         dck_d[idk]=dc_d[idx];
      }
   }
}

/////////////////////////////////////////////////////////////

//expand kernel to input size Nx*Ny
__global__ void pad_k(cufftReal *dck_d, cufftReal *dc_d, int dM, int dD, int Nx, int Ny, int Nk, int Nl)
{
   int idk = threadIdx.x + blockDim.x*blockIdx.x;
   if(idk<dM*dD*Nk*Nl)
   {
      int m=(int)idk/(dD*Nk*Nl);
      int d=(int)(idk-m*dD*Nk*Nl)/(Nk*Nl);
      int k=(int)(idk-m*dD*Nk*Nl-d*Nk*Nl)/Nl;
      int l=(int)(idk-m*dD*Nk*Nl-d*Nk*Nl-k*Nl);
      if(k>=Nk/2 && l>=Nl/2)
      {
         int idx=m*dD*Nx*Ny+d*Nx*Ny+(k-Nk/2)*Ny+(l-Nl/2);
         dc_d[idx]=dck_d[idk];
      }
      if(k<Nk/2 && l>=Nl/2)
      {
         int idx=m*dD*Nx*Ny+d*Nx*Ny+(k+Nx-Nk/2)*Ny+(l-Nl/2);
         dc_d[idx]=dck_d[idk];
      }
      if(k>=Nk/2 && l<Nl/2)
      {
         int idx=m*dD*Nx*Ny+d*Nx*Ny+(k-Nk/2)*Ny+(l+Ny-Nl/2);
         dc_d[idx]=dck_d[idk];
      }
      if(k<Nk/2 && l<Nl/2)
      {
         int idx=m*dD*Nx*Ny+d*Nx*Ny+(k+Nx-Nk/2)*Ny+(l+Ny-Nl/2);
         dc_d[idx]=dck_d[idk];
      }
   }
}

/////////////////////////////////////////////////////////////

//compute backpropagation in coordinate space for kernel weights c, f and biases b, p
__global__ void backprop_d(cufftReal *c_d, cufftReal *f_d, cufftReal *b_d, cufftReal *p_d, cufftReal *dck_d, cufftReal *dfk_d, cufftReal *db_d, cufftReal *dp_d, cufftReal *Dc_d, cufftReal *Df_d, cufftReal *Db_d, cufftReal *Dp_d, cufftReal *ddc, cufftReal *ddf, cufftReal *ddb, cufftReal *ddp, int dD, int dM, int Nk, int Nl, float del)
{
   //float del=0.00001, delmax=0.1;
   float alpha=0.9;
   int idk = threadIdx.x + blockDim.x*blockIdx.x;
   if(idk<dM*dD*Nk*Nl)
   {
      //c update
      float dDdC=dck_d[idk];
      float dc=Dc_d[idk];
//      adapt_rateR(del, delmax, dDdC, ddc[idk], dc);
      float Dc=(1-alpha)*del*dDdC/((10<abs(dDdC))?abs(dDdC):10)+alpha*dc;
      c_d[idk]+= -Dc;
      Dc_d[idk]=Dc;
//      c_d[idk]+= -del*dck_d[idk];
      //f update
      float dDdF=dfk_d[idk];
      float df=Df_d[idk];
//      adapt_rateR(del, delmax, dDdF, ddf[idk], df);
      float Df=(1-alpha)*del*dDdF/((10<abs(dDdF))?abs(dDdF):10)+alpha*df;
      f_d[idk]+= -Df;
      Df_d[idk]=Df;
//      f_d[idk]+= -del*dfk_d[idk];
      //b update
      if(idk<dM)
      {
         float dDdB=db_d[idk];
         float db=Db_d[idk];
//         adapt_rateR(del, delmax, dDdB, ddb[idk], db);
         float Db=(1-alpha)*del*dDdB/((10<abs(dDdB))?abs(dDdB):10)+alpha*db;
         b_d[idk]+= -Db;
         Db_d[idk]=Db;
//         b_d[idk]+= -del*db_d[idk];
      }
      //p update
      if(idk<dD)
      {
         float dDdP=dp_d[idk];
         float dp=Dp_d[idk];
//         adapt_rateR(del, delmax, dDdP, ddp[idk], dp);
         float Dp=(1-alpha)*del*dDdP/((10<abs(dDdP))?abs(dDdP):10)+alpha*dp;
         p_d[idk]+= -Dp;
         Dp_d[idk]=Dp;
//         p_d[idk]+= -del*dp_d[idk];
      }
   }

}

/////////////////////////////////////////////////////////////

//compute backpropagation in coordinate space for kernel weights c, f and biases b, p in multiobjective optimization (D=w0*D0-w1*D1)
__global__ void backprop_double(cufftReal *c_d, cufftReal *f_d, cufftReal *b_d, cufftReal *p_d, cufftReal *dck_d, cufftReal *dfk_d, cufftReal *db_d, cufftReal *dp_d, cufftReal *Dc_d, cufftReal *Df_d, cufftReal *Db_d, cufftReal *Dp_d, cufftReal *ddc, cufftReal *ddf, cufftReal *ddb, cufftReal *ddp, cufftReal *cd_d, cufftReal *fd_d, cufftReal *bd_d, cufftReal *pd_d, int dD, int dM, int Nk, int Nl, float del, float w0, float w1)
{
   //float del=0.00001, delmax=0.1;
   float alpha=0.9;
   int idk = threadIdx.x + blockDim.x*blockIdx.x;
   if(idk<dM*dD*Nk*Nl)
   {
      //c update
      float dDdC=w0*dck_d[idk]-w1*cd_d[idk];
      float dc=Dc_d[idk];
//      adapt_rateR(del, delmax, dDdC, ddc[idk], dc);
      float Dc=(1-alpha)*del*dDdC/((10<abs(dDdC))?abs(dDdC):10)+alpha*dc;
      c_d[idk]+= -Dc;
      Dc_d[idk]=Dc;
//      c_d[idk]+= -del*dck_d[idk];
      //f update
      float dDdF=w0*dfk_d[idk]-w1*fd_d[idk];
      float df=Df_d[idk];
//      adapt_rateR(del, delmax, dDdF, ddf[idk], df);
      float Df=(1-alpha)*del*dDdF/((10<abs(dDdF))?abs(dDdF):10)+alpha*df;
      f_d[idk]+= -Df;
      Df_d[idk]=Df;
//      f_d[idk]+= -del*dfk_d[idk];
      //b update
      if(idk<dM)
      {
         float dDdB=w0*db_d[idk]-w1*bd_d[idk];
         float db=Db_d[idk];
//         adapt_rateR(del, delmax, dDdB, ddb[idk], db);
         float Db=(1-alpha)*del*dDdB/((10<abs(dDdB))?abs(dDdB):10)+alpha*db;
         b_d[idk]+= -Db;
         Db_d[idk]=Db;
//         b_d[idk]+= -del*db_d[idk];
      }
      //p update
      if(idk<dD)
      {
         float dDdP=w0*dp_d[idk]-w1*pd_d[idk];
         float dp=Dp_d[idk];
//         adapt_rateR(del, delmax, dDdP, ddp[idk], dp);
         float Dp=(1-alpha)*del*dDdP/((10<abs(dDdP))?abs(dDdP):10)+alpha*dp;
         p_d[idk]+= -Dp;
         Dp_d[idk]=Dp;
//         p_d[idk]+= -del*dp_d[idk];
      }
   }

}

/////////////////////////////////////////////////////////////

//compute gradient for kernel distance to use in multiobjective optimization
__global__ void gradient_diff(cufftReal *cd_d, cufftReal *fd_d, cufftReal *bd_d, cufftReal *pd_d, cufftReal *c_d, cufftReal *f_d, cufftReal *b_d, cufftReal * p_d, int dD, int dM, int Nk, int Nl)
{

   int idk = threadIdx.x + blockDim.x*blockIdx.x;
   if(idk < dD*dM*Nk*Nl)
   {
      int m=(int)idk/(dD*Nk*Nl);
      int d=(int)(idk-m*dD*Nk*Nl)/(Nk*Nl);
      int k=(int)(idk-m*dD*Nk*Nl-d*Nk*Nl)/Nl;
      int l=(int)(idk-m*dD*Nk*Nl-d*Nk*Nl-k*Nl);
      float sum_c=0, sum_f=0, sum_b=0, sum_p=0;
      for(int m1=0; m1<dM; m1++)
      {
         for(int d1=0; d1<dD; d1++)
         {
            if(m1!=m && d1!=d)
            {
               float den_c=0, den_f=0;
               for(int k1=0; k1<Nk; k1++)
               {
                  for(int l1=0; l1<Nl; l1++)
                  {
                     float den_c_t=c_d[m*dD*Nk*Nl+d*Nk*Nl+k1*Nl+l1]-c_d[m1*dD*Nk*Nl+d1*Nk*Nl+k1*Nl+l1];
                     float den_f_t=f_d[d*dM*Nk*Nl+m*Nk*Nl+k1*Nl+l1]-f_d[d1*dM*Nk*Nl+m1*Nk*Nl+k1*Nl+l1];
                     den_c+=den_c_t*den_c_t;
                     den_f+=den_f_t*den_f_t;
                  }
               }
               //den_c*=den_c;
               //den_f*=den_f;
               sum_c+=(c_d[m*dD*Nk*Nl+d*Nk*Nl+k*Nl+l]-c_d[m1*dD*Nk*Nl+d1*Nk*Nl+k*Nl+l])/den_c;
               sum_f+=(f_d[d*dM*Nk*Nl+m*Nk*Nl+k*Nl+l]-f_d[d1*dM*Nk*Nl+m1*Nk*Nl+k*Nl+l])/den_f;
            }
            if(m1==0 && d1!=d)
               sum_p+=1./(p_d[d]-p_d[d1]);
         }
         if(m1!=m)
            sum_b+=1./(b_d[m]-b_d[m1]);
      }
      cd_d[m*dD*Nk*Nl+d*Nk*Nl+k*Nl+l]=sum_c;
      fd_d[d*dM*Nk*Nl+m*Nk*Nl+k*Nl+l]=sum_f;
      bd_d[m]=sum_b;
      pd_d[d]=sum_p;
   }
}





/////////////////////////////////////////////////////////////
//       HOST CUDA FUNCTIONS
/////////////////////////////////////////////////////////////

//compute direct fft
void fft(vector<vector<vector<float> > >& in, cufftComplex *freq_d)
{
   int dD=in.size();
   int Nx=in[0].size();
   int Ny=in[0][0].size();
   int Ntot=dD*Nx*Ny;

   // cuFFT 2D plans for image FFT
   cufftHandle f_plan;
   int rank = 2;
   int n[2] = {Nx, Ny};
   int idist = Nx*Ny, odist = Nx*(Ny/2+1);
   int inembed[] = {Nx, Ny};
   int onembed[] = {Nx, Ny/2+1};
   int istride = 1, ostride = 1;
   cufftPlanMany(&f_plan,rank,n,inembed,istride,idist,onembed,ostride,odist,CUFFT_R2C,dD);

   //input vectors
   cufftReal *in_h, *in_d;
   cudaMallocHost((void **) &in_h, Ntot*sizeof(cufftReal));
   cudaMalloc(&in_d, Ntot*sizeof(cufftReal));
   for(int i=0;i<Nx;i++)
   {
      for(int j=0;j<Ny;j++)
      {
         for(int d=0;d<dD;d++)
            in_h[d*Nx*Ny+i*Ny+j]=(cufftReal)in[d][i][j];
      }
   }
   cudaMemcpy(in_d, in_h, Ntot*sizeof(cufftReal), cudaMemcpyHostToDevice);

   //Compute Forward FFT
   cufftExecR2C(f_plan, in_d, freq_d);

   cufftDestroy(f_plan);
   cudaFree(in_d);
   cudaFreeHost(in_h);
}

/////////////////////////////////////////////////////////////

//compute inverse fft
void fft_inv(cufftComplex *freq_d, vector<vector<vector<float> > >& out)
{
   int dM=out.size();
   int Nx=out[0].size();
   int Ny=out[0][0].size();
   int Ntot=dM*Nx*Ny;

   // cuFFT 2D plans for image FFT
   cufftHandle i_plan;
   int rank = 2;
   int n[2] = {Nx, Ny};
   int idist = Nx*Ny, odist = Nx*(Ny/2+1);
   int inembed[] = {Nx, Ny};
   int onembed[] = {Nx, Ny/2+1};
   int istride = 1, ostride = 1;
   cufftPlanMany(&i_plan,rank,n,onembed,ostride,odist,inembed,istride,idist,CUFFT_C2R,dM);

   //output vector
   cufftReal *out_h, *out_d;
   cudaMallocHost(&out_h, Ntot*sizeof(cufftReal));
   cudaMalloc(&out_d, Ntot*sizeof(cufftReal));

   //Compute Inverse FFT
   cufftExecC2R(i_plan, freq_d, out_d);

   float norm=1./(Nx*Ny);
   cudaMemcpy(out_h, out_d, Ntot*sizeof(cufftReal), cudaMemcpyDeviceToHost);

   for(int i=0;i<Nx;i++)
   {
      for(int j=0;j<Ny;j++)
      {
         for(int m=0;m<dM;m++)
         {
            int ind=m*Nx*Ny+i*Ny+j;
            out[m][i][j]=(float)out_h[ind]*norm;
            //out[m][i][j]=max(0.,min(out[m][i][j],255.));
            //cout<<out[d][i][j]<<" ";
         }
      }
   }

//      cout<<"N16inv={";
//      for(int m=0;m<dM;m++)
//      {
//         for(int i=0;i<Nx;i++)
//         {
//            for(int j=0;j<Ny;j++)
//            {
//               cout<<out[m][i][j]<<",";
//            }
//         }
//      }
//      cout<<"};"<<endl<<endl;

   cufftDestroy(i_plan);
   cudaFreeHost(out_h);
   cudaFree(out_d);
}

/////////////////////////////////////////////////////////////

//compute fft for convolutional layer
void kfft(vector<vector<vector<vector<float> > > >& c, vector<float>& b, cufftComplex *cfreq_d, cufftReal *b_d)
{
   int dM=c.size();
   int dD=c[0].size();
   int Nk=c[0][0].size();
   int Nl=c[0][0][0].size();
   int Ntot=dM*dD*Nk*Nl;

   // cuFFT 2D plans for kernel FFT
   cufftHandle f_plan;
   int rank = 2;
   int n[2] = {Nk, Nl};
   int idist = Nk*Nl, odist = Nk*(Nl/2+1);
   int inembed[] = {Nk, Nl};
   int onembed[] = {Nk, Nl/2+1};
   int istride = 1, ostride = 1;
   cufftPlanMany(&f_plan,rank,n,inembed,istride,idist,onembed,ostride,odist,CUFFT_R2C,dM*dD);

   //conv kernel
   cufftReal *c_h, *c_d, *b_h;
   cudaMallocHost((void **) &c_h, Ntot*sizeof(cufftReal));
   cudaMallocHost((void **) &b_h, dM*sizeof(cufftReal));
   cudaMalloc(&c_d, Ntot*sizeof(cufftReal));
   for(int m=0;m<dM;m++)
   {
      for(int d=0;d<dD;d++)
      {
         for(int i=0;i<Nk;i++)
         {
            for(int j=0;j<Nl;j++)
            {
              c_h[m*dD*Nk*Nl+d*Nk*Nl+i*Nl+j]=(cufftReal)c[m][d][i][j];
            }
         }
      }
      b_h[m]=(cufftReal)b[m];
   }
   cudaMemcpy(c_d, c_h, Ntot*sizeof(cufftReal), cudaMemcpyHostToDevice);
   cudaMemcpy(b_d, b_h, dM*sizeof(cufftReal), cudaMemcpyHostToDevice);

   //Compute Forward FFT kernel
   cufftExecR2C(f_plan, c_d, cfreq_d);

   cufftDestroy(f_plan);
   cudaFree(c_d);
   cudaFreeHost(c_h);
   cudaFreeHost(b_h);
}

/////////////////////////////////////////////////////////////

//compute inverse fft for convolutional layer
void kfft_inv(cufftComplex *cfreq_d, cufftReal *b_d, vector<vector<vector<vector<float> > > >& c, vector<float>& b)
{
   int dM=c.size();
   int dD=c[0].size();
   int Nk=c[0][0].size();
   int Nl=c[0][0][0].size();
   int Ntot=dM*dD*Nk*Nl;

   // cuFFT 2D plans for kernel FFT
   cufftHandle i_plan;
   int rank = 2;
   int n[2] = {Nk, Nl};
   int idist = Nk*Nl, odist = Nk*(Nl/2+1);
   int inembed[] = {Nk, Nl};
   int onembed[] = {Nk, Nl/2+1};
   int istride = 1, ostride = 1;
   cufftPlanMany(&i_plan,rank,n,onembed,ostride,odist,inembed,istride,idist,CUFFT_C2R,dM*dD);

   //output vector
   cufftReal *c_h, *c_d, *b_h;
   cudaMallocHost(&c_h, Ntot*sizeof(cufftReal));
   cudaMallocHost(&b_h, dM*sizeof(cufftReal));
   cudaMalloc(&c_d, Ntot*sizeof(cufftReal));

   //Compute Inverse FFT
   cufftExecC2R(i_plan, cfreq_d, c_d);

   float norm=1./(Nk*Nl);
   cudaMemcpy(c_h, c_d, Ntot*sizeof(cufftReal), cudaMemcpyDeviceToHost);
   cudaMemcpy(b_h, b_d, dM*sizeof(cufftReal), cudaMemcpyDeviceToHost);
   for(int m=0;m<dM;m++)
   {
      for(int d=0;d<dD;d++)
      {
         for(int i=0;i<Nk;i++)
         {
            for(int j=0;j<Nl;j++)
            {
               int ind=m*dD*Nk*Nl+d*Nk*Nl+i*Nl+j;
               c[m][d][i][j]=(float)c_h[ind]*norm;
            }
         }
      }
      b[m]=(float)b_h[m];
   }
   cufftDestroy(i_plan);
   cudaFree(c_d);
   cudaFreeHost(c_h);
   cudaFreeHost(b_h);
}

/////////////////////////////////////////////////////////////

//Pooling in fft space
void pool_fft(cufftComplex * &freq_d, int dD, int& Nx, int& Ny, int scale)
{
   if(scale!=1)
   {
      //crop spectrum around zero frequency
      float l=(float)scale;
      if(scale<0) l=-1./((float)scale);
      //resize spectrum
      int Nxs=Nx/l;
      int Nys=Ny/l;
      int threads=256;
      int blocks=(dD*Nxs*(Nys/2+1))/threads+1;
      cufftComplex *freqs_d;
      cudaMalloc(&freqs_d, dD*Nxs*(Nys/2+1)*sizeof(cufftComplex));
      //if(Nxs>Nx)
         cudaMemset(freqs_d, 0, dD*Nxs*(Nys/2+1)*sizeof(cufftComplex));

      resize<<<blocks,threads>>>(freq_d, freqs_d, dD, Nx, Ny, Nxs, Nys, l);

      cudaFree(freq_d);
      cudaMalloc(&freq_d, dD*Nxs*(Nys/2+1)*sizeof(cufftComplex));
      cudaMemcpy(freq_d, freqs_d, dD*Nxs*(Nys/2+1)*sizeof(cufftComplex), 
                  cudaMemcpyDeviceToDevice);
      cudaFree(freqs_d);
      Nx=Nxs;
      Ny=Nys;
   }
}

/////////////////////////////////////////////////////////////

//compute convolution in fft space
void conv_fft(cufftComplex *freq_d, cufftComplex *ofreq_d, cufftComplex *cfreq_d, cufftReal *b_d, int dM, int dD, int Nx, int Ny)
{
   cudaMemset(ofreq_d, 0, dM*Nx*(Ny/2+1)*sizeof(cufftComplex));
   int threads=256;
   int blocks=dM*Nx*(Ny/2+1)/threads+1;
   conv_k<<<blocks,threads>>>(freq_d, ofreq_d, cfreq_d, b_d, dM, dD, Nx, Ny);
}

/////////////////////////////////////////////////////////////

//Pad kernel to input size 
void kernel_pad(vector<vector<vector<vector<float> > > >& c, vector<vector<vector<vector<float> > > >& c_pad, int Nx, int Ny)
{
   int dM=c.size();
   int dD=c[0].size();
   int Nk=c[0][0].size();
   int Nl=c[0][0][0].size();
   vector<vector<vector<vector<float> > > > c_new(dM, vector<vector<vector<float> > >(dD, vector<vector<float> >(Nx, vector<float>(Ny))));
   // Copy original kernel to padded kernel
   for (int m = 0; m < dM; m++) 
   {
      for (int d = 0; d < dD; d++) 
      {
         for (int k = 0; k < Nx; k++) 
         {
            for (int l = 0; l < Ny; l++)
            {
               if(k>=0 && k<=Nk/2 && l>=0 && l<=Nl/2)
               {
                  int ik=Nk/2+k;
                  int il=Nl/2+l;
                  c_new[m][d][k][l] = c[m][d][ik][il];
               }
               else if(k>=Nx-Nk/2 && k<Nx && l>=0 && l<=Nl/2)
               {
                  int ik=k-(Nx-Nk/2);
                  int il=Nl/2+l;
                  c_new[m][d][k][l] = c[m][d][ik][il];
               }
               else if(k>=0 && k<=Nk/2 && l>=Ny-Nl/2 && l<Ny)
               {
                  int ik=Nk/2+k;
                  int il=l-(Ny-Nl/2);
                  c_new[m][d][k][l] = c[m][d][ik][il];
               }
               else if(k>=Nx-Nk/2 && k<Nx && l>=Ny-Nl/2 && l<Ny)
               {
                  int ik=k-(Nx-Nk/2);
                  int il=l-(Ny-Nl/2);
                  c_new[m][d][k][l] = c[m][d][ik][il];
               }
               else c_new[m][d][k][l] = 0;
            }
         }
      }
   }
   c_pad=c_new;
}

/////////////////////////////////////////////////////////////

//Shrink padded kernel to original size 
void kernel_invpad(vector<vector<vector<vector<float> > > >& c_pad, vector<vector<vector<vector<float> > > >& c, int Nx, int Ny)
{
   int dM=c.size();
   int dD=c[0].size();
   int Nk=c[0][0].size();
   int Nl=c[0][0][0].size();
   // Copy original kernel to padded kernel
   for (int m = 0; m < dM; m++) 
   {
      for (int d = 0; d < dD; d++) 
      {
         for (int k = 0; k < Nx; k++) 
         {
            for (int l = 0; l < Ny; l++)
            {
               if(k>=0 && k<=Nk/2 && l>=0 && l<=Nl/2)
               {
                  int ik=Nk/2+k;
                  int il=Nl/2+l;
                  c[m][d][ik][il] = c_pad[m][d][k][l];
               }
               else if(k>=Nx-Nk/2 && k<Nx && l>=0 && l<=Nl/2)
               {
                  int ik=k-(Nx-Nk/2);
                  int il=Nl/2+l;
                  c[m][d][ik][il] = c_pad[m][d][k][l];
               }
               else if(k>=0 && k<=Nk/2 && l>=Ny-Nl/2 && l<Ny)
               {
                  int ik=Nk/2+k;
                  int il=l-(Ny-Nl/2);
                  c[m][d][ik][il] = c_pad[m][d][k][l];
               }
               else if(k>=Nx-Nk/2 && k<Nx && l>=Ny-Nl/2 && l<Ny)
               {
                  int ik=k-(Nx-Nk/2);
                  int il=l-(Ny-Nl/2);
                  c[m][d][ik][il] = c_pad[m][d][k][l];
               }
            }
         }
      }
   }
}

/////////////////////////////////////////////////////////////

//store convolutional kernels (in fourier space) from device to vector
void store_cfreq(cufftComplex *cfreq_d, vector<float>& c_freq, int dM, int dD, int Nx, int Nyr)
{
   thrust::host_vector<float> c_h(dM*dD*Nx*Nyr*2);
   thrust::device_vector<float> c_d(c_h);
   float* Cc_d=thrust::raw_pointer_cast(&c_d[0]);
   int threads=256;
   int blocks=2*dM*dD*Nx*Nyr/threads+1;
   copy_out<<<blocks,threads>>>(cfreq_d, Cc_d, dM, dD, Nx, Nyr);
   thrust::copy(c_d.begin(), c_d.end(), c_freq.begin());
}

/////////////////////////////////////////////////////////////

//load convolutional kernels (in fourier space) to device
void load_cfreq(vector<float>& c_freq, vector<float>& net_b, cufftComplex *cfreq_d, cufftReal *b_d, int dM, int dD, int Nx, int Nyr)
{
   thrust::device_vector<float> c_d(c_freq);
   thrust::device_vector<cufftReal> netb_d(net_b);
   float* Cc_d=thrust::raw_pointer_cast(&c_d[0]);
   cufftReal* Cb_d=thrust::raw_pointer_cast(&netb_d[0]);
   int threads=256;
   int blocks=dM*dD*Nx*Nyr/threads+1;
   copy_in<<<blocks,threads>>>(Cc_d, cfreq_d, dM, dD, Nx, Nyr);
   cudaMemcpy(b_d, Cb_d, dM*sizeof(cufftReal), cudaMemcpyDeviceToDevice);
}

/////////////////////////////////////////////////////////////

//store or load fft transformed convolutional kernels
void StoreLoad_cfreq(vector<vector<float> >& net_cfreq, vector<vector<vector<vector<vector<float> > > > >& net_c, vector<vector<float> >& net_b, cufftComplex *cfreq_d, cufftReal *b_d, int dM, int dD, int Nx, int Ny, int n)
{
         if(net_cfreq.size() < net_c.size()) //cfreq not yet computed and stored
         {
            vector<vector<vector<vector<float> > > > c_pad;
            kernel_pad(net_c[n], c_pad, Nx, Ny);
            kfft(c_pad, net_b[n], cfreq_d, b_d);
            //store fft kernel to host
            vector<float> c_freq(dM*dD*Nx*(Ny/2+1)*2);
            store_cfreq(cfreq_d, c_freq, dM, dD, Nx, Ny/2+1);
            if(n==net_cfreq.size())
               net_cfreq.push_back(c_freq);
         }
         else //copy stored fft kernel to device
            load_cfreq(net_cfreq[n], net_b[n], cfreq_d, b_d, dM, dD, Nx, Ny/2+1);
}

/////////////////////////////////////////////////////////////

//export fft convolutional kernels to standard convolutional kernels
void export_cfreq(vector<vector<vector<vector<float> > > >& c, vector<float>& b, cufftComplex *cfreq_d, cufftReal *b_d, int dM, int dD, int Nx, int Ny)
{
   vector<vector<vector<vector<float> > > > c_pad(dM, vector<vector<vector<float> > >(dD, 
                                            vector<vector<float> >(Nx, vector<float>(Ny))));
   kfft_inv(cfreq_d, b_d, c_pad, b);
   kernel_invpad(c_pad, c, Nx, Ny);
}


/////////////////////////////////////////////////////////////

//calculate mse fft
float mse_fft(cufftComplex *freq_d, cufftComplex *ofreq_d, int dM, int dD, int Nx, int Ny)
{
   thrust::device_vector<cufftReal> mse_d(dD*Nx*(Ny/2+1));
   cufftReal* Cmse=thrust::raw_pointer_cast(&mse_d[0]);
   if(true)
   {
      int threads=256;
      int blocks=(dD*Nx*(Ny/2+1))/threads+1;
      calc_mse<<<blocks,threads>>>(freq_d, ofreq_d, dD, Nx, Ny, Cmse);
   }
   float vmse = thrust::reduce(mse_d.begin(), mse_d.end());
   float norm=2*dM*Nx*Ny;
   vmse/=norm;
   return vmse;
}

/////////////////////////////////////////////////////////////

//backpropagation in coordinate space
void backprop(cufftReal *c_d, cufftReal *f_d, cufftComplex *cfreq_d, cufftComplex *ffreq_d, cufftReal *b_d, cufftReal *p_d, cufftComplex *fdc_d, cufftComplex *fdf_d, cufftReal *db_d, cufftReal *dp_d, cufftReal *Dc_d, cufftReal *Df_d, cufftReal *Db_d, cufftReal *Dp_d, cufftReal *ddc, cufftReal *ddf, cufftReal *ddb, cufftReal *ddp, int dM, int dD, int Nx, int Ny, int Nk, int Nl, float del, int maxdiff)
{

   // cuFFT 2D plans for kernel FFT
   cufftHandle i_plan, f_plan;
   int rank = 2;
   int n[2] = {Nx, Ny};
   int idist = Nx*Ny, odist = Nx*(Ny/2+1);
   int inembed[] = {Nx, Ny};
   int onembed[] = {Nx, Ny/2+1};
   int istride = 1, ostride = 1;
   cufftPlanMany(&i_plan,rank,n,onembed,ostride,odist,inembed,istride,idist,CUFFT_C2R,dM*dD);
   cufftPlanMany(&f_plan,rank,n,inembed,istride,idist,onembed,ostride,odist,CUFFT_R2C,dM*dD);

   //output vector
   cufftReal *dc_d, *df_d, *dck_d, *dfk_d;
   cudaMalloc(&dc_d, dM*dD*Nx*Ny*sizeof(cufftReal));
   cudaMalloc(&df_d, dD*dM*Nx*Ny*sizeof(cufftReal));
   cudaMalloc(&dck_d, dM*dD*Nk*Nl*sizeof(cufftReal));
   cudaMalloc(&dfk_d, dD*dM*Nk*Nl*sizeof(cufftReal));

   //Compute Inverse FFT of gradients
   cufftExecC2R(i_plan, fdc_d, dc_d);
   cufftExecC2R(i_plan, fdf_d, df_d);
   
   //shrink kernel
   int threads=256;
   int blocks=(dM*dD*Nk*Nl)/threads+1;
   shrink_k<<<blocks,threads>>>(dc_d, dck_d, dM, dD, Nx, Ny, Nk, Nl);
   shrink_k<<<blocks,threads>>>(df_d, dfk_d, dD, dM, Nx, Ny, Nk, Nl);

   //multiobjective optimization (min reconstruction error max filter difference)
   if(maxdiff)
   {
      cufftReal *cd_d, *fd_d, *bd_d, *pd_d;
      cudaMalloc(&cd_d, dM*dD*Nk*Nl*sizeof(cufftReal));
      cudaMalloc(&fd_d, dD*dM*Nk*Nl*sizeof(cufftReal));
      cudaMalloc(&bd_d, dM*sizeof(cufftReal));
      cudaMalloc(&pd_d, dD*sizeof(cufftReal));
      //gradient to maximize filter difference
      gradient_diff<<<blocks,threads>>>(cd_d, fd_d, bd_d, pd_d, 
                                                  c_d, f_d, b_d, p_d, 
                                                  dD, dM, Nk, Nl);

//      cufftReal *cd_h;//, *fd_h, *bd_h, *pd_h;
//      cudaMallocHost(&cd_h, dM*dD*Nk*Nl*sizeof(cufftReal));
//      //cudaMallocHost(&fd_h, dD*dM*Nk*Nl*sizeof(cufftReal));
//      //cudaMallocHost(&bd_h, dM*sizeof(cufftReal));
//      //cudaMallocHost(&pd_h, dD*sizeof(cufftReal));
//      cudaMemcpy(cd_h, cd_d, dM*dD*Nk*Nl*sizeof(cufftReal), cudaMemcpyDeviceToHost);
//      for(int i=0;i<dM*dD*Nk*Nl;i++) cout<<cd_h[i]<<", ";
//      cout<<endl;
//      cudaFreeHost(cd_h);

      //update kernel values in coordinate space
      float w0=1, w1=10;
      backprop_double<<<blocks,threads>>>(c_d, f_d, b_d, p_d,  
                                     dck_d, dfk_d, db_d, dp_d,
                                     Dc_d, Df_d, Db_d, Dp_d,
                                     ddc, ddf, ddb, ddp, 
                                     cd_d, fd_d, bd_d, pd_d,
                                     dD, dM, Nk, Nl, del, w0, w1);
      cudaFree(cd_d);
      cudaFree(fd_d);
      cudaFree(bd_d);
      cudaFree(pd_d);
   }
   else
   {
      //update kernel values in coordinate space
      backprop_d<<<blocks,threads>>>(c_d, f_d, b_d, p_d,  
                                     dck_d, dfk_d, db_d, dp_d,
                                     Dc_d, Df_d, Db_d, Dp_d,
                                     ddc, ddf, ddb, ddp, 
                                     dD, dM, Nk, Nl, del);
   }
   //pad kernel
   cudaMemset(dc_d, 0, dM*dD*Nx*Ny*sizeof(cufftReal));
   cudaMemset(df_d, 0, dM*dD*Nx*Ny*sizeof(cufftReal));
   pad_k<<<blocks,threads>>>(c_d, dc_d, dM, dD, Nx, Ny, Nk, Nl);
   pad_k<<<blocks,threads>>>(f_d, df_d, dD, dM, Nx, Ny, Nk, Nl);


   //Compute FFT of updated kernel
   cufftExecR2C(f_plan, dc_d, cfreq_d);
   cufftExecR2C(f_plan, df_d, ffreq_d);

   cufftDestroy(i_plan);
   cufftDestroy(f_plan);
   cudaFree(dc_d);
   cudaFree(df_d);
   cudaFree(dck_d);
   cudaFree(dfk_d);

}

/////////////////////////////////////////////////////////////

void flatten_kernel(vector<vector<vector<vector<float> > > >& c, cufftReal *c_d)
{

   int dM=c.size();
   int dD=c[0].size();
   int Nk=c[0][0].size();
   int Nl=c[0][0][0].size();
   int Ntot=dM*dD*Nk*Nl;

   cufftReal *c_h;
   cudaMallocHost((void **) &c_h, Ntot*sizeof(cufftReal));
   for(int m=0;m<dM;m++)
   {
      for(int d=0;d<dD;d++)
      {
         for(int i=0;i<Nk;i++)
         {
            for(int j=0;j<Nl;j++)
            {
              c_h[m*dD*Nk*Nl+d*Nk*Nl+i*Nl+j]=(cufftReal)c[m][d][i][j];
            }
         }
      }
   }
   cudaMemcpy(c_d, c_h, Ntot*sizeof(cufftReal), cudaMemcpyHostToDevice);
   cudaFreeHost(c_h);
}



/////////////////////////////////////////////////////////////
//          HOST EXTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////


//run autoencoder in fft space
void autoenc_fft(vector<vector<vector<vector<float> > > >& layers, vector<vector<vector<vector<vector<float> > > > >& net_c, vector<vector<float> >& net_cfreq, vector<vector<float> >& net_b, vector<int>& scale, int fft_l)
{
   int dD=layers[0].size();
   int dM=net_c[0].size();
   int Nx=layers[0][0].size();
   int Ny=layers[0][0][0].size();
   cufftComplex *freq_d, *ofreq_d, *cfreq_d;
   cufftReal *b_d;
   cudaMalloc(&freq_d, dD*Nx*(Ny/2+1)*sizeof(cufftComplex));
   fft(layers[0],freq_d);
   int l=1;
   for(int n=0;n<net_c.size();n++)
   {
      if(n<net_c.size()/2) 
      {
         pool_fft(freq_d, dD, Nx, Ny, scale[n]);
         if(fft_l) {fft_inv(freq_d,layers[l]); l+=1;}
      }
      cudaMalloc(&ofreq_d, dM*Nx*(Ny/2+1)*sizeof(cufftComplex));
      cudaMalloc(&cfreq_d, dM*dD*Nx*(Ny/2+1)*sizeof(cufftComplex));
      cudaMalloc(&b_d, dM*sizeof(cufftReal));
      cudaMemset(cfreq_d, 0, dM*dD*Nx*(Ny/2+1)*sizeof(cufftComplex));
      StoreLoad_cfreq(net_cfreq, net_c, net_b, cfreq_d, b_d, dM, dD, Nx, Ny, n);
//cudaMemcpy(ofreq_d, freq_d, dM*Nx*(Ny/2+1)*sizeof(cufftComplex), 
//                  cudaMemcpyDeviceToDevice);
      conv_fft(freq_d, ofreq_d, cfreq_d, b_d, dM, dD, Nx, Ny);
      if(fft_l) {fft_inv(ofreq_d,layers[l]); l+=1;}
      if(n>=net_c.size()/2) 
      {
         pool_fft(ofreq_d, dM, Nx, Ny, scale[n]);
         if(fft_l) {fft_inv(ofreq_d,layers[l]); l+=1;}
      }
      cudaFree(freq_d);
      cudaMalloc(&freq_d, dM*Nx*(Ny/2+1)*sizeof(cufftComplex));
      cudaMemcpy(freq_d, ofreq_d, dM*Nx*(Ny/2+1)*sizeof(cufftComplex), 
                  cudaMemcpyDeviceToDevice);
      dD=dM;
      dM=net_c[n+1].size();
      cudaFree(ofreq_d);
      cudaFree(cfreq_d);
      cudaFree(b_d);
   }
   if(!fft_l) fft_inv(freq_d,layers.back());
   cudaFree(freq_d);

}

/////////////////////////////////////////////////////////////

//run backpropagation in fft space
void backprop_fft(vector<vector<vector<float> > >& in, vector<vector<vector<float> > >& out, vector<float>& cfreq, vector<vector<vector<vector<float> > > >& c, vector<float>& ffreq, vector<vector<vector<vector<float> > > >& f, vector<float>& b, vector<float>& p, int dM, float del0, int maxdiff)
{
   int dD=in.size();
   int Nx=in[0].size();
   int Ny=in[0][0].size();
   int Nk=c[0][0].size();
   int Nl=c[0][0][0].size();
   cufftComplex *freq_d, *hfreq_d, *ofreq_d, *cfreq_d, *ffreq_d;
   cufftReal *c_d, *f_d, *b_d, *p_d;
   cufftReal *Dc_d, *Df_d, *Db_d, *Dp_d;
   cufftReal *ddc, *ddf, *ddb, *ddp;
   cufftComplex *dc_d, *df_d;
   cufftReal *db_d, *dp_d;
   cudaMalloc(&freq_d, dD*Nx*(Ny/2+1)*sizeof(cufftComplex));
   cudaMalloc(&hfreq_d, dM*Nx*(Ny/2+1)*sizeof(cufftComplex));
   cudaMalloc(&ofreq_d, dD*Nx*(Ny/2+1)*sizeof(cufftComplex));
   cudaMalloc(&cfreq_d, dM*dD*Nx*(Ny/2+1)*sizeof(cufftComplex));
   cudaMalloc(&ffreq_d, dD*dM*Nx*(Ny/2+1)*sizeof(cufftComplex));
   cudaMalloc(&c_d, dM*dD*Nk*Nl*sizeof(cufftReal));
   cudaMalloc(&f_d, dD*dM*Nk*Nl*sizeof(cufftReal));
   cudaMalloc(&b_d, dM*sizeof(cufftReal));
   cudaMalloc(&p_d, dD*sizeof(cufftReal));
   cudaMalloc(&dc_d, dM*dD*Nx*(Ny/2+1)*sizeof(cufftComplex));
   cudaMalloc(&df_d, dD*dM*Nx*(Ny/2+1)*sizeof(cufftComplex));
   cudaMalloc(&db_d, dM*sizeof(cufftReal));
   cudaMalloc(&dp_d, dD*sizeof(cufftReal));
   cudaMalloc(&Dc_d, dM*dD*Nk*Nl*sizeof(cufftReal));
   cudaMalloc(&Df_d, dD*dM*Nk*Nl*sizeof(cufftReal));
   cudaMalloc(&Db_d, dM*sizeof(cufftReal));
   cudaMalloc(&Dp_d, dD*sizeof(cufftReal));
   cudaMalloc(&ddc, dM*dD*Nk*Nl*sizeof(cufftReal));
   cudaMalloc(&ddf, dD*dM*Nk*Nl*sizeof(cufftReal));
   cudaMalloc(&ddb, dM*sizeof(cufftReal));
   cudaMalloc(&ddp, dD*sizeof(cufftReal));
   cudaMemset(dc_d, 0, dM*dD*Nx*(Ny/2+1)*sizeof(cufftComplex));
   cudaMemset(df_d, 0, dD*dM*Nx*(Ny/2+1)*sizeof(cufftComplex));
   cudaMemset(db_d, 0, dM*sizeof(cufftReal));
   cudaMemset(dp_d, 0, dD*sizeof(cufftReal));
   cudaMemset(Dc_d, 0, dM*dD*Nk*Nl*sizeof(cufftReal));
   cudaMemset(Df_d, 0, dD*dM*Nk*Nl*sizeof(cufftReal));
   cudaMemset(Db_d, 0, dM*sizeof(cufftReal));
   cudaMemset(Dp_d, 0, dD*sizeof(cufftReal));
   cudaMemset(ddc, 0, dM*dD*Nk*Nl*sizeof(cufftReal));
   cudaMemset(ddf, 0, dD*dM*Nk*Nl*sizeof(cufftReal));
   cudaMemset(ddb, 0, dM*sizeof(cufftReal));
   cudaMemset(ddp, 0, dD*sizeof(cufftReal));

   //fft in out
   fft(in,freq_d);
   fft(out,ofreq_d);
   //load fft conv weights
   load_cfreq(cfreq, b, cfreq_d, b_d, dM, dD, Nx, Ny/2+1);
   load_cfreq(ffreq, p, ffreq_d, p_d, dD, dM, Nx, Ny/2+1);
   flatten_kernel(c,c_d);
   flatten_kernel(f,f_d);

   //mse fft
   float vmse=mse_fft(freq_d, ofreq_d, dM, dD, Nx, Ny);
   cout<<"mse fft: "<<vmse<<endl;
   //float vmse_prev=vmse;
   //backpropagation
   //float del=0.00002;
   float del=0.01*del0;
   for(int n=0;n<100;n++)
   {
      int threads=256;
      int blocks=(dM*dD*Nx*(Ny/2+1))/threads+1;
//      backprop_k<<<blocks,threads>>>(freq_d, ofreq_d, cfreq_d, ffreq_d, b_d, p_d, 
//                                    cfreq1_d, ffreq1_d, b1_d, p1_d, 
//                                    dc_d, df_d, db_d, dp_d, 
//                                    ddc_d, ddf_d, ddb_d, ddp_d, 
//                                    dM, dD, Nx, Ny);
      //float del=del0;
      //if(vmse<10) del=10*del0;
      gradient_k<<<blocks,threads>>>(freq_d, ofreq_d, cfreq_d, ffreq_d, b_d, p_d,  
                                    dc_d, df_d, db_d, dp_d, dM, dD, Nx, Ny);
      backprop(c_d, f_d, cfreq_d, ffreq_d, b_d, p_d, dc_d, df_d, db_d, dp_d, 
               Dc_d, Df_d, Db_d, Dp_d, ddc, ddf, ddb, ddp, 
               dM, dD, Nx, Ny, Nk, Nl, del, maxdiff);

      conv_fft(freq_d, hfreq_d, cfreq_d, b_d, dM, dD, Nx, Ny);
      conv_fft(hfreq_d, ofreq_d, ffreq_d, p_d, dD, dM, Nx, Ny);
      float vmse=mse_fft(freq_d, ofreq_d, dM, dD, Nx, Ny);
      cout<<"n: "<<n<<" mse: "<<vmse<<endl;
//      if(vmse<vmse_prev)
//      {
//         cudaMemcpy(cfreq_d, cfreq1_d, dM*dD*Nx*(Ny/2+1)*sizeof(cufftComplex), 
//                     cudaMemcpyDeviceToDevice);
//         cudaMemcpy(ffreq_d, ffreq1_d, dM*dD*Nx*(Ny/2+1)*sizeof(cufftComplex), 
//                     cudaMemcpyDeviceToDevice);
//         cudaMemcpy(b_d, b1_d, dM*sizeof(cufftReal), 
//                     cudaMemcpyDeviceToDevice);
//         cudaMemcpy(p_d, p1_d, dD*sizeof(cufftReal), 
//                     cudaMemcpyDeviceToDevice);
//         vmse_prev=vmse;
//      }
//      ofstream file;
//      file.open("./fmse.txt", ios::out|ios::app);
//      file<<setprecision(9)<<vmse<<"\n"; 
//      file.close();
   }
   //inverse fft and store learned weights
   //fft_inv(ofreq_d,out);
   store_cfreq(cfreq_d, cfreq, dM, dD, Nx, Ny/2+1);
   store_cfreq(ffreq_d, ffreq, dD, dM, Nx, Ny/2+1);
   //export learned kernels
   export_cfreq(c, b, cfreq_d, b_d, dM, dD, Nx, Ny);
   export_cfreq(f, p, ffreq_d, p_d, dD, dM, Nx, Ny);
   cudaFree(freq_d);
   cudaFree(hfreq_d);
   cudaFree(ofreq_d);
   cudaFree(cfreq_d);
   cudaFree(ffreq_d);
   cudaFree(c_d);
   cudaFree(f_d);
   cudaFree(b_d);
   cudaFree(p_d);
   cudaFree(dc_d);
   cudaFree(df_d);
   cudaFree(db_d);
   cudaFree(dp_d);
   cudaFree(Dc_d);
   cudaFree(Df_d);
   cudaFree(Db_d);
   cudaFree(Dp_d);
   cudaFree(ddc);
   cudaFree(ddf);
   cudaFree(ddb);
   cudaFree(ddp);
}

