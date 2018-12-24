#include<ctime>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include<vector>
#include <chrono>
#include <fstream>
#include<iomanip>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>


#include "backproplib.h"

using namespace std;

// CUDA: use 512 threads per block
//const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
//inline int CAFFE_GET_BLOCKS(const int N) {
//  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;

void adapt_rate(float& del, float delmax, int active, float dDdX, float& ddx, float& dx)
{
   float dddx=dDdX-ddx;
   if(dddx!=0 && active>0) del=abs(dx/dddx);
   if(del>delmax) del=delmax;
   ddx=dDdX;
   del=delmax;
}


float act(float x) //activation function
{
   //float a=0.01;
   //if(x>0) return x;
   //else return a*x;
   return x;
}
float act1(float x) //activation function derivative
{
   //float a=0.01;
   //if(x>0) return 1;
   //else return a;
   return 1;
}

__device__ float act_d(float x) //activation function
{
   //float a=0.01;
   //if(x>0) return x;
   //else return a*x;
   return x;
}
__device__ float act1_d(float x) //activation function derivative
{
   //float a=0.01;
   //if(x>0) return 1;
   //else return a;
   return 1;
}

////////////////////////////////////////////////////////////////////////////////

__global__ void conv_parallel(int Nx, int Ny, int dD, int dM, int Nk, int Nl,int ak, int al,
                              float* __restrict__ const in, float* __restrict__ out, 
                              float* __restrict__ const c, float* __restrict__ const b)
//__global__ void conv_parallel(int Nx, int Ny, int dD, int dM, int Nk, int Nl,int ak, int al,
//                                 float *in, float *out, float *c, float *b)
{
//   int i = blockDim.x * blockIdx.x + threadIdx.x;
//   int j = blockDim.y * blockIdx.y + threadIdx.y;
//   int m = blockDim.z * blockIdx.z + threadIdx.z;
   int n = blockDim.x * blockIdx.x + threadIdx.x;
   int m=(int)n/(Nx*Ny);
   int i=(int)(n-m*Nx*Ny)/Ny;
   int j=(int)(n-m*Nx*Ny-i*Ny);

   if(i<Nx && j<Ny && m<dM)
   {
      float h=0;
      for(int d=0;d<dD;d++)
      {
         int ik=-2*ak-1;
         for(int k=0;k<Nk;k++)
         {
            int il=-2*al-1;
            for(int l=0;l<Nl;l++)
            {
               if(i-ik>=0 && i-ik<Nx && j-il>=0 && j-il<Ny)
               {
                  float c_t=(float)c[m*dD*Nk*Nl+d*Nk*Nl+k*Nl+l];
                  float in_t=(float)in[d*Nx*Ny+(i-ik)*Ny+(j-il)];
                  //h+=c[m*dD*Nk*Nl+d*Nk*Nl+k*Nl+l]*in[d*Nx*Ny+(i-ik)*Ny+(j-il)];
                  h+=(float)c_t*in_t;
               }
               il+=1;
            }
            ik+=1;
         }
      }
      h+=b[m];
      out[m*Nx*Ny+i*Ny+j]=act_d(h);
      //out[m*Nx*Ny+i*Ny+j]=max(min(act_d(h),255.),0.);
   }
}

//Convolution
void Conv_gpu(vector<vector<vector<float> > >& in, vector<vector<vector<float> > >& out, vector<vector<vector<vector<float > > > >& c, vector<float>& b)
{
   //int stride=1;
   int Nx=in[0].size();
   int Ny=in[0][0].size();
   int dM=c.size();
   int dD=c[0].size();
   int Nk=c[0][0].size();
   int Nl=c[0][0][0].size();
   int ak=((Nk-1)/2-1)/2;
   int al=((Nl-1)/2-1)/2;
   //thrust vectors
   thrust::host_vector<float> in_h(dD*Nx*Ny);//, out_h(dM*Nx*Ny);
   thrust::host_vector<float> c_h(dM*dD*Nk*Nl);
   thrust::host_vector<float> b_h(b);
   for(int i=0;i<Nx;i++)
   {
      for(int j=0;j<Ny;j++)
      {
         for(int d=0;d<dD;d++)
            in_h[d*Nx*Ny+i*Ny+j]=in[d][i][j]/dM;
      }
   }
   for(int m=0;m<dM;m++)
   {
      for(int d=0;d<dD;d++)
      {
         for(int k=0;k<Nk;k++)
         {
            for(int l=0;l<Nl;l++)      
            {
               c_h[m*dD*Nk*Nl+d*Nk*Nl+k*Nl+l]=c[m][d][k][l];
            }
         }
      }
   }
   thrust::device_vector<float> in_d(in_h), out_d(dM*Nx*Ny);
   thrust::device_vector<float> c_d(c_h), b_d(b_h);
   //cast thrust vectors to pointer to be processed by kernel
   float* Cin_d   = thrust::raw_pointer_cast( &in_d[0]  );
   float* Cout_d  = thrust::raw_pointer_cast( &out_d[0] );
   float* Cc_d    = thrust::raw_pointer_cast( &c_d[0]   );
   float* Cb_d    = thrust::raw_pointer_cast( &b_d[0]   );

   //call kernel function to run on gpu
   //dim3 threads(8,8,4);
   //dim3 blocks(Nx/threads.x+1,Ny/threads.y+1,dM/threads.z+1);
   int threads=256;
   int blocks=(dM*Nx*Ny)/threads+1;
//         auto start0 = std::chrono::high_resolution_clock::now(); 
   conv_parallel<<<blocks,threads>>>(Nx,Ny,dD,dM,Nk,Nl,ak,al,
                                     Cin_d, Cout_d, Cc_d, Cb_d);
//   cudaDeviceSynchronize();
//         auto finish0 = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsed0 = finish0 - start0;
//         std::cout<<"Time: " << elapsed0.count() << " s\n";
   //Copying device vectors back to host
   thrust::host_vector<float> out_h(out_d);
   for(int i=0;i<Nx;i++)
   {
      for(int j=0;j<Ny;j++)
      {
         for(int m=0;m<dM;m++)
         {
            out[m][i][j]=out_h[m*Nx*Ny+i*Ny+j];
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////

__global__ void backprop_parallel_CFBP(int Nx, int Ny, int dD, int dM, int Nk, int Nl, 
                                 float Norm, int ak, int al, int m, int d, int ik, int il, 
                                 float* __restrict__ const in, float* __restrict__ const out, 
                                 float* __restrict__ const hin, float* __restrict__ const f,
                                 float* __restrict__ dDdC, float* __restrict__ dDdF, 
                                 float* __restrict__ dDdB, float* __restrict__ dDdP)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int j = blockDim.y * blockIdx.y + threadIdx.y;
   if(i<Nx && j< Ny)
   {
      float dDdC2=0;
      float dDdB2=0;
      for(int d1=0;d1<dD;d1++)
      {
         float dDdB1=0;
         float dDdC1=0;
         for(int k1=0;k1<Nk;k1++)   //sum
         {
            int ik1=-2*ak-1+k1;
            for(int l1=0;l1<Nl;l1++)   //sum
            {
               int il1=-2*al-1+l1;
               if(i-ik1>=0 && i-ik1<Nx && j-il1>=0 && j-il1<Ny)
               {
                  float prod=f[d1*dM*Nk*Nl+m*Nk*Nl+k1*Nl+l1]*act1_d(hin[m*Nx*Ny+(i-ik1)*Ny+(j-il1)]);
                  dDdB1+=prod;
                  if(i-ik1-ik>=0 && i-ik1-ik<Nx && j-il1-il>=0 && j-il1-il<Ny)
                     dDdC1+=prod*in[d*Nx*Ny+(i-ik1-ik)*Ny+(j-il1-il)];
               }
            }
         }
         float sum0=(out[d1*Nx*Ny+i*Ny+j]-in[d1*Nx*Ny+i*Ny+j])*act1_d(out[d1*Nx*Ny+i*Ny+j]);
         dDdC2+=sum0*dDdC1/Norm;
         dDdB2=sum0*dDdB1/Norm;
         //dDdC[d1*Nx*Ny+i*Ny+j]=sum0*dDdC1/Norm;
         //dDdB[d1*Nx*Ny+i*Ny+j]=sum0*dDdB1/Norm;
         if(d1==d) 
         {
            if(i-ik>=0 && i-ik<Nx && j-il>=0 && j-il<Ny)
               dDdF[i*Ny+j]=sum0*act_d(hin[m*Nx*Ny+(i-ik)*Nx+(j-il)])/Norm;
            dDdP[i*Ny+j]=sum0/Norm;
         }
      }
      dDdC[i*Ny+j]=dDdC2;
      dDdB[i*Ny+j]=dDdB2;
   }
}
//#define ksize 10
__global__ void backprop_parallel_CF(int Nx, int Ny, int dD, int dM, int Nk, int Nl, 
                                 float Norm, int ak, int al, int m, int d, int ik, int il, 
                                 float* __restrict__ const in, float* __restrict__ const out, 
                                 float* __restrict__ const hin, float* __restrict__ const f,
                                 float* __restrict__ dDdC, float* __restrict__ dDdF)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int j = blockDim.y * blockIdx.y + threadIdx.y;
   //int d1 = blockDim.z * blockIdx.z + threadIdx.z;
   if(i<Nx && j< Ny)// && d1<dD)
   {
      float dDdC2=0;
      //__device__ float hint[ksize];
      //__device__ float intt[ksize];
      //float hint[Nk*Nl];
      //float intt[Nk*Nl];
      #pragma unroll
      for(int d1=0;d1<dD;d1++)
      {
         float dDdC1=0;
         for(int k1=0;k1<Nk;k1++)   //sum
         {
            int ik1=-2*ak-1+k1;
            for(int l1=0;l1<Nl;l1++)   //sum
            {
               int il1=-2*al-1+l1;
               if(i-ik1>=0 && i-ik1<Nx && j-il1>=0 && j-il1<Ny)
               {
                  //if(d1==0) hint[k1*Nl+l1]=hin[m*Nx*Ny+(i-ik1)*Ny+(j-il1)];
                  float ft=f[d1*dM*Nk*Nl+m*Nk*Nl+k1*Nl+l1];
                  //float prod=ft*act1_d(hint[k1*Nl+l1]);
                  float prod=ft*act1_d(hin[m*Nx*Ny+(i-ik1)*Ny+(j-il1)]);
                  if(i-ik1-ik>=0 && i-ik1-ik<Nx && j-il1-il>=0 && j-il1-il<Ny)
                  {
                     //if(d1==0) intt[k1*Nl+l1]=in[d*Nx*Ny+(i-ik1-ik)*Ny+(j-il1-il)];
                     //dDdC1+=prod*intt[k1*Nl+l1];
                     dDdC1+=prod*in[d*Nx*Ny+(i-ik1-ik)*Ny+(j-il1-il)];
                  }
               }
            }
         }
         float outp=out[d1*Nx*Ny+i*Ny+j];
         float sum0=(outp-in[d1*Nx*Ny+i*Ny+j])*act1_d(outp);
         dDdC2+=sum0*dDdC1/Norm;
         //dDdC[d1*Nx*Ny+i*Ny+j]=sum0*dDdC1/Norm;
         if(d1==d) 
         {
            if(i-ik>=0 && i-ik<Nx && j-il>=0 && j-il<Ny)
               dDdF[i*Ny+j]=sum0*act_d(hin[m*Nx*Ny+(i-ik)*Nx+(j-ik)])/Norm;
         }
      }
      dDdC[i*Ny+j]=dDdC2;
   }
}

//Learn convolutional autoencoder by backpropagation
void backprop_gpu(vector<vector<vector<float> > >& in, vector<vector<vector<float> > >& out, vector<vector<vector<float> > >& hin ,vector<vector<vector<vector<float> > > >& c, vector<float>& b, vector<vector<vector<vector<float> > > >& f, vector<float>& p, vector<vector<vector<vector<float> > > >& dc, vector<float>& db, vector<vector<vector<vector<float> > > >& df, vector<float>& dp, vector<vector<vector<vector<float> > > >& ddc, vector<float>& ddb, vector<vector<vector<vector<float> > > >& ddf, vector<float>& ddp, float delmax, float alpha, int active)
{
   //float del=0.001;
   //int stride=1;
   int Nx=in[0].size();
   int Ny=in[0][0].size();
   int dM=c.size();
   int dD=c[0].size();
   int Nk=c[0][0].size();
   int Nl=c[0][0][0].size();
   int ak=((Nk-1)/2-1)/2;
   int al=((Nl-1)/2-1)/2;
   float Norm=(dD*dM*Nk*Nl*Nx*Ny);
   float dist=0;

   //thrust vectors
   thrust::host_vector<float> in_h(dD*Nx*Ny), out_h(dD*Nx*Ny), hin_h(dM*Nx*Ny);
   thrust::host_vector<float> f_h(dD*dM*Nk*Nl);
   for(int i=0;i<Nx;i++)
   {
      for(int j=0;j<Ny;j++)
      {
         for(int d=0;d<dD;d++)
         {
            in_h[d*Nx*Ny+i*Ny+j]=in[d][i][j];
            out_h[d*Nx*Ny+i*Ny+j]=out[d][i][j];
         }
         for(int m=0;m<dM;m++)
            hin_h[m*Nx*Ny+i*Ny+j]=hin[m][i][j];
      }
   }
   for(int d=0;d<dD;d++)
   {
      for(int m=0;m<dM;m++)
      {
         for(int k=0;k<Nk;k++)
         {
            for(int l=0;l<Nl;l++)      
               f_h[d*dM*Nk*Nl+m*Nk*Nl+k*Nl+l]=f[d][m][k][l];
         }
      }
   }
   thrust::device_vector<float> in_d(in_h), out_d(out_h), hin_d(hin_h);
   thrust::device_vector<float> f_d(f_h);
   thrust::device_vector<float> dDdC_d(Nx*Ny),dDdF_d(Nx*Ny),dDdB_d(Nx*Ny),dDdP_d(Nx*Ny); 
   //cast thrust vectors to pointer to be processed by kernel
   float* Cin_d   = thrust::raw_pointer_cast( &in_d[0]   );
   float* Cout_d  = thrust::raw_pointer_cast( &out_d[0]  );
   float* Chin_d  = thrust::raw_pointer_cast( &hin_d[0]  );
   float* Cf_d    = thrust::raw_pointer_cast( &f_d[0] );
   float* CdDdC_d = thrust::raw_pointer_cast( &dDdC_d[0] );
   float* CdDdF_d = thrust::raw_pointer_cast( &dDdF_d[0] );
   float* CdDdB_d = thrust::raw_pointer_cast( &dDdB_d[0] );
   float* CdDdP_d = thrust::raw_pointer_cast( &dDdP_d[0] );
   //calculate mse difference
   for(int d1=0;d1<dD;d1++)
   {
      for(int i=0;i<Nx;i++)
      {
         for(int j=0;j<Ny;j++)
         {
            dist+=pow(in[d1][i][j]-out[d1][i][j],2);
         }
      }
   }
   dist=dist/Norm;
   cout<<"mse: "<<dist<<endl;
//   ofstream file;
//   file.open("./mse.txt", ios::out|ios::app);
//   file<<setprecision(9)<<dist<<"\n"; 
//   file.close();
   //start backpropagation 
   for(int m=0;m<dM;m++)
   {
      for(int d=0;d<dD;d++)
      {
         for(int k=0;k<Nk;k++)
         {
            int ik=-2*ak-1+k;
            for(int l=0;l<Nl;l++)
            {
               int il=-2*al-1+l;
               //call kernel function to run on gpu
               dim3 threads(8,8);
               dim3 blocks(Nx/threads.x+1,Ny/threads.y+1);
               if(k==0 && l==0)               
                  backprop_parallel_CFBP<<<blocks,threads>>>(Nx,Ny,dD,dM,Nk,Nl,Norm,
                                                 ak,al,m,d,ik,il,
                                                 Cin_d, Cout_d, Chin_d, Cf_d,
                                                 CdDdC_d, CdDdF_d, CdDdB_d, CdDdP_d);
               else
                  backprop_parallel_CF<<<blocks,threads>>>(Nx,Ny,dD,dM,Nk,Nl,Norm,
                                                 ak,al,m,d,ik,il,
                                                 Cin_d, Cout_d, Chin_d, Cf_d,
                                                 CdDdC_d, CdDdF_d);
               //sum the parallel results obtained in the kernel to get the gradient
               float dDdC = thrust::reduce(dDdC_d.begin(), dDdC_d.end());
               float dDdF = thrust::reduce(dDdF_d.begin(), dDdF_d.end());
               //update the synaptic values with inertia and adaptive learning rate
               float del=delmax;
               adapt_rate(del, delmax, active, dDdC, ddc[m][d][k][l], dc[m][d][k][l]);
               dc[m][d][k][l]=(1-alpha)*del*dDdC/((10<abs(dDdC))?abs(dDdC):10)+alpha*dc[m][d][k][l];
               adapt_rate(del, delmax, active, dDdF, ddf[d][m][k][l], df[d][m][k][l]);
               df[d][m][k][l]=(1-alpha)*del*dDdF/((10<abs(dDdF))?abs(dDdF):10)+alpha*df[d][m][k][l];
               c[m][d][k][l]+=-dc[m][d][k][l];
               f[d][m][k][l]+=-df[d][m][k][l];
               if(k==0 && l==0)
               {
                  if(d==0) 
                  {
                     float dDdB = thrust::reduce(dDdB_d.begin(), dDdB_d.end());
                     adapt_rate(del, delmax, active, dDdB, ddb[m], db[m]);
                     db[m]=(1-alpha)*del*dDdB/((10<abs(dDdB))?abs(dDdB):10)+alpha*db[m];
                     b[m]+=-db[m];
                  }
                  if(m==0)
                  {
                     float dDdP = thrust::reduce(dDdP_d.begin(), dDdP_d.end());
                     adapt_rate(del, delmax, active, dDdP, ddp[d], dp[d]);
                     dp[d]=(1-alpha)*del*dDdP/((10<abs(dDdP))?abs(dDdP):10)+alpha*dp[d];
                     p[d]+=-dp[d];
                  }
               }
            }
         }
      }
   }
}



////////////////////////////////////////////////////////////////////////////////////////

__global__ void backprop_parallel_CCBP(int Nx, int Ny, int dD, int dM, int Nk, int Nl, float Norm, int ak, int al,
                                 int m, int d, int ik, int il, 
                                 float *in, float *out, float *hin, float *f,
                                 float *dDdC, float *dDdB, float *dDdP)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int j = blockDim.y * blockIdx.y + threadIdx.y;
   //int d1 = blockDim.z * blockIdx.z + threadIdx.z;
   if(i<Nx && j< Ny)
   {
      float dDdC2=0;
      float dDdB2=0;
      for(int d1=0;d1<dD;d1++)
      {
         float dDdB1=0;
         float dDdC1=0;
         for(int k1=0;k1<Nk;k1++)   //sum
         {
            int ik1=-2*ak-1+k1;
            for(int l1=0;l1<Nl;l1++)   //sum
            {
               int il1=-2*al-1+l1;
               if(i-ik1>=0 && i-ik1<Nx && j-il1>=0 && j-il1<Ny)
               {
                  float prod=f[d1*dM*Nk*Nl+m*Nk*Nl+k1*Nl+l1]*act1_d(hin[m*Nx*Ny+(i-ik1)*Ny+(j-il1)]);
                  dDdB1+=prod;
                  if(i-ik1-ik>=0 && i-ik1-ik<Nx && j-il1-il>=0 && j-il1-il<Ny)
                     dDdC1+=prod*in[d*Nx*Ny+(i-ik1-ik)*Ny+(j-il1-il)];
               }
            }
         }
         float sum0=(out[d1*Nx*Ny+i*Ny+j]-in[d1*Nx*Ny+i*Ny+j])*act1_d(out[d1*Nx*Ny+i*Ny+j]);
         dDdC2+=sum0*dDdC1/Norm;
         dDdB2+=sum0*dDdB1/Norm;
         //dDdC[d1*Nx*Ny+i*Ny+j]=sum0*dDdC1/Norm;
         //dDdB[d1*Nx*Ny+i*Ny+j]=sum0*dDdB1/Norm;
         if(d1==d) 
         {
            if(i-ik>=0 && i-ik<Nx && j-il>=0 && j-il<Ny)
            {
               float dDdF=sum0*act_d(hin[m*Nx*Ny+(i-ik)*Nx+(j-il)])/Norm;
               //dDdC[d1*Nx*Ny+i*Ny+j]+=dDdF;
               dDdC2+=dDdF;
            }
            dDdP[i*Ny+j]=sum0/Norm;
         }
      }
      dDdC[i*Ny+j]=dDdC2;
      dDdB[i*Ny+j]=dDdB2;
   }
}

__global__ void backprop_parallel_CC(int Nx, int Ny, int dD, int dM, int Nk, int Nl, float Norm, int ak, int al,
                                 int m, int d, int ik, int il, 
                                 float *in, float *out, float *hin, float *f,
                                 float *dDdC)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int j = blockDim.y * blockIdx.y + threadIdx.y;
   if(i<Nx && j< Ny)
   {
      float dDdC2=0;
      for(int d1=0;d1<dD;d1++)
      {
         float dDdC1=0;
         for(int k1=0;k1<Nk;k1++)   //sum
         {
            int ik1=-2*ak-1+k1;
            for(int l1=0;l1<Nl;l1++)   //sum
            {
               int il1=-2*al-1+l1;
               if(i-ik1>=0 && i-ik1<Nx && j-il1>=0 && j-il1<Ny)
               {
                  float prod=f[d1*dM*Nk*Nl+m*Nk*Nl+k1*Nl+l1]*act1_d(hin[m*Nx*Ny+(i-ik1)*Ny+(j-il1)]);
                  if(i-ik1-ik>=0 && i-ik1-ik<Nx && j-il1-il>=0 && j-il1-il<Ny)
                     dDdC1+=prod*in[d*Nx*Ny+(i-ik1-ik)*Ny+(j-il1-il)];
               }
            }
         }
         float sum0=(out[d1*Nx*Ny+i*Ny+j]-in[d1*Nx*Ny+i*Ny+j])*act1_d(out[d1*Nx*Ny+i*Ny+j]);
         //dDdC[d1*Nx*Ny+i*Ny+j]=sum0*dDdC1/Norm;
         dDdC2+=sum0*dDdC1/Norm;
         if(d1==d) 
         {
            if(i-ik>=0 && i-ik<Nx && j-il>=0 && j-il<Ny)
            {
               float dDdF=sum0*act_d(hin[m*Nx*Ny+(i-ik)*Nx+(j-il)])/Norm;
               //dDdC[d1*Nx*Ny+i*Ny+j]+=dDdF;
               dDdC2+=dDdF;
            }
         }
      }
      dDdC[i*Ny+j]=dDdC2;
   }
}

//Learn convolutional autoencoder by backpropagation with simmetric weights
void backprop_gpu_cc(vector<vector<vector<float> > >& in, vector<vector<vector<float> > >& out, vector<vector<vector<float> > >& hin ,vector<vector<vector<vector<float> > > >& c, vector<float>& b, vector<vector<vector<vector<float> > > >& f, vector<float>& p, vector<vector<vector<vector<float> > > >& dc, vector<float>& db, vector<vector<vector<vector<float> > > >& df, vector<float>& dp, vector<vector<vector<vector<float> > > >& ddc, vector<float>& ddb, vector<vector<vector<vector<float> > > >& ddf, vector<float>& ddp, float delmax, float alpha, int active)
{
   //float del=0.001;
   //int stride=1;
   int Nx=in[0].size();
   int Ny=in[0][0].size();
   int dM=c.size();
   int dD=c[0].size();
   int Nk=c[0][0].size();
   int Nl=c[0][0][0].size();
   int ak=((Nk-1)/2-1)/2;
   int al=((Nl-1)/2-1)/2;
   float Norm=(2*dD*dM*Nk*Nl*Nx*Ny);
   float dist=0;

   //thrust vectors
   thrust::host_vector<float> in_h(dD*Nx*Ny), out_h(dD*Nx*Ny), hin_h(dM*Nx*Ny);
   thrust::host_vector<float> f_h(dD*dM*Nk*Nl);
   for(int i=0;i<Nx;i++)
   {
      for(int j=0;j<Ny;j++)
      {
         for(int d=0;d<dD;d++)
         {
            in_h[d*Nx*Ny+i*Ny+j]=in[d][i][j];
            out_h[d*Nx*Ny+i*Ny+j]=out[d][i][j];
         }
         for(int d=0;d<dM;d++)
            hin_h[d*Nx*Ny+i*Ny+j]=hin[d][i][j];
      }
   }
   for(int d=0;d<dD;d++)
   {
      for(int m=0;m<dM;m++)
      {
         for(int k=0;k<Nk;k++)
         {
            for(int l=0;l<Nl;l++)      
            {
               f_h[d*dM*Nk*Nl+m*Nk*Nl+k*Nl+l]=f[d][m][k][l];
            }
         }
      }
   }
   thrust::device_vector<float> in_d(in_h), out_d(out_h), hin_d(hin_h);
   thrust::device_vector<float> f_d(f_h);
   thrust::device_vector<float> dDdC_d(Nx*Ny),dDdB_d(Nx*Ny),dDdP_d(Nx*Ny); 
   //cast thrust vectors to pointer to be processed by kernel
   float* Cin_d   = thrust::raw_pointer_cast( &in_d[0]   );
   float* Cout_d  = thrust::raw_pointer_cast( &out_d[0]  );
   float* Chin_d  = thrust::raw_pointer_cast( &hin_d[0]  );
   float* Cf_d    = thrust::raw_pointer_cast( &f_d[0] );
   float* CdDdC_d = thrust::raw_pointer_cast( &dDdC_d[0] );
   float* CdDdB_d = thrust::raw_pointer_cast( &dDdB_d[0] );
   float* CdDdP_d = thrust::raw_pointer_cast( &dDdP_d[0] );
   //calculate mse difference
   for(int d1=0;d1<dD;d1++)
   {
      for(int i=0;i<Nx;i++)
      {
         for(int j=0;j<Ny;j++)
         {
            dist+=pow(in[d1][i][j]-out[d1][i][j],2);
         }
      }
   }
   dist=dist/Norm;
   cout<<"mse: "<<dist<<endl;
//   ofstream file;
//   file.open("./mse.txt", ios::out|ios::app);
//   file<<setprecision(9)<<dist<<"\n"; 
//   file.close();
   //start backpropagation 
   for(int m=0;m<dM;m++)
   {
      for(int d=0;d<dD;d++)
      {
         for(int k=0;k<Nk;k++)
         {
            int ik=-2*ak-1+k;
            for(int l=0;l<Nl;l++)
            {
               int il=-2*al-1+l;
               //call kernel function to run on gpu
               dim3 threads(8,8);
               dim3 blocks(Nx/threads.x+1,Ny/threads.y+1);
               if(k==0 && l==0)               
                  backprop_parallel_CCBP<<<blocks,threads>>>(Nx,Ny,dD,dM,Nk,Nl,Norm,ak,al,m,d,ik,il,
                                                 Cin_d, Cout_d, Chin_d, Cf_d,
                                                 CdDdC_d, CdDdB_d, CdDdP_d);
               else
                  backprop_parallel_CC<<<blocks,threads>>>(Nx,Ny,dD,dM,Nk,Nl,Norm,ak,al,m,d,ik,il,
                                                 Cin_d, Cout_d, Chin_d, Cf_d,
                                                 CdDdC_d);
               //sum the parallel results obtained in the kernel to get the gradient
               float dDdC = thrust::reduce(dDdC_d.begin(), dDdC_d.end());
               //update the synaptic values with inertia and adaptive learning rate
               float del=delmax;
               adapt_rate(del, delmax, active, dDdC, ddc[m][d][k][l], dc[m][d][k][l]);
               dc[m][d][k][l]=(1-alpha)*del*dDdC/((10<abs(dDdC))?abs(dDdC):10)+alpha*dc[m][d][k][l];
               c[m][d][k][l]+=-dc[m][d][k][l];
               f[d][m][k][l]=c[m][d][k][l];
               if(k==0 && l==0)
               {
                  if(d==0) 
                  {
                     float dDdB = thrust::reduce(dDdB_d.begin(), dDdB_d.end());
                     adapt_rate(del, delmax, active, dDdB, ddb[m], db[m]);
                     db[m]=(1-alpha)*del*dDdB/((10<abs(dDdB))?abs(dDdB):10)+alpha*db[m];
                     b[m]+=-db[m];
                  }
                  if(m==0)
                  {
                     float dDdP = thrust::reduce(dDdP_d.begin(), dDdP_d.end());
                     adapt_rate(del, delmax, active, dDdP, ddp[d], dp[d]);
                     dp[d]=(1-alpha)*del*dDdP/((10<abs(dDdP))?abs(dDdP):10)+alpha*dp[d];
                     p[d]+=-dp[d];
                  }
               }
            }
         }
      }
   }
}
