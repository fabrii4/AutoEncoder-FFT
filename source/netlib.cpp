#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <bitset>
#include <iostream>
#include <fstream> 
#include<string>
#include<vector>
#include<ctime>
#include<cmath>
#include <algorithm>
#include <unistd.h>
#include<fstream>
#include <chrono>

#include "backproplib.h"

using namespace cv;
using namespace std;


//float act(float x) //activation function
//{
//   float a=0.01;
//   if(x>0) return x;
//   else return a*x;
//}
//float act1(float x) //activation function derivative
//{
//   float a=0.01;
//   if(x>0) return 1;
//   else return a;
//}

//convert color image to vector spin configuration
void ImageToSpin_C(Mat& img, vector<vector<vector<float> > >& spin)
{
   int Nx=img.cols;
   int Ny=img.rows;
   for(int i=0;i<Nx;i++)
   {
      for(int j=0;j<Ny;j++)
      {
         Vec3b col=img.at<Vec3b>(j,i);
         spin[0][i][j]=(float)col[0];///255.; //B
         spin[1][i][j]=(float)col[1];///255.; //G
         spin[2][i][j]=(float)col[2];///255.; //R
      }
   }
}

//convert vector spin configuration to color image
void SpinToImage_C(Mat& img, vector<vector<vector<float> > >& spin)
{
   int Nx=spin[0].size();
   int Ny=spin[0][0].size();
   for(int i=0;i<Nx;i++)
   {
      for(int j=0;j<Ny;j++)
      {
         Vec3b col;
         col[0]=(int)(spin[0][i][j]);//*255); //B
         col[1]=(int)(spin[1][i][j]);//*255); //G
         col[2]=(int)(spin[2][i][j]);//*255); //R
         img.at<Vec3b>(j,i)=col;
      }
   }
}

//convert vector spin configuration to scalar image
void SpinToImage_V(Mat& img, vector<vector<float> >& spin)
{
   int Nx=spin.size();
   int Ny=spin[0].size();
   for(int i=0;i<Nx;i++)
   {
      for(int j=0;j<Ny;j++)
      {
         int intens=1*(int)spin[i][j];
         img.at<uchar>(j,i)=intens;
      }
   }
}

//convert convolutional kernel configuration to scalar image
void SpinToImage_K(Mat& img, vector<vector<float> >& spin)
{
   int Nx=spin.size();
   int Ny=spin[0].size();
   for(int i=0;i<Nx;i++)
   {
      for(int j=0;j<Ny;j++)
      {
         int intens=100*spin[i][j];
         if(intens>0) intens+=128;
         else intens=128-intens;
         img.at<uchar>(j,i)=intens;
      }
   }
}

//Resize pattern (max Pooling)
void Pool(vector<vector<vector<float> > >& in, vector<vector<vector<float> > >& out, int scale)
{
   int D=in.size();
   if(scale>0) //reduce size
   {
      int Nx=in[0].size();
      int Ny=in[0][0].size();
      for(int d=0;d<D;d++)
      {
         for(int i=0;i<Nx;i+=scale)
         {
            for(int j=0;j<Ny;j+=scale)
            {
               int smax=0;
               for(int k=0;k<scale;k++)
               {
                  for(int l=0;l<scale;l++)
                  {
                     if(i+k<Nx && j+l<Ny && in[d][i+k][j+l]>smax)
                        smax=in[d][i+k][j+l];
                  }
               }
               out[d][i/scale][j/scale]=smax;
            }
         }
      }
   }
   else //increase size for negative scale
   {
      int Nx=out[0].size();
      int Ny=out[0][0].size();
      scale=-scale;
      for(int d=0;d<D;d++)
      {
         for(int i=0;i<Nx;i+=scale)
         {
            for(int j=0;j<Ny;j+=scale)
            {
               for(int k=0;k<scale;k++)
               {
                  for(int l=0;l<scale;l++)
                  {
                     if(i+k<Nx && j+l<Ny)
                        out[d][i+k][j+l]=in[d][i/scale][j/scale];
                  }
               }
            }
         }
      }
   }
}

//init Convolutional layer
void Init_conv(vector<vector<vector<vector<float > > > >& c, vector<float>& b, int mS, int dD, int kS, int lS, float max)
{
   c.clear();
   b.clear();
   int Nk=kS;
   int Nl=lS;
   for(int m=0;m<mS;m++)
   {
      vector<vector<vector<float> > > cd;
      for(int d=0;d<dD;d++)
      {
         vector<vector<float> > ck;
         for(int k=0;k<Nk;k++)
         {
            vector<float> cl;
            for(int l=0;l<Nl;l++)
            {
               //float r=2*(rand()%2)-1;
               float r= -max + 2*max*(float)rand()/(float)RAND_MAX;
               cl.push_back(r);
            }
            ck.push_back(cl);
         }
         cd.push_back(ck);
      }
      c.push_back(cd);
      //float r=2*(rand()%2)-1;
      float r= -max + 2*max*(float)rand()/(float)RAND_MAX;
      b.push_back(r);
   }
}

//Save/Load vector
void SaveLoad_vec(vector<float>& vec, string path, int write)
{
   if(write==1)
   {
      ofstream file;
      file.open(path, ios::out | ios::binary);
      //copy(vec.begin(), vec.end(), ostreambuf_iterator<char>(file));
      file.write(reinterpret_cast<char*>(&vec[0]), vec.size()*sizeof(float)); 
      file.close();
   }
   else
   {
      ifstream file;
      file.open(path, ios::in | ios::binary);
      file.read(reinterpret_cast<char*>(&vec[0]), vec.size()*sizeof(float));
      file.close();
   }
}

//Save/Load Convolutional layer
void SaveLoad_conv(vector<vector<vector<vector<float> > > >& c, vector<float>& b, int L, int io, int write)
{
   int dM=c.size();
   int dD=c[0].size();
   int Nk=c[0][0].size();
   int Nl=c[0][0][0].size();
   vector<float> vec(dM*dD*Nk*Nl+dM);
   string inout;
   if(io==0) inout="_in";
   else inout="_out";
   string path("./weights/C_weights_");
   path = path+to_string(L)+inout+\
         "_D="+to_string(dD)+"_M="+to_string(dM)+\
         "_Lk="+to_string(((Nk-1)/2-1)/2)+"_Ll="+to_string(((Nl-1)/2-1)/2)+".conv";
   cout<<"path "<<L<<inout<<" "<<path<<endl;
   if(write==1) //Save
   {
      for(int m=0;m<dM;m++)
      {
         for(int d=0;d<dD;d++)
         {
            for(int k=0;k<Nk;k++)
            {
               for(int l=0;l<Nl;l++)      
               {
                  vec[m*dD*Nk*Nl+d*Nk*Nl+k*Nl+l]=c[m][d][k][l];
               }
            }
         }
         vec[dM*dD*Nk*Nl+m]=b[m];
      }
      SaveLoad_vec(vec, path, 1);
   }
   else //Load
   {
      SaveLoad_vec(vec, path, 0);
      for(int m=0;m<dM;m++)
      {
         for(int d=0;d<dD;d++)
         {
            for(int k=0;k<Nk;k++)
            {
               for(int l=0;l<Nl;l++)      
               {
                  c[m][d][k][l]=vec[m*dD*Nk*Nl+d*Nk*Nl+k*Nl+l];
               }
            }
         }
         b[m]=vec[dM*dD*Nk*Nl+m];
      }
   }
}

void LoadParam(int& dM, int& Lk, int& Ll, int& scal, float& rmax)
{
   vector<int> values;
   int param_value;
   string param_name;
   ifstream file("New_Layer_Param.txt");
   while ( file >> param_name >> param_value )
   {
       values.push_back(param_value);
   }
   dM=values[0];
   Lk=values[1];
   Ll=values[2];
   scal=values[3];
   rmax=values[4];
}

//Use only small portion of patterns for backpropagation
void Portion(vector<vector<vector<float> > >& in, vector<vector<vector<float> > >& hin, vector<vector<vector<float> > >& out, vector<vector<vector<float> > >& in_s, vector<vector<vector<float> > >& hin_s, vector<vector<vector<float> > >& out_s, int q)
{
   int Nx=in[0].size();
   int Ny=in[0][0].size();
   int D=in.size();
   int M=hin.size();
   int dx=(Nx-Nx/q)/2;
   int dy=(Ny-Ny/q)/2;
   //dx+=-dx + rand()%(2*dx + 1);
   //dy+=-dy + rand()%(2*dy + 1);
   for(int i=0;i<Nx/q;i++)
   {
      for(int j=0;j<Ny/q;j++)
      {
         for(int d=0;d<D;d++)
         {
            in_s[d][i][j]=in[d][i+dx][j+dy];
            out_s[d][i][j]=out[d][i+dx][j+dy];
         }
         for(int d=0;d<M;d++)
            hin_s[d][i][j]=hin[d][i+dx][j+dy];
      }
   }
}

//Convolution
void Conv(vector<vector<vector<float> > >& in, vector<vector<vector<float> > >& out, vector<vector<vector<vector<float > > > >& c, vector<float>& b)
{
   int stride=1;
   int Nx=in[0].size();
   int Ny=in[0][0].size();
   int Nk=c[0][0].size();
   int Nl=c[0][0][0].size();
   int ak=((Nk-1)/2-1)/2;
   int al=((Nl-1)/2-1)/2;
   int dM=c.size();
   int dD=c[0].size();
   for(int m=0;m<dM;m++)
   {
      for(int i=0;i<Nx;i+=stride)
      {
         for(int j=0;j<Ny;j+=stride)
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
                     if(i-ik>0 && i-ik<Nx && j-il>0 && j-il<Ny)
                     {
                        h+=c[m][d][k][l]*in[d][i-ik][j-il];
                     }
                     il+=1;
                  }
                  ik+=1;
               }
            }
            h+=b[m];
            out[m][i][j]=act(h); 
         }
      }    
   }
}

//Learn convolutional autoencoder by backpropagation
void backprop(vector<vector<vector<float> > >& in, vector<vector<vector<float> > >& out, vector<vector<vector<float> > >& hin, vector<vector<vector<vector<float > > > >& c, vector<float>& b, vector<vector<vector<vector<float > > > >& f, vector<float>& p, float del)
{
   //float del=0.001;
   int stride=1;
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
   cout<<"mse: "<<dist<<endl;
   //start
   for(int m=0;m<dM;m++)   //parallel
   {
      for(int d=0;d<dD;d++)   //parallel
      {
         int ik=-2*ak-1;
         for(int k=0;k<Nk;k++)   //parallel
         {
            int il=-2*al-1;
            for(int l=0;l<Nl;l++)   //parallel
            {
               float dDdC=0, dDdF=0, dDdB=0, dDdP=0;
               for(int d1=0;d1<dD;d1++)   //sum
               {
                  for(int i=0;i<Nx;i++)   //sum
                  {
                     for(int j=0;j<Ny;j++)   //sum
                     {
                        float dDdB1=0;
                        float dDdC1=0;
                        int ik1=-2*ak-1;
                        for(int k1=0;k1<Nk;k1++)   //sum
                        {
                           int il1=-2*al-1;
                           for(int l1=0;l1<Nl;l1++)   //sum
                           {
                              if(i-ik1>0 && i-ik1<Nx && j-il1>0 && j-il1<Ny)
                              {
                                 float prod=f[d1][m][k1][l1]*act1(hin[m][i-ik1][j-il1]);
                                 dDdB1+=prod;
                                 if(i-ik1-ik>0 && i-ik1-ik<Nx && j-il1-il>0 && j-il1-il<Ny)
                                    dDdC1+=prod*in[d][i-ik1-ik][j-il1-il];
                              }
                              il1+=1;
                           }
                           ik1+=1;
                        }
                        float sum0=(out[d1][i][j]-in[d1][i][j])*act1(out[d1][i][j]);
                        dDdC+=sum0*dDdC1/Norm;
                        dDdB+=sum0*dDdB1/Norm;
                        if(d1==d) 
                        {
                           if(i-ik>0 && i-ik<Nx && j-il>0 && j-il<Ny)
                              dDdF+=sum0*act(hin[m][i-ik][j-il])/Norm;
                           dDdP+=sum0/Norm;
                        }
                     }
                  }
               }
               //if(m==1 && d==1 && k==0 && l==0)
               //   cout<<"dDdC cpu "<<dDdC<<" "<<dDdB<<" "<<dDdF<<" "<<dDdP<<endl;
               c[m][d][k][l]+=-del*dDdC/((10<abs(dDdC))?abs(dDdC):10);
               f[d][m][k][l]+=-del*dDdF/((10<abs(dDdF))?abs(dDdF):10);
               cout<<m<<"|"<<dM<<" "<<d<<"|"<<dD<<" "<<k<<"|"<<Nk<<" "<<l<<"|"<<Nl<<" "<<c[m][d][k][l]<<endl;
               if(k==0 && l==0)
               {
                  if(d==0) b[m]+=-del*dDdB/((10<abs(dDdB))?abs(dDdB):10);
                  if(m==0) p[d]+=-del*dDdP/((10<abs(dDdP))?abs(dDdP):10);
               }
               il+=1;
            }
            ik+=1;
         }
      }
   }
}

