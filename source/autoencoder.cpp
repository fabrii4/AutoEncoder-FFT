#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <bitset>
#include <iostream>
#include<string>
#include<vector>
#include<ctime>
#include<cmath>
#include <algorithm>
#include <unistd.h>
#include<fstream>
#include <chrono>

#include "netlib.h"
#include "backproplib.h"
#include "fft_backproplib.h"

using namespace cv;
using namespace std;


int main()
{
   //video size
   int Nx=256, Ny=256;
   //input depth
   int D=3;
   //convolution depth
   int M=50;
   //convolution size |k|<2L+1
   int Lk=0, Ll=0;
   int Nk=2*(2*Lk+1)+1;
   int Nl=2*(2*Ll+1)+1;
   //pooling scale
   int s=1;
   //max value in kernels initialization
   float rmax=1.;
   //Load parameters from file
   LoadParam(M,Lk,Ll,s,rmax);
   Nk=2*(2*Lk+1)+1;
   Nl=2*(2*Ll+1)+1;

   //opencv init
   Mat rgb;
   Mat imgC(Size(Nx,Ny),CV_8UC3);
   VideoCapture cam(0);
   namedWindow("input",CV_WINDOW_NORMAL);
   moveWindow("input",100,100);
   resizeWindow("input", 200,200);
   namedWindow("output",CV_WINDOW_NORMAL);
   moveWindow("output",400,100);
   resizeWindow("output", 200,200);
   namedWindow("output1",CV_WINDOW_NORMAL);
   moveWindow("output1",100,400);
   resizeWindow("output1", 200,200);
   namedWindow("output2",CV_WINDOW_NORMAL);
   moveWindow("output2",400,400);
   resizeWindow("output2", 200,200);

   //Convolutional and pooling layers
   vector<vector<vector<float> > > in(D, vector<vector<float> >(Nx, vector<float>(Ny)));
   vector<vector<vector<float> > > Pin(D, vector<vector<float> >(Nx/s, vector<float>(Ny/s)));
   vector<vector<vector<float> > > hC(M, vector<vector<float> >(Nx/s, vector<float>(Ny/s)));
   vector<vector<vector<float> > > PhC(D, vector<vector<float> >(Nx/s, vector<float>(Ny/s)));
   vector<vector<vector<float> > > out(D, vector<vector<float> >(Nx, vector<float>(Ny)));
   vector<vector<vector<vector<float> > > > layers;
   //convolutional filters
   vector<vector<vector<vector<float> > > > c, f, dc, df, ddc, ddf;
   vector<vector<vector<vector<vector<float> > > > > net_c;
   vector<vector<float> > net_cfreq;
   //bias
   vector<float> b(M), p(D), db(M), dp(D), ddb(M), ddp(D);  
   vector<vector<float> > net_b;
   //layers info vectors
   vector<int> scale;
   //keyboard controls
   int sel=0; //backpropagation activation
   int q=1; //scale of the backpropagation input
   float del=0.2, ddel=0.1; //max learning rate and keyboard step
   int active=1; //active learning rate activation 
   float alpha=0.9; //inertia weight
   int feat=0; //select feature map to display
   int n_l=0;  //select active layer
   int gpu=1;  //activate gpu
   int sym=0; //symmetric weights
   int fft=0; //activate fft convolution

   //Init random convolutional filters
   srand(time(0));
   Init_conv(c, b, M, D, Nk, Nl, rmax);
   Init_conv(f, p, D, M, Nk, Nl, rmax);
   //Init vectors of previous weight update dc= c(t-1)-c(t-2) (used in inertia update)
   Init_conv(dc, db, M, D, Nk, Nl, 0);
   Init_conv(df, dp, D, M, Nk, Nl, 0);
   //Init vectors of previous gradient values ddc=grad(c)(t-1) (used in adaptive learning rate)
   Init_conv(ddc, ddb, M, D, Nk, Nl, 0);
   Init_conv(ddf, ddp, D, M, Nk, Nl, 0);

   //Push layers into network
   layers.push_back(in);
   layers.push_back(Pin);
   layers.push_back(hC);
   layers.push_back(PhC);
   layers.push_back(out);
   net_c.push_back(c);
   net_c.push_back(f);
   net_b.push_back(b);
   net_b.push_back(p);
   scale.push_back(s);
   scale.push_back(-s);
//int count=0;
   while(true)
   {
//if(count==110) count=0;
//count++;
      cam>>rgb;
//string fname="../../tensorflow/img_train/train_"+to_string(count)+".jpg";
//cout<<fname<<endl;
//rgb=imread(fname, CV_LOAD_IMAGE_COLOR);
      resize(rgb,imgC,Size(Nx,Ny));
      ImageToSpin_C(imgC,layers[0]);
//int ix=67,iy=115;
//cout<<layers[0][0][ix][iy]<<" "<<layers[0][1][ix][iy]<<" "<<layers[0][2][ix][iy]<<endl;

      //apply coder-decoder convolutions
         auto start0 = std::chrono::high_resolution_clock::now();
      if(fft==1)
         autoenc_fft(layers, net_c, net_cfreq, net_b, scale);
      else
      {
         for(int n=0;n<net_c.size();n++)
         {
            int nl=2*n;
            if(n<net_c.size()/2)
            {
               Pool(layers[nl],layers[nl+1],scale[n]);
               //Conv(layers[nl+1], layers[nl+2], net_c[n], net_b[n]);
               Conv_gpu(layers[nl+1], layers[nl+2], net_c[n], net_b[n]);
            }
            else
            {
               Conv_gpu(layers[nl], layers[nl+1], net_c[n], net_b[n]);
               //Conv(layers[nl], layers[nl+1], net_c[n], net_b[n]);
               Pool(layers[nl+1],layers[nl+2],scale[n]);
            }
         }
      }
         auto finish0 = std::chrono::high_resolution_clock::now();
         std::chrono::duration<double> elapsed0 = finish0 - start0;
         //std::cout << "Convolution Time: " << elapsed0.count() << " s\r"<<flush;


//Conv_gpu(layers[0], layers[4], net_c[0], net_b[0]);

//for(int k=0;k<net_c[n_l][0][0].size();k++)
//{
//   for(int l=0;l<net_c[n_l][0][0][0].size();l++)
//   {
//      cout<<setprecision(3)<<net_c[net_c.size()-1-n_l][0][1][k][l]<<" ";
//   }
//   cout<<"         ";
//   for(int l=0;l<net_c[n_l][0][0][0].size();l++)
//   {
//      cout<<net_c[n_l][1][0][k][l]<<" ";
//   }
//   cout<<endl;
//}
//cout<<endl;

      //backpropagation
      if(sel==1)
      {
         auto start = std::chrono::high_resolution_clock::now();
         int dD=layers[2*n_l+1].size();
         int dNx=layers[2*n_l+1][0].size();
         int dNy=layers[2*n_l+1][0][0].size();
         int dM=layers[2*n_l+2].size();
         vector<vector<vector<float> > > in_s(dD, vector<vector<float> >(dNx/q, vector<float>(dNy/q)));
         vector<vector<vector<float> > > out_s(dD, vector<vector<float> >(dNx/q, vector<float>(dNy/q)));
         vector<vector<vector<float> > > hC_s(dM, vector<vector<float> >(dNx/q, vector<float>(dNy/q)));
         Portion(layers[2*n_l+1],layers[2*n_l+2],layers[layers.size()-2-2*n_l],in_s,hC_s,out_s,q);
         if(gpu==1 && fft==0) 
         {
            if(sym==0)
               backprop_gpu(in_s, out_s, hC_s,
                                 net_c[n_l], net_b[n_l], net_c[net_c.size()-1-n_l], 
                                 net_b[net_c.size()-1-n_l], 
                                 dc, db, df, dp, 
                                 ddc, ddb, ddf, ddp, 
                                 del, alpha, active);
            else
               backprop_gpu_cc(in_s, out_s, hC_s,
                                 net_c[n_l], net_b[n_l], net_c[net_c.size()-1-n_l], 
                                 net_b[net_c.size()-1-n_l], 
                                 dc, db, df, dp, 
                                 ddc, ddb, ddf, ddp, 
                                 del, alpha, active);
         }
         else if(gpu==1 && fft==1)
         {
            backprop_fft(layers[0], layers.back(), net_cfreq[n_l], net_c[n_l], 
                         net_cfreq[net_c.size()-1-n_l], net_c[net_c.size()-1-n_l], 
                         net_b[n_l], net_b[net_c.size()-1-n_l], dM);
            sel=0;
         }
         else backprop(in_s, out_s, hC_s, net_c[n_l], net_b[n_l], net_c[net_c.size()-1-n_l], 
                                 net_b[net_c.size()-1-n_l], del);
         auto finish = std::chrono::high_resolution_clock::now();
         std::chrono::duration<double> elapsed = finish - start;
         //std::cout << "Time: " << elapsed.count() << " s\n";
      }

//ix=126,iy=208;
//cout<<layers[layers.size()-1][0][ix][iy]<<" "<<layers[layers.size()-1][1][ix][iy]<<" "<<layers[layers.size()-1][2][ix][iy]<<endl;
       

      //show results
      int Nxo=layers[2*n_l][0].size();
      int Nyo=layers[2*n_l][0][0].size();
      int Nxh=layers[2*n_l+1][0].size();
      int Nyh=layers[2*n_l+1][0][0].size();
      //input module layer
      Mat output_in(Size(Nxo,Nyo),CV_8UC3);
      SpinToImage_C(output_in,layers[2*n_l]);
      imshow("input",output_in);
      //output module layer
      Mat output_out(Size(Nxo,Nyo),CV_8UC3);
      SpinToImage_C(output_out,layers[layers.size()-1-2*n_l]);
      imshow("output",output_out);
      //module feaute maps
      Mat output_h(Size(Nxh,Nyh),CV_8U);
      SpinToImage_V(output_h,layers[2*n_l+2][feat]);
      imshow("output1",output_h);
      //module features
      int dMo=net_c[n_l].size();
      int dDo=net_c[n_l][0].size();
      int Nko=net_c[n_l][0][0].size();
      int Nlo=net_c[n_l][0][0][0].size();
      Mat output_c(Size(dDo*Nko,Nlo),CV_8U);
      for(int i=0;i<dDo;i++)
      {
         Rect Rec(Nko*i, 0, Nko, Nlo);
         Mat Roi = output_c(Rec);
         SpinToImage_K(Roi,net_c[n_l][feat][i]);
         Roi.copyTo(output_c(Rec));
      }
      //normalize(output_c, output_c, 255, 0);
      imshow("output2",output_c);

      //keyboard controls
      char ch = cvWaitKey(10);
      if(27 == char(ch)) break;
      if(ch=='1') {sel=(sel+1)%2; cout<<"backpropagation "<<sel<<endl;}
      if(ch=='2') {q=q+1; cout<<"q                          "<<q<<endl;}
      if(ch=='3') {q=max(1,q-1); cout<<"q                          "<<q<<endl;}
      if(ch=='4') 
      {
         del=del+ddel; 
         if(del>0.1 && del<1) ddel=0.1;
         if(del>0.01 && del<0.1) ddel=0.01;
         if(del>0.001 && del<0.01) ddel=0.001; 
         if(del>0.0001 && del<0.001) ddel=0.0001;
         if(del>1) del=1; 
         cout<<"del                        "<<del<<endl;
      }
      if(ch=='5') 
      {
         del=del-ddel; 
         if(del>0.1 && del<=1) ddel=0.1;
         if(del>0.01 && del<=0.11) ddel=0.01;
         if(del>0.001 && del<=0.011) ddel=0.001; 
         if(del>0.0001 && del<=0.0011) ddel=0.0001;
         if(del<0) del=0; 
         cout<<"del                        "<<del<<endl;
      }
      if(ch=='6') {alpha=alpha+0.1; if(alpha>1) alpha=1; cout<<"alpha                        "<<alpha<<endl;}
      if(ch=='7') {alpha=alpha-0.1; if(alpha<0) alpha=0; cout<<"alpha                        "<<alpha<<endl;}
      if(ch=='9') {active=(active+1)%2; cout<<"active learning rate "<<active<<endl;}
      if(ch=='0') {gpu=(gpu+1)%2; cout<<"gpu "<<gpu<<endl;}
      if(ch=='f') {fft=(fft+1)%2; cout<<"fft "<<fft<<endl;}
      if(ch=='q') {feat=(feat+1)%(net_c[n_l].size()); cout<<"feature map "<<feat<<endl;}
      if(ch=='w') {feat=(feat-1)%(net_c[n_l].size()); cout<<"feature map "<<feat<<endl;}
      if(ch=='z') 
      {
         n_l=(n_l+1)%(net_c.size()/2); 
         feat=0;
         int dM=net_c[n_l].size();
         int dD=net_c[n_l][0].size();
         int Nk=net_c[n_l][0][0].size();
         int Nl=net_c[n_l][0][0][0].size();
         //Init vectors of previous weight update dc= c(t-1)-c(t-2) (used in inertia update)
         Init_conv(dc, db, dM, dD, Nk, Nl, 0);
         Init_conv(df, dp, dD, dM, Nk, Nl, 0);
         //Init vectors of previous gradient values ddc=grad(c)(t-1) (used in adaptive learning rate)
         Init_conv(ddc, ddb, dM, dD, Nk, Nl, 0);
         Init_conv(ddf, ddp, dD, dM, Nk, Nl, 0);
         cout<<"Active layer "<<n_l<<endl;
      }
      if(ch=='x') 
      {
         n_l=(n_l-1)%(net_c.size()/2); 
         feat=0;
         int dM=net_c[n_l].size();
         int dD=net_c[n_l][0].size();
         int Nk=net_c[n_l][0][0].size();
         int Nl=net_c[n_l][0][0][0].size();
         //Init vectors of previous weight update dc= c(t-1)-c(t-2) (used in inertia update)
         Init_conv(dc, db, dM, dD, Nk, Nl, 0);
         Init_conv(df, dp, dD, dM, Nk, Nl, 0);
         //Init vectors of previous gradient values ddc=grad(c)(t-1) (used in adaptive learning rate)
         Init_conv(ddc, ddb, dM, dD, Nk, Nl, 0);
         Init_conv(ddf, ddp, dD, dM, Nk, Nl, 0);
         cout<<"Active layer "<<n_l<<endl;
      }
      if(ch=='e') 
      {
         int N=net_c.size()-1;
         int dM=net_c[n_l].size();
         int dD=net_c[n_l][0].size();
         int Nk=net_c[n_l][0][0].size();
         int Nl=net_c[n_l][0][0][0].size();
         float rmax;
         int a1,a2,a3,a4;
         LoadParam(a1,a2,a3,a4, rmax);
         Init_conv(net_c[n_l], net_b[n_l], dM, dD, Nk, Nl, rmax);
         Init_conv(net_c[N-n_l], net_b[N-n_l], dD, dM, Nk, Nl, rmax);
         cout<<"Initialize random convolutional weights "<<endl;
      }
      if(ch=='c')
      {
         net_cfreq.clear();
         cout<<"Cleaned fft convolutional kernels vector"<<endl;
      }
      if(ch=='p') 
      {
         sym=(sym+1)%2;
         cout<<"Symmetric weights "<<sym<<endl;
         if(sym==1)
         {
            int N=net_c.size()-1;
            int dM=net_c[n_l].size();
            int dD=net_c[n_l][0].size();
            int Nk=net_c[n_l][0][0].size();
            int Nl=net_c[n_l][0][0][0].size();
            for(int m=0;m<dM;m++)
            {
               for(int d=0;d<dD;d++)
               {
                  for(int k=0;k<Nk;k++)
                  {
                     for(int l=0;l<Nl;l++)
                     {
                        net_c[N-n_l][d][m][k][l]=net_c[n_l][m][d][k][l];
                     }
                  }
               }
            }
         }
      }
      if(ch=='s') 
      {
         int N=net_c.size()-1;
         SaveLoad_conv(net_c[n_l], net_b[n_l], n_l, 0, 1);
         SaveLoad_conv(net_c[N-n_l], net_b[N-n_l], n_l, 1, 1);
         //for(int n=0;n<N/2;n++)
         //{
         //   SaveLoad_conv(net_c[n], net_b[n], n, 0, 1);
         //   SaveLoad_conv(net_c[N-1-n], net_b[N-1-n], n, 1, 1);
         //}
         cout<<"Saved convolutional weights "<<endl;
      }
      if(ch=='l') 
      {
         int N=net_c.size()-1;
         SaveLoad_conv(net_c[n_l], net_b[n_l], n_l, 0, 0);
         SaveLoad_conv(net_c[N-n_l], net_b[N-n_l], n_l, 1, 0);
         //for(int n=0;n<N/2;n++)
         //{
         //   SaveLoad_conv(net_c[n], net_b[n], n, 0, 0);
         //   SaveLoad_conv(net_c[N-1-n], net_b[N-1-n], n, 1, 0);
         //}
         cout<<"Loaded convolutional weights "<<endl;
      }
      if(ch=='n')
      {
         int dM=10, Lk=0, Ll=0, scal=2; 
         float rmax=3;
         LoadParam(dM,Lk,Ll,scal,rmax);
         int dNk=2*(2*Lk+1)+1;
         int dNl=2*(2*Ll+1)+1;
         int n=(layers.size()-1)/2;
         int dD=layers[n].size();
         int dNx=layers[n][0].size();
         int dNy=layers[n][0][0].size();
         //New Convolutional and pooling layers
         vector<vector<vector<float> > > Pin_n(dD, vector<vector<float> >(dNx/scal, 
                                         vector<float>(dNy/scal)));
         vector<vector<vector<float> > > hC_n(dM, vector<vector<float> >(dNx/scal, 
                                         vector<float>(dNy/scal)));
         vector<vector<vector<float> > > PhC_n(dD, vector<vector<float> >(dNx/scal, 
                                         vector<float>(dNy/scal)));
         vector<vector<vector<float> > > out_n(dD, vector<vector<float> >(dNx, 
                                         vector<float>(dNy)));
         layers.insert(layers.begin()+n+1,out_n);
         layers.insert(layers.begin()+n+1,PhC_n);
         layers.insert(layers.begin()+n+1,hC_n);
         layers.insert(layers.begin()+n+1,Pin_n);
         //New convolutional filters
         vector<vector<vector<vector<float> > > > c_n, f_n;
         //New bias
         vector<float> b_n(dM), p_n(dD);  
         Init_conv(c_n, b_n, dM, dD, dNk, dNl, rmax);
         Init_conv(f_n, p_n, dD, dM, dNk, dNl, rmax);
         n=net_c.size()/2;
         net_c.insert(net_c.begin()+n,f_n);
         net_c.insert(net_c.begin()+n,c_n);
         net_b.insert(net_b.begin()+n,p_n);
         net_b.insert(net_b.begin()+n,b_n);
         scale.insert(scale.begin()+n,-scal);
         scale.insert(scale.begin()+n,scal);
         //Init vectors of previous weight update dc= c(t-1)-c(t-2) (used in inertia update)
         Init_conv(dc, db, dM, dD, dNk, dNl, 0);
         Init_conv(df, dp, dD, dM, dNk, dNl, 0);
         //Init vectors of previous gradient values ddc=grad(c)(t-1) (used in adaptive learning rate)
         Init_conv(ddc, ddb, dM, dD, dNk, dNl, 0);
         Init_conv(ddf, ddp, dD, dM, dNk, dNl, 0);
         n_l=n;
         cout<<"Added new layer L "<<net_c.size()/2<<endl;
      }
      if(ch=='d')
      {
         if(net_c.size()>2)
         {
            int n=net_c.size()/2;
            net_c.erase(net_c.begin()+n-1, net_c.begin()+n+1);
            net_b.erase(net_b.begin()+n-1, net_b.begin()+n+1);
            scale.erase(scale.begin()+n-1, scale.begin()+n+1);
            n=(layers.size()-1)/2;
            layers.erase(layers.begin()+n-1, layers.begin()+n+3);
            n_l=0%(net_c.size()/2); 
            int dM=net_c[n_l].size();
            int dD=net_c[n_l][0].size();
            int Nk=net_c[n_l][0][0].size();
            int Nl=net_c[n_l][0][0][0].size();
            //Init vectors of previous weight update dc= c(t-1)-c(t-2) (used in inertia update)
            Init_conv(dc, db, dM, dD, Nk, Nl, 0);
            Init_conv(df, dp, dD, dM, Nk, Nl, 0);
            //Init vectors of previous gradient values ddc=grad(c)(t-1) (used in adaptive learning rate)
            Init_conv(ddc, ddb, dM, dD, Nk, Nl, 0);
            Init_conv(ddf, ddp, dD, dM, Nk, Nl, 0);
            cout<<"Deleted last layer"<<endl;
         }
      }
      if(ch=='i')
      {
         cout<<"Network structure"<<endl<<endl;
         for(int n=0;n<net_c.size();n++)
         {
            int nl=2*n;
            if(n<net_c.size()/2)
            {
               cout<<"    L="<<nl<<" D="<<layers[nl].size()<<" Nx="
                   <<layers[nl][0].size()<<" Ny="<<layers[nl][0][0].size()<<endl;
               cout<<"P="<<n<<" S="<<scale[n]<<endl;
               cout<<"    L="<<nl+1<<" D="<<layers[nl+1].size()<<" Nx="
                   <<layers[nl+1][0].size()<<" Ny="<<layers[nl+1][0][0].size()<<endl;
               cout<<"C="<<n<<" M="<<net_c[n].size()<<" D="
                   <<net_c[n][0].size()<<" Nk="<<net_c[n][0][0].size()<<" Nl="<<net_c[n][0][0][0].size()<<endl;
               cout<<"B="<<n<<" M="<<net_b[n].size()<<endl;
               cout<<"----------"<<endl;
            }
            else
            {
               cout<<"    L="<<nl<<" D="<<layers[nl].size()<<" Nx="
                   <<layers[nl][0].size()<<" Ny="<<layers[nl][0][0].size()<<endl;
               cout<<"C="<<n<<" M="<<net_c[n].size()<<" D="
                   <<net_c[n][0].size()<<" Nk="<<net_c[n][0][0].size()<<" Nl="<<net_c[n][0][0][0].size()<<endl;
               cout<<"B="<<n<<" M="<<net_b[n].size()<<endl;
               cout<<"    L="<<nl+1<<" D="<<layers[nl+1].size()<<" Nx="
                   <<layers[nl+1][0].size()<<" Ny="<<layers[nl+1][0][0].size()<<endl;
               cout<<"P="<<n<<" S="<<scale[n]<<endl;
               cout<<"----------"<<endl;
            }
         }
         int nl=2*(net_c.size()-1)+2;
         cout<<"    L="<<nl<<" D="<<layers[nl].size()<<" Nx="
             <<layers[nl][0].size()<<" Ny="<<layers[nl][0][0].size()<<endl;
      }



   }


   return 0;
}
