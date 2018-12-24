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

using namespace cv;
using namespace std;


int main()
{
   //video size
   int Nx=400, Ny=400;
   //input depth
   int D=3;
   //convolution depth
   int M=20;
   //convolution size |k|<2L+1
   int L=0;
   int Nkl=2*(2*L+1)+1;

   //opencv init
   Mat rgb;
   Mat img(Size(Nx,Ny),CV_8U);
   Mat imgC(Size(Nx,Ny),CV_8UC3);
   Mat output(Size(Nx,Ny),CV_8U);
   Mat outputC(Size(Nx,Ny),CV_8UC3);
   Mat output1(Size(3*Nkl,Nkl),CV_8U);
   VideoCapture cam(1);
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

   //Convolutional layer
   vector<vector<vector<float> > > in(D, vector<vector<float> >(Nx, vector<float>(Ny)));
   vector<vector<vector<float> > > out(D, vector<vector<float> >(Nx, vector<float>(Ny)));
   vector<vector<vector<float> > > hC(M, vector<vector<float> >(Nx, vector<float>(Ny)));
   vector<vector<vector<vector<float> > > > layers;
   //convolutional filters
   vector<vector<vector<vector<float> > > > c, f, dc, df, ddc, ddf;
   vector<vector<vector<vector<vector<float> > > > > net_c, net_f;
   //bias
   vector<float> b(M), p(D), db(M), dp(D), ddb(M), ddp(D);  
   vector<vector<float> > net_b, net_p;
   //keyboard controls
   int sel=0;
   int q=1;
   float del=0.1, ddel=0.01;
   int active=1; 
   float alpha=0.1;
   int feat=0;
   int gpu=0;

   //Init random convolutional filters
   srand(time(0));
   Init_conv(c, b, M, D, Nkl, Nkl, 3);
   Init_conv(f, p, D, M, Nkl, Nkl, 3);
   //Init vectors of previous weight update dc= c(t-1)-c(t-2) (used in inertia update)
   Init_conv(dc, db, M, D, Nkl, Nkl, 0);
   Init_conv(df, dp, D, M, Nkl, Nkl, 0);
   //Init vectors of previous gradient values ddc=grad(c)(t-1) (used in adaptive learning rate)
   Init_conv(ddc, ddb, M, D, Nkl, Nkl, 0);
   Init_conv(ddf, ddp, D, M, Nkl, Nkl, 0);

   while(true)
   {
      cam>>rgb;
      resize(rgb,imgC,Size(Nx,Ny));
      ImageToSpin_C(imgC,in);

      //apply coder-decoder convolutions
      Conv(in, hC, c, b);
      //Pool(hC, hCs, scale);
//      Conv(hCs, hCs1, c1, b1);
//      Conv(hCs1, outs1, f1, p1);      
      //RPool(hCs, hC, scale);
      Conv(hC, out, f, p);
      //Conv_gpu(in, hC, c, b);
      //Conv_gpu(hC, out, f, p);
      
      //backpropagation
      if(sel==1)
      {
         auto start = std::chrono::high_resolution_clock::now();
         vector<vector<vector<float> > > in_s(D, vector<vector<float> >(Nx/q, vector<float>(Ny/q)));
         vector<vector<vector<float> > > out_s(D, vector<vector<float> >(Nx/q, vector<float>(Ny/q)));
         vector<vector<vector<float> > > hC_s(M, vector<vector<float> >(Nx/q, vector<float>(Ny/q)));
         Portion(in,out,hC,in_s,out_s,hC_s,q);
         if(gpu==1) backprop_gpu(in_s, out_s, hC_s,
                                 c, b, f, p, 
                                 dc, db, df, dp, 
                                 ddc, ddb, ddf, ddp, 
                                 del, alpha, active);
         else backprop(in_s, out_s, hC_s, c, b, f, p, del);
         auto finish = std::chrono::high_resolution_clock::now();
         std::chrono::duration<double> elapsed = finish - start;
         std::cout << "Time: " << elapsed.count() << " s\n";
      }

       

      //show results
      imshow("input",imgC);
      SpinToImage_C(outputC,out);
      imshow("output",outputC);
      SpinToImage_V(output,hC[feat]);
      imshow("output1",output);
      for(int i=0;i<3;i++)
      {
         Rect Rec(Nkl*i, 0, Nkl, Nkl);
         Mat Roi = output1(Rec);
         SpinToImage_V(Roi,c[feat][i]);
         Roi.copyTo(output1(Rec));
      }
      imshow("output2",output1);

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
      if(ch=='q') {feat=(feat+1)%M; cout<<"feature map "<<feat<<endl;}
      if(ch=='w') {feat=(feat-1)%M; cout<<"feature map "<<feat<<endl;}
      if(ch=='e') 
      {
         Init_conv(c, b, M, D, Nkl, Nkl, 3);
         Init_conv(f, p, D, M, Nkl, Nkl, 3);
         cout<<"Initialize random convolutional weights "<<endl;
      }
      if(ch=='s') 
      {
         //Save_conv(c, b, f, p);
         SaveLoad_conv(c, b, 0, 1);
         SaveLoad_conv(f, p, 1, 1);
         cout<<"Save convolutional weights "<<endl;
      }
      if(ch=='l') 
      {  
         SaveLoad_conv(c, b, 0, 0);
         SaveLoad_conv(f, p, 1, 0);
         cout<<"Load convolutional weights "<<endl;
      }



   }


   return 0;
}
