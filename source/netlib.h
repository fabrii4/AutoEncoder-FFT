#ifndef NETLIB_H
#define NETLIB_H

void ImageToSpin_C(cv::Mat& img, std::vector<std::vector<std::vector<float> > >& spin);

void SpinToImage_C(cv::Mat& img, std::vector<std::vector<std::vector<float> > >& spin);

void SpinToImage_V(cv::Mat& img, std::vector<std::vector<float> >& spin);

void SpinToImage_K(cv::Mat& img, std::vector<std::vector<float> >& spin);

void Pool(std::vector<std::vector<std::vector<float> > >& in, std::vector<std::vector<std::vector<float> > >& out, int scale);

void Init_conv(std::vector<std::vector<std::vector<std::vector<float > > > >& c, std::vector<float>& b, int mS, int dS, int kS, int lS, float max);

void SaveLoad_conv(std::vector<std::vector<std::vector<std::vector<float > > > >& c, std::vector<float>& b, int L, int io, int write);

void LoadParam(int& dM, int& Lk, int& Ll, int& scal, float& rmax);

void Portion(std::vector<std::vector<std::vector<float> > >& in, std::vector<std::vector<std::vector<float> > >& hin, std::vector<std::vector<std::vector<float> > >& out, std::vector<std::vector<std::vector<float> > >& in_s, std::vector<std::vector<std::vector<float> > >& hin_s, std::vector<std::vector<std::vector<float> > >& out_s, int q);

void Conv(std::vector<std::vector<std::vector<float> > >& in, std::vector<std::vector<std::vector<float> > >& out,  std::vector<std::vector<std::vector<std::vector<float > > > >& c, std::vector<float>& b);

void backprop(std::vector<std::vector<std::vector<float> > >& in, std::vector<std::vector<std::vector<float> > >& out, std::vector<std::vector<std::vector<float> > >& hin, std::vector<std::vector<std::vector<std::vector<float > > > >& c, std::vector<float>& b, std::vector<std::vector<std::vector<std::vector<float > > > >& f, std::vector<float>& p, float del);

#endif
