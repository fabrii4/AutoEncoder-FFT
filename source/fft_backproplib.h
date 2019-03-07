#ifndef FFTBACKPROPLIB_H
#define FFTBACKPROPLIB_H


void autoenc_fft(std::vector<std::vector<std::vector<std::vector<float> > > >& layers, std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > >& net_c, std::vector<std::vector<float> > & net_cfreq, std::vector<std::vector<float> >& net_b, std::vector<int>& scale, int fft_l);

void conv_fft(std::vector<std::vector<std::vector<float> > >& in, std::vector<std::vector<std::vector<float> > >& out, std::vector<std::vector<std::vector<std::vector<float> > > >& c, std::vector<float>& b, int scale, int comp_magn);

void kernel_pad(std::vector<std::vector<std::vector<std::vector<float> > > >& c, std::vector<std::vector<std::vector<std::vector<float> > > >& c_pad, int Nx, int Ny);

void backprop_fft(std::vector<std::vector<std::vector<float> > >& in, std::vector<std::vector<std::vector<float> > >& out, std::vector<float>& cfreq, std::vector<std::vector<std::vector<std::vector<float> > > >& c, std::vector<float>& ffreq, std::vector<std::vector<std::vector<std::vector<float> > > >& f,  std::vector<float>& b, std::vector<float>& p, int dM, float del0, int maxdiff);


#endif
