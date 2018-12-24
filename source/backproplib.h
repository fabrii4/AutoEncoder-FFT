#ifndef BACKPROPLIB_H
#define BACKPROPLIB_H


void backprop_gpu(std::vector<std::vector<std::vector<float> > >& in, std::vector<std::vector<std::vector<float> > >& out, std::vector<std::vector<std::vector<float> > >& hin,
std::vector<std::vector<std::vector<std::vector<float> > > >& c, std::vector<float>& b, std::vector<std::vector<std::vector<std::vector<float> > > >& f, std::vector<float>& p, std::vector<std::vector<std::vector<std::vector<float> > > >& dc, std::vector<float>& db, std::vector<std::vector<std::vector<std::vector<float> > > >& df, std::vector<float>& dp, std::vector<std::vector<std::vector<std::vector<float> > > >& ddc, std::vector<float>& ddb, std::vector<std::vector<std::vector<std::vector<float> > > >& ddf, std::vector<float>& ddp, float delmax, float alpha, int active);

void Conv_gpu(std::vector<std::vector<std::vector<float> > >& in, std::vector<std::vector<std::vector<float> > >& out,  std::vector<std::vector<std::vector<std::vector<float> > > >& c, std::vector<float>& b);

void backprop_gpu_cc(std::vector<std::vector<std::vector<float> > >& in, std::vector<std::vector<std::vector<float> > >& out, std::vector<std::vector<std::vector<float> > >& hin,
std::vector<std::vector<std::vector<std::vector<float> > > >& c, std::vector<float>& b, std::vector<std::vector<std::vector<std::vector<float> > > >& f, std::vector<float>& p, std::vector<std::vector<std::vector<std::vector<float> > > >& dc, std::vector<float>& db, std::vector<std::vector<std::vector<std::vector<float> > > >& df, std::vector<float>& dp, std::vector<std::vector<std::vector<std::vector<float> > > >& ddc, std::vector<float>& ddb, std::vector<std::vector<std::vector<std::vector<float> > > >& ddf, std::vector<float>& ddp, float delmax, float alpha, int active);



float act(float x);
float act1(float x);
//void adapt_rate(float del, float delmax, int active, float dDdX, float ddx, float dx);

#endif
