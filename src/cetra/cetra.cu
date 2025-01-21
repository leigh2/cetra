// Copyright (c) 2024 Leigh C. Smith - lsmith@ast.cam.ac.uk
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <float.h>

// open the shared memory array
extern __shared__ char sm[];

__device__ void warpSumReductionf(volatile float* sdata, int tid) {
    //sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__device__ void warpSumReductiond(volatile double* sdata, int tid) {
    //sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__device__ void warpSumReductioni(volatile int* sdata, int tid) {
    //sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// detrender - quadratic-only fit
__global__ void detrender_quadfit(
    const double * time,  // offset time array
    const double * flux,  // offset flux array
    const double * wght,  // offset flux weight array
    const int kernel_half_width,  // half-width of the detection kernel in samples
    const int min_obs_in_window,  // the minimum acceptable number of observations in the window
    const float cadence,  // the cadence of the light curve
    const int lc_size,  // number of light curve elements
    const float t0_stride_length,  // the number of reference times per duration
    const int t0_stride_count,  // number of reference time strides
    double * sw,  // intermediate value (to be filled)
    double * swx,  // intermediate value (to be filled)
    double * swy,  // intermediate value (to be filled)
    double * swxx,  // intermediate value (to be filled)
    double * swxy,  // intermediate value (to be filled)
    double * swxxx,  // intermediate value (to be filled)
    double * swxxy,  // intermediate value (to be filled)
    double * swxxxx,  // intermediate value (to be filled)
    int * num_pts,  // number of points in window (to be filled)
    double * ll_quad  // the log-likelihood of the quadratic model (to be filled)
){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= t0_stride_count) return;

    // compute the reference time
    float t0 = tid * t0_stride_length;

    // compute the index of the first and last points in the window
    int frst_idx = lrintf(t0 / cadence) - kernel_half_width;
    int last_idx = lrintf(t0 / cadence) + kernel_half_width;

    // accumulators
    double _sw = 0.0;
    double _swx = 0.0;
    double _swy = 0.0;
    double _swxx = 0.0;
    double _swxy = 0.0;
    double _swxxx = 0.0;
    double _swxxy = 0.0;
    double _swxxxx = 0.0;
    int _num_pts = 0;

    // loop over the light curve in the detrending window
    for (int lc_idx = frst_idx; lc_idx <= last_idx; lc_idx++){

        // deal with light curve ends
        if (lc_idx < 0) continue;
        if (lc_idx >= lc_size) break;

        // skip if the light curve point has infinite error (i.e. zero weight)
        double lc_w = (double) wght[lc_idx];
        if (lc_w == 0.0) continue;

        // grab the values of time and flux to make the following code more readable
        // (assume the compiler is smart)
        double lc_t = (double) time[lc_idx];
        double lc_f = (double) flux[lc_idx];
        // we don't want to use the absolute value of time because the floating
        // point errors make a difference when you take the fourth power...
        double delta_t = lc_t - t0;

        // accumulate various values
        _sw += lc_w;
        _swx += lc_w * delta_t;
        _swy += lc_w * lc_f;
        _swxx += lc_w * delta_t * delta_t;
        _swxy += lc_w * delta_t * lc_f;
        _swxxx += lc_w * delta_t * delta_t * delta_t;
        _swxxy += lc_w * delta_t * delta_t * lc_f;
        _swxxxx += lc_w * delta_t * delta_t * delta_t * delta_t;
        _num_pts += 1;
    }

    // nothing more to do if there were insufficient data points
    num_pts[tid] = _num_pts;
    if (_num_pts < min_obs_in_window) {
        sw[tid] = nan(0);
        swx[tid] = nan(0);
        swy[tid] = nan(0);
        swxx[tid] = nan(0);
        swxy[tid] = nan(0);
        swxxx[tid] = nan(0);
        swxxy[tid] = nan(0);
        swxxxx[tid] = nan(0);
        ll_quad[tid] = nan(0);
        return;
    }

    // send these intermediate values to the output arrays
    sw[tid] = _sw;
    swx[tid] = _swx;
    swy[tid] = _swy;
    swxx[tid] = _swxx;
    swxy[tid] = _swxy;
    swxxx[tid] = _swxxx;
    swxxy[tid] = _swxxy;
    swxxxx[tid] = _swxxxx;

    /*
    B1 is the quadratic coefficient
    B2 is the linear coefficient
    B3 is the constant coefficient
    i.e.
    f(t) = B1*t^2 + B2*t + B3
    */

    // calculate the least squares parameters for the quadratic model
    double B1 = (_sw*_swxx*_swxxy - _sw*_swxxx*_swxy - _swx*_swx*_swxxy + _swx*_swxx*_swxy + _swx*_swxxx*_swy - _swxx*_swxx*_swy)
                 /(_sw*_swxx*_swxxxx - _sw*_swxxx*_swxxx - _swx*_swx*_swxxxx + 2*_swx*_swxx*_swxxx - _swxx*_swxx*_swxx);
    double B2 = (-_sw*_swxxx*_swxxy + _sw*_swxxxx*_swxy + _swx*_swxx*_swxxy - _swx*_swxxxx*_swy - _swxx*_swxx*_swxy + _swxx*_swxxx*_swy)
                 /(_sw*_swxx*_swxxxx - _sw*_swxxx*_swxxx - _swx*_swx*_swxxxx + 2*_swx*_swxx*_swxxx - _swxx*_swxx*_swxx);
    double B3 = (_swx*_swxxx*_swxxy - _swx*_swxxxx*_swxy - _swxx*_swxx*_swxxy + _swxx*_swxxx*_swxy + _swxx*_swxxxx*_swy - _swxxx*_swxxx*_swy)
                 /(_sw*_swxx*_swxxxx - _sw*_swxxx*_swxxx - _swx*_swx*_swxxxx + 2*_swx*_swxx*_swxxx - _swxx*_swxx*_swxx);

    // now loop through the detrending window again to calculate the log likelihood
    double _ll_sum = 0.0;
    for (int lc_idx = frst_idx; lc_idx <= last_idx; lc_idx++){

        // deal with light curve ends
        if (lc_idx < 0) continue;
        if (lc_idx >= lc_size) break;

        // skip if the light curve point has infinite error (i.e. zero weight)
        double lc_w = (double) wght[lc_idx];
        if (lc_w == 0.0) continue;

        // grab the values of time and flux to make the following code more readable
        // (assume the compiler is smart)
        double lc_t = (double) time[lc_idx];
        double lc_f = (double) flux[lc_idx];
        // we don't want to use the absolute value of time because the floating
        // point errors make a difference when you take the fourth power...
        double delta_t = lc_t - t0;

        // compute the best fit models for this point
        double model_flux = B1 * delta_t * delta_t + B2 * delta_t + B3;

        // compute the residual
        double resid = model_flux - lc_f;
        double e_term = -0.5 * log(2.0 * M_PI / lc_w);
        _ll_sum += (-0.5 * resid * resid * lc_w) + e_term;
    }

    // send the log likelihood to the output array
    ll_quad[tid] = _ll_sum;
}

// detrender - quadratic plus transit fit
__global__ void detrender_qtrfit(
    const double * time,  // offset time array
    const double * flux,  // offset flux array
    const double * wght,  // offset flux weight array
    const int kernel_half_width,  // half-width of the detection kernel in samples
    const float min_depth_ppm,  // the minimum transit depth to consider
    const int min_obs_in_window,  // the minimum acceptable number of observations in the window
    const float cadence,  // the cadence of the light curve
    const int lc_size,  // number of light curve elements
    const double * tmodel,  // offset flux transit model array
    const int tm_size,  // number of transit model elements
    const float * durations,  // the duration array
    const int n_durations,  // the number of durations
    const float t0_stride_length,  // the number of reference times per duration
    const int t0_stride_count,  // number of reference time strides
    const double * sw,  // intermediate value
    const double * swx,  // intermediate value
    const double * swy,  // intermediate value
    const double * swxx,  // intermediate value
    const double * swxy,  // intermediate value
    const double * swxxx,  // intermediate value
    const double * swxxy,  // intermediate value
    const double * swxxxx,  // intermediate value
    const int * num_pts,  // number of points in detrending window
    double * ll_qtr  // the log-likelihood of the quad+transit model (to be filled)
){
    // specify the shared memory array locations and types
    // todo: could set these with metaprogramming, maybe there is performance to be gained...
    double * sm_tmodel = (double*)sm;
    double * sm_stjwj = (double*)&sm_tmodel[tm_size];
    double * sm_sttjwj = (double*)&sm_stjwj[blockDim.x];
    double * sm_stjwjxj = (double*)&sm_sttjwj[blockDim.x];
    double * sm_stjwjyj = (double*)&sm_stjwjxj[blockDim.x];
    double * sm_stjwjxxj = (double*)&sm_stjwjyj[blockDim.x];
    double * sm_ll_tr = (double*)&sm_stjwjxxj[blockDim.x];

    // read the transit model into shared memory
    for (int i = 0; i < tm_size; i += blockDim.x){
        int sm_idx = i + threadIdx.x;
        if (sm_idx >= tm_size) break;
        sm_tmodel[sm_idx] = tmodel[sm_idx];
    }

    // duration index
    const int dur_id = blockIdx.y;

    // grab the duration
    if (dur_id >= n_durations) return;
    const float duration = durations[dur_id];

    // Stride through the reference time steps
    // Each block reads the transit model from global to shared memory once.
    // Striding through the reference time steps means each block computes the likelihood
    // ratio of multiple reference time steps, but it still only reads the transit model
    // once, so the total number of reads from global memory is reduced.
    for (int s = 0; s < t0_stride_count; s += gridDim.x) {
        // t0 number
        int t0_num = blockIdx.x + s;
        if (t0_num >= t0_stride_count) return;

        // 2d output array pointer
        int arr2d_ptr = t0_num + t0_stride_count * dur_id;

        // if there are too few data points in the window, skip this loop
        if (num_pts[t0_num] < min_obs_in_window){
            ll_qtr[arr2d_ptr] = nan(0);
            continue;
        }

        // calculate ts, the transit start time
        float t0 = t0_num * t0_stride_length;
        float ts = t0 - 0.5 * duration;

        // zero out the additional arrays in shared memory
        sm_stjwj[threadIdx.x] = 0.0;
        sm_sttjwj[threadIdx.x] = 0.0;
        sm_stjwjxj[threadIdx.x] = 0.0;
        sm_stjwjyj[threadIdx.x] = 0.0;
        sm_stjwjxxj[threadIdx.x] = 0.0;
        sm_ll_tr[threadIdx.x] = 0.0;

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // compute the indices of the first and last in-transit points
        int itr_frst_idx = lrintf(ceilf(ts / cadence));
        int itr_last_idx = lrintf(floorf((ts+duration) / cadence));
        // compute the indices of the first and last detection kernel points
        int dtr_frst_idx = lrintf(t0 / cadence) - kernel_half_width;
        int dtr_last_idx = lrintf(t0 / cadence) + kernel_half_width;

        // loop over the light curve in the transit window
        for (int lc_idx = (itr_frst_idx + threadIdx.x); lc_idx <= itr_last_idx; lc_idx += blockDim.x){

            // deal with light curve ends
            if (lc_idx < 0) continue;
            if (lc_idx >= lc_size) break;

            // skip if the light curve point has infinite error (i.e. zero weight)
            double lc_w = (double) wght[lc_idx];
            if (lc_w == 0.0) continue;

            // grab the values of time and flux to make the following code more readable
            // (assume the compiler is smart)
            double lc_t = (double) time[lc_idx];
            double lc_f = (double) flux[lc_idx];
            // we don't want to use the absolute value of time because the floating
            // point errors make a difference when you take the fourth power...
            double delta_t = lc_t - t0;

            // find the nearest model point index
            int model_idx = lrintf(( lc_t - ts ) / duration * tm_size);
            // just in case we're out of bounds:
            if ((model_idx < 0) || (model_idx >= tm_size)) continue;
            double modval = (double) sm_tmodel[model_idx];

            // accumulate some additional values
            sm_stjwj[threadIdx.x] += lc_w * modval;
            sm_sttjwj[threadIdx.x] += lc_w * modval * modval;
            sm_stjwjxj[threadIdx.x] += lc_w * modval * delta_t;
            sm_stjwjyj[threadIdx.x] += lc_w * modval * lc_f;
            sm_stjwjxxj[threadIdx.x] += lc_w * modval * delta_t * delta_t;
        }

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // sum reduction of the additional arrays in shared memory
        warpSumReductiond(sm_stjwj, threadIdx.x);
        warpSumReductiond(sm_sttjwj, threadIdx.x);
        warpSumReductiond(sm_stjwjxj, threadIdx.x);
        warpSumReductiond(sm_stjwjyj, threadIdx.x);
        warpSumReductiond(sm_stjwjxxj, threadIdx.x);

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // pull these values out
        double _sw = sw[t0_num];
        double _swx = swx[t0_num];
        double _swy = swy[t0_num];
        double _swxx = swxx[t0_num];
        double _swxy = swxy[t0_num];
        double _swxxx = swxxx[t0_num];
        double _swxxy = swxxy[t0_num];
        double _swxxxx = swxxxx[t0_num];
        double _stjwj = sm_stjwj[0];
        double _sttjwj = sm_sttjwj[0];
        double _stjwjxj = sm_stjwjxj[0];
        double _stjwjyj = sm_stjwjyj[0];
        double _stjwjxxj = sm_stjwjxxj[0];

        /*
        B1 is the quadratic coefficient
        B2 is the linear coefficient
        B3 is the constant coefficient
        B4 is the transit model depth
        i.e.
        f(t) = B1*t^2 + B2*t + B3 + B4*tmodel
        */

        // calculate the least squares parameters for the transit model
        double B1 = (_stjwj*_stjwj*_swxx*_swxxy - _stjwj*_stjwj*_swxxx*_swxy - 2*_stjwj*_stjwjxj*_swx*_swxxy + _stjwj*_stjwjxj*_swxx*_swxy + _stjwj*_stjwjxj*_swxxx*_swy + _stjwj*_stjwjxxj*_swx*_swxy - _stjwj*_stjwjxxj*_swxx*_swy + _stjwj*_stjwjyj*_swx*_swxxx - _stjwj*_stjwjyj*_swxx*_swxx + _stjwjxj*_stjwjxj*_sw*_swxxy - _stjwjxj*_stjwjxj*_swxx*_swy - _stjwjxj*_stjwjxxj*_sw*_swxy + _stjwjxj*_stjwjxxj*_swx*_swy - _stjwjxj*_stjwjyj*_sw*_swxxx + _stjwjxj*_stjwjyj*_swx*_swxx + _stjwjxxj*_stjwjyj*_sw*_swxx - _stjwjxxj*_stjwjyj*_swx*_swx - _sttjwj*_sw*_swxx*_swxxy + _sttjwj*_sw*_swxxx*_swxy + _sttjwj*_swx*_swx*_swxxy - _sttjwj*_swx*_swxx*_swxy - _sttjwj*_swx*_swxxx*_swy + _sttjwj*_swxx*_swxx*_swy)
                     /(_stjwj*_stjwj*_swxx*_swxxxx - _stjwj*_stjwj*_swxxx*_swxxx - 2*_stjwj*_stjwjxj*_swx*_swxxxx + 2*_stjwj*_stjwjxj*_swxx*_swxxx + 2*_stjwj*_stjwjxxj*_swx*_swxxx - 2*_stjwj*_stjwjxxj*_swxx*_swxx + _stjwjxj*_stjwjxj*_sw*_swxxxx - _stjwjxj*_stjwjxj*_swxx*_swxx - 2*_stjwjxj*_stjwjxxj*_sw*_swxxx + 2*_stjwjxj*_stjwjxxj*_swx*_swxx + _stjwjxxj*_stjwjxxj*_sw*_swxx - _stjwjxxj*_stjwjxxj*_swx*_swx - _sttjwj*_sw*_swxx*_swxxxx + _sttjwj*_sw*_swxxx*_swxxx + _sttjwj*_swx*_swx*_swxxxx - 2*_sttjwj*_swx*_swxx*_swxxx + _sttjwj*_swxx*_swxx*_swxx);
        double B2 = (-_stjwj*_stjwj*_swxxx*_swxxy + _stjwj*_stjwj*_swxxxx*_swxy + _stjwj*_stjwjxj*_swxx*_swxxy - _stjwj*_stjwjxj*_swxxxx*_swy + _stjwj*_stjwjxxj*_swx*_swxxy - 2*_stjwj*_stjwjxxj*_swxx*_swxy + _stjwj*_stjwjxxj*_swxxx*_swy - _stjwj*_stjwjyj*_swx*_swxxxx + _stjwj*_stjwjyj*_swxx*_swxxx - _stjwjxj*_stjwjxxj*_sw*_swxxy + _stjwjxj*_stjwjxxj*_swxx*_swy + _stjwjxj*_stjwjyj*_sw*_swxxxx - _stjwjxj*_stjwjyj*_swxx*_swxx + _stjwjxxj*_stjwjxxj*_sw*_swxy - _stjwjxxj*_stjwjxxj*_swx*_swy - _stjwjxxj*_stjwjyj*_sw*_swxxx + _stjwjxxj*_stjwjyj*_swx*_swxx + _sttjwj*_sw*_swxxx*_swxxy - _sttjwj*_sw*_swxxxx*_swxy - _sttjwj*_swx*_swxx*_swxxy + _sttjwj*_swx*_swxxxx*_swy + _sttjwj*_swxx*_swxx*_swxy - _sttjwj*_swxx*_swxxx*_swy)
                     /(_stjwj*_stjwj*_swxx*_swxxxx - _stjwj*_stjwj*_swxxx*_swxxx - 2*_stjwj*_stjwjxj*_swx*_swxxxx + 2*_stjwj*_stjwjxj*_swxx*_swxxx + 2*_stjwj*_stjwjxxj*_swx*_swxxx - 2*_stjwj*_stjwjxxj*_swxx*_swxx + _stjwjxj*_stjwjxj*_sw*_swxxxx - _stjwjxj*_stjwjxj*_swxx*_swxx - 2*_stjwjxj*_stjwjxxj*_sw*_swxxx + 2*_stjwjxj*_stjwjxxj*_swx*_swxx + _stjwjxxj*_stjwjxxj*_sw*_swxx - _stjwjxxj*_stjwjxxj*_swx*_swx - _sttjwj*_sw*_swxx*_swxxxx + _sttjwj*_sw*_swxxx*_swxxx + _sttjwj*_swx*_swx*_swxxxx - 2*_sttjwj*_swx*_swxx*_swxxx + _sttjwj*_swxx*_swxx*_swxx);
        double B3 = (_stjwj*_stjwjxj*_swxxx*_swxxy - _stjwj*_stjwjxj*_swxxxx*_swxy - _stjwj*_stjwjxxj*_swxx*_swxxy + _stjwj*_stjwjxxj*_swxxx*_swxy + _stjwj*_stjwjyj*_swxx*_swxxxx - _stjwj*_stjwjyj*_swxxx*_swxxx - _stjwjxj*_stjwjxj*_swxx*_swxxy + _stjwjxj*_stjwjxj*_swxxxx*_swy + _stjwjxj*_stjwjxxj*_swx*_swxxy + _stjwjxj*_stjwjxxj*_swxx*_swxy - 2*_stjwjxj*_stjwjxxj*_swxxx*_swy - _stjwjxj*_stjwjyj*_swx*_swxxxx + _stjwjxj*_stjwjyj*_swxx*_swxxx - _stjwjxxj*_stjwjxxj*_swx*_swxy + _stjwjxxj*_stjwjxxj*_swxx*_swy + _stjwjxxj*_stjwjyj*_swx*_swxxx - _stjwjxxj*_stjwjyj*_swxx*_swxx - _sttjwj*_swx*_swxxx*_swxxy + _sttjwj*_swx*_swxxxx*_swxy + _sttjwj*_swxx*_swxx*_swxxy - _sttjwj*_swxx*_swxxx*_swxy - _sttjwj*_swxx*_swxxxx*_swy + _sttjwj*_swxxx*_swxxx*_swy)
                     /(_stjwj*_stjwj*_swxx*_swxxxx - _stjwj*_stjwj*_swxxx*_swxxx - 2*_stjwj*_stjwjxj*_swx*_swxxxx + 2*_stjwj*_stjwjxj*_swxx*_swxxx + 2*_stjwj*_stjwjxxj*_swx*_swxxx - 2*_stjwj*_stjwjxxj*_swxx*_swxx + _stjwjxj*_stjwjxj*_sw*_swxxxx - _stjwjxj*_stjwjxj*_swxx*_swxx - 2*_stjwjxj*_stjwjxxj*_sw*_swxxx + 2*_stjwjxj*_stjwjxxj*_swx*_swxx + _stjwjxxj*_stjwjxxj*_sw*_swxx - _stjwjxxj*_stjwjxxj*_swx*_swx - _sttjwj*_sw*_swxx*_swxxxx + _sttjwj*_sw*_swxxx*_swxxx + _sttjwj*_swx*_swx*_swxxxx - 2*_sttjwj*_swx*_swxx*_swxxx + _sttjwj*_swxx*_swxx*_swxx);
        double B4 = (_stjwj*_swx*_swxxx*_swxxy - _stjwj*_swx*_swxxxx*_swxy - _stjwj*_swxx*_swxx*_swxxy + _stjwj*_swxx*_swxxx*_swxy + _stjwj*_swxx*_swxxxx*_swy - _stjwj*_swxxx*_swxxx*_swy - _stjwjxj*_sw*_swxxx*_swxxy + _stjwjxj*_sw*_swxxxx*_swxy + _stjwjxj*_swx*_swxx*_swxxy - _stjwjxj*_swx*_swxxxx*_swy - _stjwjxj*_swxx*_swxx*_swxy + _stjwjxj*_swxx*_swxxx*_swy + _stjwjxxj*_sw*_swxx*_swxxy - _stjwjxxj*_sw*_swxxx*_swxy - _stjwjxxj*_swx*_swx*_swxxy + _stjwjxxj*_swx*_swxx*_swxy + _stjwjxxj*_swx*_swxxx*_swy - _stjwjxxj*_swxx*_swxx*_swy - _stjwjyj*_sw*_swxx*_swxxxx + _stjwjyj*_sw*_swxxx*_swxxx + _stjwjyj*_swx*_swx*_swxxxx - 2*_stjwjyj*_swx*_swxx*_swxxx + _stjwjyj*_swxx*_swxx*_swxx)
                     /(_stjwj*_stjwj*_swxx*_swxxxx - _stjwj*_stjwj*_swxxx*_swxxx - 2*_stjwj*_stjwjxj*_swx*_swxxxx + 2*_stjwj*_stjwjxj*_swxx*_swxxx + 2*_stjwj*_stjwjxxj*_swx*_swxxx - 2*_stjwj*_stjwjxxj*_swxx*_swxx + _stjwjxj*_stjwjxj*_sw*_swxxxx - _stjwjxj*_stjwjxj*_swxx*_swxx - 2*_stjwjxj*_stjwjxxj*_sw*_swxxx + 2*_stjwjxj*_stjwjxxj*_swx*_swxx + _stjwjxxj*_stjwjxxj*_sw*_swxx - _stjwjxxj*_stjwjxxj*_swx*_swx - _sttjwj*_sw*_swxx*_swxxxx + _sttjwj*_sw*_swxxx*_swxxx + _sttjwj*_swx*_swx*_swxxxx - 2*_sttjwj*_swx*_swxx*_swxxx + _sttjwj*_swxx*_swxx*_swxx);

        // require some minimum depth
        if (B4 < (min_depth_ppm * 1e-6)) {
            ll_qtr[arr2d_ptr] = nan(0);
            continue;
        }

        // now loop through the detrending window and calculate the log-likelihood
        for (int lc_idx = (dtr_frst_idx + threadIdx.x); lc_idx <= dtr_last_idx; lc_idx += blockDim.x){

            // deal with light curve ends
            if (lc_idx < 0) continue;
            if (lc_idx >= lc_size) break;

            // skip if the light curve point has infinite error (i.e. zero weight)
            double lc_w = (double) wght[lc_idx];
            if (lc_w == 0.0) continue;

            // grab the values of time and flux to make the following code more readable
            // (assume the compiler is smart)
            double lc_t = (double) time[lc_idx];
            double lc_f = (double) flux[lc_idx];
            // we don't want to use the absolute value of time because the floating
            // point errors make a difference when you take the fourth power...
            double delta_t = lc_t - t0;

            // compute the best fit models for this point
            double model_flux = B1 * delta_t * delta_t + B2 * delta_t + B3;

            // is this point in the transit window?
            if ((lc_idx >= itr_frst_idx) & (lc_idx <= itr_last_idx)) {
                // find the nearest model point index
                int model_idx = lrintf(( lc_t - ts ) / duration * tm_size);
                // just in case we're out of bounds:
                if ((model_idx < 0) || (model_idx >= tm_size)) continue;
                double modval = (double) sm_tmodel[model_idx];

                // incorporate the transit model if appropriate
                model_flux += B4 * modval;
            }

            // accumulate the log-likelihood of the data for the models in shared memory
            double resid = model_flux - lc_f;
            double e_term = -0.5 * log(2.0 * M_PI / lc_w);
            sm_ll_tr[threadIdx.x] += (-0.5 * resid * resid * lc_w) + e_term;
        }

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // sum reduction of the log-likelihood array in shared memory
        warpSumReductiond(sm_ll_tr, threadIdx.x);

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // record the log-likelihood of the quad+transit model
        ll_qtr[arr2d_ptr] = (double) sm_ll_tr[0];
    }
}

// detrender - quadratic plus transit fit
__global__ void detrender_calc_IC(
    const double * ll_quad,  // quadratic log-likelihood array
    const double * ll_qtr,  // quadratic+transit log-likelihood array
    const int * num_pts,  // number of data points array
    const int min_obs_in_window,  // the minimum acceptable number of observations in the window
    const int IC_type,  // Type of information criterion: 0 is Bayesian, 1 is Akaike
    const int n_durations,  // the number of durations
    const int t0_stride_count,  // number of reference time strides
    double * delta_IC  // the information criterion difference array (to be filled)
){
    // compute t0 and duration indices
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int did = threadIdx.y + blockIdx.y * blockDim.y;
    if ((tid >= t0_stride_count) || (did >= n_durations)) return;

    // 2d output array pointer
    int arr2d_ptr = tid + t0_stride_count * did;

    // nullify this element if there are too few data points in the window
    int _num_pts = num_pts[tid];
    if (_num_pts < min_obs_in_window){
        delta_IC[arr2d_ptr] = nan(0);
        return;
    }

    // compute the information criteria
    double IC_quad, IC_qtr;
    if (IC_type == 0) {
        // Bayesian
        IC_qtr = 5 * log(1.0 * _num_pts) - 2 * ll_qtr[arr2d_ptr];
        IC_quad = 4 * log(1.0 * _num_pts) - 2 * ll_quad[tid];
    } else if (IC_type == 1) {
        // Akaike
        IC_qtr = 2 * 5 - 2 * ll_qtr[arr2d_ptr];
        IC_quad = 2 * 4 - 2 * ll_quad[tid];
    }
    // record the BIC difference and the log-likelihood of the transit model
    delta_IC[arr2d_ptr] = (double) (IC_quad - IC_qtr);
}

// detrender - get trend of light curve
__global__ void detrender_fit_trend(
    const double * time,  // offset time array
    const double * flux,  // offset flux array
    const double * wght,  // offset flux weight array
    const double * model,  // transit(s) model array
    const int kernel_half_width,  // width of the detrending kernel in samples
    const int min_obs_in_window,  // the minimum acceptable number of observations in the window
    const int lc_size,  // number of light curve elements
    double * trend  // the output trend array
){
    // light curve element index
    const int lc_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (lc_idx >= lc_size) return;

    // some accumulators
    double sw = 0.0;
    double swx = 0.0;
    double swy = 0.0;
    double swxx = 0.0;
    double swxy = 0.0;
    double swxxx = 0.0;
    double swxxy = 0.0;
    double swxxxx = 0.0;
    double stjwj = 0.0;
    double sttjwj = 0.0;
    double stjwjxj = 0.0;
    double stjwjyj = 0.0;
    double stjwjxxj = 0.0;
    int num_pts = 0;

    // loop through the kernel window
    for (int pt_idx = (lc_idx - kernel_half_width) ;
             pt_idx <= (lc_idx + kernel_half_width);
             pt_idx += 1){
        if (pt_idx < 0) continue;
        if (pt_idx >= lc_size) break;

        // skip if the light curve point has zero weight
        double w = (double) wght[pt_idx];
        if (w == 0.0) continue;

        // grab the values of time and flux to make the following code more readable
        // (assume the compiler is smart)
        double dt = (double) time[pt_idx] - time[lc_idx];
        double f = (double) flux[pt_idx];
        // also grab the transit(s) model value
        double m = (double) model[pt_idx];

        // accumulate various values
        sw += w;
        swx += w * dt;
        swy += w * f;
        swxx += w * dt * dt;
        swxy += w * dt * f;
        swxxx += w * dt * dt * dt;
        swxxy += w * dt * dt * f;
        swxxxx += w * dt * dt * dt * dt;
        stjwj += w * m;
        sttjwj += w * m * m;
        stjwjxj += w * m * dt;
        stjwjyj += w * m * f;
        stjwjxxj += w * m * dt * dt;
        num_pts += 1;
    }

    // skip the rest of the loop if too few observations
    if (num_pts < min_obs_in_window){
        trend[lc_idx] = nan(0);
        return;
    };

    // calculate the least squares parameters for the model
    // only interested in the baseline flux at the current time point (which has dT=0)
    // i.e. we're only interested in B3, since the other 2 terms (3 if window includes
    // a transit) are zero.
    double B3;
    if (stjwj > 0.0){
        // contains transit, use these eqns
        // B1 = (stjwj*stjwj*swxx*swxxy - stjwj*stjwj*swxxx*swxy - 2*stjwj*stjwjxj*swx*swxxy + stjwj*stjwjxj*swxx*swxy + stjwj*stjwjxj*swxxx*swy + stjwj*stjwjxxj*swx*swxy - stjwj*stjwjxxj*swxx*swy + stjwj*stjwjyj*swx*swxxx - stjwj*stjwjyj*swxx*swxx + stjwjxj*stjwjxj*sw*swxxy - stjwjxj*stjwjxj*swxx*swy - stjwjxj*stjwjxxj*sw*swxy + stjwjxj*stjwjxxj*swx*swy - stjwjxj*stjwjyj*sw*swxxx + stjwjxj*stjwjyj*swx*swxx + stjwjxxj*stjwjyj*sw*swxx - stjwjxxj*stjwjyj*swx*swx - sttjwj*sw*swxx*swxxy + sttjwj*sw*swxxx*swxy + sttjwj*swx*swx*swxxy - sttjwj*swx*swxx*swxy - sttjwj*swx*swxxx*swy + sttjwj*swxx*swxx*swy)
        //     /(stjwj*stjwj*swxx*swxxxx - stjwj*stjwj*swxxx*swxxx - 2*stjwj*stjwjxj*swx*swxxxx + 2*stjwj*stjwjxj*swxx*swxxx + 2*stjwj*stjwjxxj*swx*swxxx - 2*stjwj*stjwjxxj*swxx*swxx + stjwjxj*stjwjxj*sw*swxxxx - stjwjxj*stjwjxj*swxx*swxx - 2*stjwjxj*stjwjxxj*sw*swxxx + 2*stjwjxj*stjwjxxj*swx*swxx + stjwjxxj*stjwjxxj*sw*swxx - stjwjxxj*stjwjxxj*swx*swx - sttjwj*sw*swxx*swxxxx + sttjwj*sw*swxxx*swxxx + sttjwj*swx*swx*swxxxx - 2*sttjwj*swx*swxx*swxxx + sttjwj*swxx*swxx*swxx);
        // B2 = (-stjwj*stjwj*swxxx*swxxy + stjwj*stjwj*swxxxx*swxy + stjwj*stjwjxj*swxx*swxxy - stjwj*stjwjxj*swxxxx*swy + stjwj*stjwjxxj*swx*swxxy - 2*stjwj*stjwjxxj*swxx*swxy + stjwj*stjwjxxj*swxxx*swy - stjwj*stjwjyj*swx*swxxxx + stjwj*stjwjyj*swxx*swxxx - stjwjxj*stjwjxxj*sw*swxxy + stjwjxj*stjwjxxj*swxx*swy + stjwjxj*stjwjyj*sw*swxxxx - stjwjxj*stjwjyj*swxx*swxx + stjwjxxj*stjwjxxj*sw*swxy - stjwjxxj*stjwjxxj*swx*swy - stjwjxxj*stjwjyj*sw*swxxx + stjwjxxj*stjwjyj*swx*swxx + sttjwj*sw*swxxx*swxxy - sttjwj*sw*swxxxx*swxy - sttjwj*swx*swxx*swxxy + sttjwj*swx*swxxxx*swy + sttjwj*swxx*swxx*swxy - sttjwj*swxx*swxxx*swy)
        //     /(stjwj*stjwj*swxx*swxxxx - stjwj*stjwj*swxxx*swxxx - 2*stjwj*stjwjxj*swx*swxxxx + 2*stjwj*stjwjxj*swxx*swxxx + 2*stjwj*stjwjxxj*swx*swxxx - 2*stjwj*stjwjxxj*swxx*swxx + stjwjxj*stjwjxj*sw*swxxxx - stjwjxj*stjwjxj*swxx*swxx - 2*stjwjxj*stjwjxxj*sw*swxxx + 2*stjwjxj*stjwjxxj*swx*swxx + stjwjxxj*stjwjxxj*sw*swxx - stjwjxxj*stjwjxxj*swx*swx - sttjwj*sw*swxx*swxxxx + sttjwj*sw*swxxx*swxxx + sttjwj*swx*swx*swxxxx - 2*sttjwj*swx*swxx*swxxx + sttjwj*swxx*swxx*swxx);
        B3 = (stjwj*stjwjxj*swxxx*swxxy - stjwj*stjwjxj*swxxxx*swxy - stjwj*stjwjxxj*swxx*swxxy + stjwj*stjwjxxj*swxxx*swxy + stjwj*stjwjyj*swxx*swxxxx - stjwj*stjwjyj*swxxx*swxxx - stjwjxj*stjwjxj*swxx*swxxy + stjwjxj*stjwjxj*swxxxx*swy + stjwjxj*stjwjxxj*swx*swxxy + stjwjxj*stjwjxxj*swxx*swxy - 2*stjwjxj*stjwjxxj*swxxx*swy - stjwjxj*stjwjyj*swx*swxxxx + stjwjxj*stjwjyj*swxx*swxxx - stjwjxxj*stjwjxxj*swx*swxy + stjwjxxj*stjwjxxj*swxx*swy + stjwjxxj*stjwjyj*swx*swxxx - stjwjxxj*stjwjyj*swxx*swxx - sttjwj*swx*swxxx*swxxy + sttjwj*swx*swxxxx*swxy + sttjwj*swxx*swxx*swxxy - sttjwj*swxx*swxxx*swxy - sttjwj*swxx*swxxxx*swy + sttjwj*swxxx*swxxx*swy)
            /(stjwj*stjwj*swxx*swxxxx - stjwj*stjwj*swxxx*swxxx - 2*stjwj*stjwjxj*swx*swxxxx + 2*stjwj*stjwjxj*swxx*swxxx + 2*stjwj*stjwjxxj*swx*swxxx - 2*stjwj*stjwjxxj*swxx*swxx + stjwjxj*stjwjxj*sw*swxxxx - stjwjxj*stjwjxj*swxx*swxx - 2*stjwjxj*stjwjxxj*sw*swxxx + 2*stjwjxj*stjwjxxj*swx*swxx + stjwjxxj*stjwjxxj*sw*swxx - stjwjxxj*stjwjxxj*swx*swx - sttjwj*sw*swxx*swxxxx + sttjwj*sw*swxxx*swxxx + sttjwj*swx*swx*swxxxx - 2*sttjwj*swx*swxx*swxxx + sttjwj*swxx*swxx*swxx);
        // B4 = (stjwj*swx*swxxx*swxxy - stjwj*swx*swxxxx*swxy - stjwj*swxx*swxx*swxxy + stjwj*swxx*swxxx*swxy + stjwj*swxx*swxxxx*swy - stjwj*swxxx*swxxx*swy - stjwjxj*sw*swxxx*swxxy + stjwjxj*sw*swxxxx*swxy + stjwjxj*swx*swxx*swxxy - stjwjxj*swx*swxxxx*swy - stjwjxj*swxx*swxx*swxy + stjwjxj*swxx*swxxx*swy + stjwjxxj*sw*swxx*swxxy - stjwjxxj*sw*swxxx*swxy - stjwjxxj*swx*swx*swxxy + stjwjxxj*swx*swxx*swxy + stjwjxxj*swx*swxxx*swy - stjwjxxj*swxx*swxx*swy - stjwjyj*sw*swxx*swxxxx + stjwjyj*sw*swxxx*swxxx + stjwjyj*swx*swx*swxxxx - 2*stjwjyj*swx*swxx*swxxx + stjwjyj*swxx*swxx*swxx)
        //     /(stjwj*stjwj*swxx*swxxxx - stjwj*stjwj*swxxx*swxxx - 2*stjwj*stjwjxj*swx*swxxxx + 2*stjwj*stjwjxj*swxx*swxxx + 2*stjwj*stjwjxxj*swx*swxxx - 2*stjwj*stjwjxxj*swxx*swxx + stjwjxj*stjwjxj*sw*swxxxx - stjwjxj*stjwjxj*swxx*swxx - 2*stjwjxj*stjwjxxj*sw*swxxx + 2*stjwjxj*stjwjxxj*swx*swxx + stjwjxxj*stjwjxxj*sw*swxx - stjwjxxj*stjwjxxj*swx*swx - sttjwj*sw*swxx*swxxxx + sttjwj*sw*swxxx*swxxx + sttjwj*swx*swx*swxxxx - 2*sttjwj*swx*swxx*swxxx + sttjwj*swxx*swxx*swxx);
    } else {
        // no transit, use these eqns
        // B1 = (sw*swxx*swxxy - sw*swxxx*swxy - swx*swx*swxxy + swx*swxx*swxy + swx*swxxx*swy - swxx*swxx*swy)
        //     /(sw*swxx*swxxxx - sw*swxxx*swxxx - swx*swx*swxxxx + 2*swx*swxx*swxxx - swxx*swxx*swxx);
        // B2 = (-sw*swxxx*swxxy + sw*swxxxx*swxy + swx*swxx*swxxy - swx*swxxxx*swy - swxx*swxx*swxy + swxx*swxxx*swy)
        //     /(sw*swxx*swxxxx - sw*swxxx*swxxx - swx*swx*swxxxx + 2*swx*swxx*swxxx - swxx*swxx*swxx);
        B3 = (swx*swxxx*swxxy - swx*swxxxx*swxy - swxx*swxx*swxxy + swxx*swxxx*swxy + swxx*swxxxx*swy - swxxx*swxxx*swy)
            /(sw*swxx*swxxxx - sw*swxxx*swxxx - swx*swx*swxxxx + 2*swx*swxx*swxxx - swxx*swxx*swxx);
    }

    // record the trend
    trend[lc_idx] = B3;
}

// linear search
__global__ void linear_search(
    const float * time,  // offset time array
    const float * flux,  // offset flux array
    const float * wght,  // offset flux weight array
    const float cadence,  // the cadence of the light curve
    const int lc_size,  // number of light curve elements
    const float * tmodel,  // offset flux transit model array
    const int tm_size,  // number of transit model elements
    const float * durations,  // the duration array
    const int n_durations,  // the number of durations
    const float t0_stride_length,  // the number of reference times per duration
    const int t0_stride_count,  // number of reference time strides
    float * like_ratio,  // the likelihood ratio array (to be filled)
    float * depth,  // the depth array (to be filled)
    float * vdepth  // the depth variance array (to be filled)
){
    // specify the shared memory array locations and types
    float * sm_tmodel = (float*)sm;
    float * sm1 = (float*)&sm_tmodel[tm_size];
    float * sm2 = (float*)&sm1[blockDim.x];

    // read the transit model into shared memory
    for (int i = 0; i < tm_size; i += blockDim.x){
        int sm_idx = i + threadIdx.x;
        if (sm_idx >= tm_size) break;
        sm_tmodel[sm_idx] = tmodel[sm_idx];
    }

    // duration index
    const int dur_id = blockIdx.y;

    // grab the duration
    if (dur_id >= n_durations) return;
    const float duration = durations[dur_id];

    // Stride through the reference time steps
    // Each block reads the transit model from global to shared memory once.
    // Striding through the reference time steps means each block computes the likelihood
    // ratio of multiple reference time steps, but it still only reads the transit model
    // once, so the total number of reads from global memory is reduced.
    // Testing indicates this optimisation halves the compute time for a 2yr
    // light curve at 10 min cadence.
    for (int s = 0; s < t0_stride_count; s += gridDim.x) {
        // t0 number
        int t0_num = blockIdx.x + s;
        if (t0_num >= t0_stride_count) return;

        // 2d output array pointer
        int arr2d_ptr = t0_num + t0_stride_count * dur_id;

        // calculate ts, the transit start time
        float ts = t0_num * t0_stride_length - 0.5 * duration;

        // zero out the non transit model portion of the shared memory
        sm1[threadIdx.x] = 0.0f;
        sm2[threadIdx.x] = 0.0f;

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // compute the index of the first and last in-transit points
        int itr_frst_idx = lrintf(ceilf(ts / cadence)) ;
        int itr_last_idx = lrintf(floorf((ts+duration) / cadence));
        // clip first indices to start of light curve
        itr_frst_idx = max(itr_frst_idx, 0);
        // clip last index to end of light curve
        itr_last_idx = min(itr_last_idx, lc_size-1);
        // width of the transit window
        int itr_size = itr_last_idx - itr_frst_idx + 1;

        // loop over the light curve in the transit window
        for (int i = 0; i <= itr_size; i += blockDim.x){
            int lc_idx = itr_frst_idx + i + threadIdx.x;

            // it shouldn't be because we clipped but just in case...
            if (lc_idx < 0) continue;
            if (lc_idx >= lc_size) break;

            // skip if the light curve point has infinite error (i.e. zero weight)
            if (wght[lc_idx] == 0.0f) continue;

            // find the nearest model point index
            int model_idx = lrintf(( time[lc_idx] - ts ) / duration * tm_size);
            // just in case we're out of bounds:
            if ((model_idx < 0) || (model_idx >= tm_size)) continue;
            float modval = sm_tmodel[model_idx];

            // transit depth implied by this light curve point
            float local_depth = flux[lc_idx] / modval;
            // weight of this light curve point
            float local_weight = modval * modval * wght[lc_idx];

            // accumulate the depth and weight
            sm1[threadIdx.x] += local_depth * local_weight;
            sm2[threadIdx.x] += local_weight;
        }

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // sum reduction of the second and third arrays in shared memory
        warpSumReductionf(sm1, threadIdx.x);
        warpSumReductionf(sm2, threadIdx.x);

        // calculate the maximum likelihood transit depth
        float wav_depth = sm1[0] / sm2[0];
        float var_depth = 1.0f / sm2[0];
        // send the depth, variance and obs count to the output arrays
        depth[arr2d_ptr] = wav_depth;
        vdepth[arr2d_ptr] = var_depth;

        // nothing more to do if there were no valid data in the window
        if (isnan(wav_depth)){
            like_ratio[arr2d_ptr] = nanf(0);
            continue;
        };

        // zero out the non transit model portion of the shared memory again
        sm1[threadIdx.x] = 0.0f;
        sm2[threadIdx.x] = 0.0f;

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // now loop through the transit window again and calculate the log-likelihood
        for (int i = 0; i <= itr_size; i += blockDim.x){
            int lc_idx = itr_frst_idx + i + threadIdx.x;

            // it shouldn't be because we clipped but just in case...
            if (lc_idx < 0) continue;
            if (lc_idx >= lc_size) break;

            // skip if the light curve point has infinite error (i.e. zero weight)
            if (wght[lc_idx] == 0.0f) continue;

            // find the nearest model point index
            int model_idx = lrintf(( time[lc_idx] - ts ) / duration * tm_size);
            // just in case we're out of bounds:
            if ((model_idx < 0) || (model_idx >= tm_size)) continue;
            float modval = sm_tmodel[model_idx];

            // log-likelihood of in-transit points
            float resid = modval * wav_depth - flux[lc_idx];
            sm1[threadIdx.x] += (-0.5f * resid * resid * wght[lc_idx]);

            // the second part of the log-likelihood is:
            //     -0.5 * log(2 * pi * error^2)
            // but it's unnecessary to add it only to subtract it again,
            // might as well avoid that expensive log operation!

            // subtract constant flux log-likelihood of in-transit points
            // to get the likelihood ratio
            sm1[threadIdx.x] -= (-0.5f * flux[lc_idx] * flux[lc_idx] * wght[lc_idx]);
        }

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // sum reduction of the second array in shared memory
        warpSumReductionf(sm1, threadIdx.x);

        // store the likelihood ratio in the output array
        like_ratio[arr2d_ptr] = sm1[0];
    }
}

// light curve resampling - stage 1
__global__ void resample_k1(
    const double * time,  // input light curve offset time array (i.e. time - ref time)
    const double * flux,  // input light curve flux array
    const double * ferr,  // input light curve flux error array
    const double cadence,  // desired output cadence
    const int n_elem,  // number of elements in input light curve
    double * sum_of_weighted_flux,  // array of sum(f*w)
    double * sum_of_weights  // array of sum(w)
){
    const int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_idx >= n_elem) return;

    // skip point if zero weight
    if (isinf(ferr[x_idx])) return;

    // calculate the weight of this point
    double weight = 1.0 / (ferr[x_idx] * ferr[x_idx]);

    // index of current light curve point in output light curve array
    int idx_out = lrint(time[x_idx] / cadence);

    // incorporate this light curve point into the output light curve
    atomicAdd(&sum_of_weighted_flux[idx_out], flux[x_idx]*weight);
    atomicAdd(&sum_of_weights[idx_out], weight);
}

// light curve resampling - stage 2
__global__ void resample_k2(
    const double * sum_fw,  // array of sum(offset flux * weight)
    const double * sum_w,  // array of sum(weight)
    double * rflux,  // array of weighted average flux
    double * eflux,  // array of error on weighted average flux
    const int n_elem
){
    const int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_idx >= n_elem) return;

    // compute inverse variance weighted relative flux
    rflux[x_idx] = sum_fw[x_idx] / sum_w[x_idx];

    // compute error on above
    eflux[x_idx] = rsqrt(sum_w[x_idx]);
}

// period search - kernel 1 (joint likelihood-ratio compute and first stage reduction)
__global__ void periodic_search_k1(
    const float period_strides,  // period in strides
    const float * in_like_ratio,  // the previously computed likelihood ratios
    const float * in_depth,  // the previously computed max-likelihood depths
    const float * in_var_depth,  // the previously computed max-likelihood depth variance
    const int long_t0_count,  // reference time stride count across whole light curve
    const int duration_idx_first,  // the index of the first duration to check
    const int duration_idx_last,  // the index of the last duration to check
    const int max_transit_count,  // the maximum possible number of transits for this period
    float * lrat_out,  // the temporary max likelihood ratio array (to be filled)
    float * depth_out,  // the temporary depth array (to be filled)
    float * vdepth_out,  // the temporary depth variance array (to be filled)
    int * d_idx_out,  // the temporary duration index array (to be filled)
    int * t0_idx_out  // the temporary reference time index array (to be filled)
){
    // variable declarations
    bool null = false;

    // pointers for the array in shared memory - split it in half
    float * sm_lr = (float*)&sm;
    float * sm_id = (float*)&sm_lr[blockDim.x];

    // nullify the arrays in shared memory
    sm_lr[threadIdx.x] = nanf(0);
    sm_id[threadIdx.x] = nanf(0);

    // reference time and duration indices
    const int t0_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dur_idx = duration_idx_first + blockIdx.y * blockDim.y + threadIdx.y;
    // nullify this thread if we're out of bounds
    // might as well still allow all the processing to run because warp divergence
    // we can't return yet as we need the threads to participate in the reduction
    // operation later
    if ((t0_idx >= period_strides) || (dur_idx > duration_idx_last)){
        null = true;
    };

    // output array pointer
    const int out2d_ptr = blockIdx.x + gridDim.x * blockIdx.y;

    // accumulators
    float _sum_dw = 0.0;
    float _sum_w = 0.0;
    // accumulate the number of transits
    int n_transits = 0;
    // loop through the input arrays to determine the maximum likelihood depth
    for (int i = 0; i < max_transit_count; i++){
        // compute the reference time index of this iteration
        int _t0_idx = t0_idx + lrintf(period_strides * i);
        if (_t0_idx >= long_t0_count) break;  // exit the loop if out of bounds

        // pointer into 2d input arrays
        int in2d_ptr = _t0_idx + long_t0_count * dur_idx;

        // do nothing in this iteration if infinite variance
        float _var = in_var_depth[in2d_ptr];
        if (isinf(_var)) continue;

        // inverse variance weight
        float weight = 1.0f / _var;

        // add to accumulators
        _sum_dw += in_depth[in2d_ptr] * weight;
        _sum_w += weight;
        n_transits += 1;
    }

    // nullify this thread if there were fewer than 2 transits
    if (n_transits < 2){
        null = true;
    }

    // compute the maximum likelihood depth
    float wav_depth = _sum_dw / _sum_w;
    float var_depth = 1.0f / _sum_w;

    // accumulators
    float _sum_lrats_sgl = 0.0;
    float _sum_logs = 0.0;
    // loop through the input arrays again to compute the joint-likelihood
    for (int i = 0; i < max_transit_count; i++){
        // simply taking the nearest element, could interpolate and probably wouldn't
        // be too expensive, but this should be adequate - test this!

        // compute the reference time index of this iteration
        int _t0_idx = t0_idx + lrintf(period_strides * i);
        if (_t0_idx >= long_t0_count) break;  // exit the loop if out of bounds

        // pointer into 2d input arrays
        int in2d_ptr = _t0_idx + long_t0_count * dur_idx;

        // do nothing in this iteration if infinite variance
        float _var_depth = in_var_depth[in2d_ptr];
        if (isinf(_var_depth)) continue;

        // depth
        float _depth = in_depth[in2d_ptr];
        // likelihood ratio
        float _lrat = in_like_ratio[in2d_ptr];

        // add single transit likelihood ratio to accumulator
        _sum_lrats_sgl += _lrat;

        // compute the second part
        float ddepth = _depth - wav_depth;
        _sum_logs += (ddepth * ddepth / _var_depth);
    }

    // combine the accumulators compute the (joint) likelihood ratio
    // only do this if we've not nullified the thread
    if (!null){
        sm_lr[threadIdx.x] = _sum_lrats_sgl - 0.5 * _sum_logs;
        sm_id[threadIdx.x] = 1.0f * threadIdx.x;
    }
    // if you were to add this to the constant model log-likelihood you
    // would obtain the joint-likelihood

    // now do the block level max reduction operation, also recording the pointers
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (threadIdx.x < s) {
            if ((isnan(sm_lr[threadIdx.x])) \
                || ((sm_lr[threadIdx.x + s] > sm_lr[threadIdx.x]) \
                    && (!isnan(sm_lr[threadIdx.x+s])))) {
                sm_lr[threadIdx.x] = sm_lr[threadIdx.x+s];
                sm_id[threadIdx.x] = sm_id[threadIdx.x+s];
            }
        }
        __syncthreads();
    }

    // only the max-likelihood ratio thread records its results
    const int best_tid = lrintf(sm_id[0]);
    if (threadIdx.x == best_tid) {
        lrat_out[out2d_ptr] = sm_lr[threadIdx.x];
        depth_out[out2d_ptr] = wav_depth;
        vdepth_out[out2d_ptr] = var_depth;
        d_idx_out[out2d_ptr] = dur_idx;
        t0_idx_out[out2d_ptr] = t0_idx;
    }
}

// periodic search - kernel 2 (second-stage reduction operation)
__global__ void periodic_search_k2(
    const float * lrat_in,  // max likelihood ratio array
    const float * depth_in,  // depth array
    const float * vdepth_in,  // depth variance array
    const int * d_idx_in,  // duration index array
    const int * t0_idx_in,  // reference time index array
    float * lrat_out,  // max likelihood ratio array (single element - to be filled)
    float * depth_out,  // depth array (single element - to be filled)
    float * vdepth_out,  // depth variance array (single element - to be filled)
    int * d_idx_out,  // duration index array (single element - to be filled)
    int * t0_idx_out,  // reference time index array (single element - to be filled)
    const int in_arr_len  // length of input arrays
){
    // pointers for the 2 arrays in shared memory
    float * sm_lr = (float*)&sm;
    float * sm_id = (float*)&sm_lr[blockDim.x];

    // nullify the arrays in shared memory
    sm_lr[threadIdx.x] = nanf(0);
    sm_id[threadIdx.x] = nanf(0);

    // cycle through the likelihood ratio input array
    // record the max and index in shared mem
    for (int i = 0 ; i <= in_arr_len ; i += blockDim.x){
        int idx = i + threadIdx.x;
        if (idx >= in_arr_len) break;
        if (   (isnan(sm_lr[threadIdx.x]))
            || ((lrat_in[idx] > sm_lr[threadIdx.x]) && (!isnan(lrat_in[idx])))
            ){
            sm_lr[threadIdx.x] = lrat_in[idx];
            sm_id[threadIdx.x] = 1.0f * idx;
        }
    }
    __syncthreads();

    // final reduction through shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (threadIdx.x < s) {
            if (   (isnan(sm_lr[threadIdx.x]))
                || ((sm_lr[threadIdx.x + s] > sm_lr[threadIdx.x]) && (!isnan(sm_lr[threadIdx.x+s])))
                ) {
                sm_lr[threadIdx.x] = sm_lr[threadIdx.x+s];
                sm_id[threadIdx.x] = sm_id[threadIdx.x+s];
            }
        }
        __syncthreads();
    }

    // record the maximum likelihood parameters
    const int best_idx = lrintf(sm_id[0]);
    lrat_out[0] = lrat_in[best_idx];
    depth_out[0] = depth_in[best_idx];
    vdepth_out[0] = vdepth_in[best_idx];
    d_idx_out[0] = d_idx_in[best_idx];
    t0_idx_out[0] = t0_idx_in[best_idx];

}
