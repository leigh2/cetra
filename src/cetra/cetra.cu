#include <float.h>

__device__ void warpSumReductionf(volatile float* sdata, int tid, int offset) {
    //sdata[offset + tid] += sdata[offset + tid + 32];
    sdata[offset + tid] += sdata[offset + tid + 16];
    sdata[offset + tid] += sdata[offset + tid + 8];
    sdata[offset + tid] += sdata[offset + tid + 4];
    sdata[offset + tid] += sdata[offset + tid + 2];
    sdata[offset + tid] += sdata[offset + tid + 1];
}

// detrender - get BICs
__global__ void detrender_k1(
    const float * time,  // offset time array
    const float * flux,  // offset flux array
    const float * wght,  // offset flux weight array
    const float kernel_width,  // width of the detrending kernel in days
    const float cadence,  // the cadence of the light curve
    const int lc_size,  // number of light curve elements
    const float * tmodel,  // offset flux transit model array
    const int tm_size,  // number of transit model elements
    const float * durations,  // the duration array
    const int n_durations,  // the number of durations
    const float ts_stride_length,  // the number of start times per duration
    const int ts_stride_count,  // number of start time strides
    float * BIC_ratio  // the likelihood ratio array (to be filled)
){

    // open the array in shared memory
    extern __shared__ float sm[];

    // read the transit model into shared memory
    for (int i = 0; i < tm_size; i += blockDim.x){
        int sm_idx = i + threadIdx.x;
        if (sm_idx >= tm_size) break;
        sm[sm_idx] = tmodel[sm_idx];
    }

    // index of this thread into the additional shared memory arrays
    const int sw_idx = threadIdx.x + tm_size;
    const int swx_idx = sw_idx + blockDim.x;
    const int swy_idx = swx_idx + blockDim.x;
    const int swxx_idx = swy_idx + blockDim.x;
    const int swxy_idx = swxx_idx + blockDim.x;
    const int swxxx_idx = swxy_idx + blockDim.x;
    const int swxxy_idx = swxxx_idx + blockDim.x;
    const int swxxxx_idx = swxxy_idx + blockDim.x;
    const int stjwj_idx = swxxxx_idx + blockDim.x;
    const int sttjwj_idx = stjwj_idx + blockDim.x;
    const int stjwjxj_idx = sttjwj_idx + blockDim.x;
    const int stjwjyj_idx = stjwjxj_idx + blockDim.x;
    const int stjwjxxj_idx = stjwjyj_idx + blockDim.x;
    const int num_pts_idx = stjwjxxj_idx + blockDim.x;
    const int BIC_tr_idx = num_pts_idx + blockDim.x;
    const int BIC_nt_idx = BIC_tr_idx + blockDim.x;

    // duration index
    const int dur_id = blockIdx.y;

    // grab the duration
    const float duration = durations[dur_id];

    // compute the width of the detrending boundary
    const float dt_border = (0.5 * (kernel_width - duration)) / cadence;

    // Stride through the start time steps
    // Each block reads the transit model from global to shared memory once.
    // Striding through the start time steps means each block computes the likelihood
    // ratio of multiple start time steps, but it still only reads the transit model
    // once, so the total number of reads from global memory is reduced.
    // Testing indicates this optimisation halves the compute time for a 2yr
    // light curve at 10 min cadence.
    for (int s = 0; s < ts_stride_count; s += gridDim.x) {
        // ts number
        int ts_num = blockIdx.x + s;
        if (ts_num >= ts_stride_count) return;

        // 2d output array pointer
        int arr2d_ptr = ts_num + ts_stride_count * dur_id;

        // calculate ts
        float ts = ts_num * ts_stride_length;

        // zero out the additional arrays in shared memory
        sm[sw_idx] = 0.0f;
        sm[swx_idx] = 0.0f;
        sm[swy_idx] = 0.0f;
        sm[swxx_idx] = 0.0f;
        sm[swxy_idx] = 0.0f;
        sm[swxxx_idx] = 0.0f;
        sm[swxxy_idx] = 0.0f;
        sm[swxxxx_idx] = 0.0f;
        sm[stjwj_idx] = 0.0f;
        sm[sttjwj_idx] = 0.0f;
        sm[stjwjxj_idx] = 0.0f;
        sm[stjwjyj_idx] = 0.0f;
        sm[stjwjxxj_idx] = 0.0f;
        sm[num_pts_idx] = 0.0f;
        sm[BIC_tr_idx] = 0.0f;
        sm[BIC_nt_idx] = 0.0f;

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // compute the indices of the first and last in-transit points
        int itr_frst_idx = lrintf(ceilf(ts / cadence)) ;
        int itr_last_idx = lrintf(floorf((ts+duration) / cadence));
        // compute the indices of the first and last kernel points
        int dtr_frst_idx = itr_frst_idx - dt_border;
        int dtr_last_idx = itr_last_idx + dt_border;
        // clip last indices to end of light curve
        itr_last_idx = min(itr_last_idx, lc_size-1);
        dtr_last_idx = min(dtr_last_idx, lc_size-1);
        // width of the detrending window
        int dtr_size = dtr_last_idx - dtr_frst_idx + 1;

        // loop over the light curve in the detrending window
        for (int i = 0; i <= dtr_size; i += blockDim.x){
            int lc_idx = dtr_frst_idx + i + threadIdx.x;
            if (lc_idx >= lc_size) break;

            // skip if the light curve point has infinite error (i.e. zero weight)
            float w = wght[lc_idx];
            if (w == 0.0f) continue;

            // grab the values of time and flux to make the following code more readable (assume the compiler is smart)
            float t = time[lc_idx];
            float f = flux[lc_idx];

            // accumulate various values
            sm[sw_idx] += w;
            sm[swx_idx] += w * t;
            sm[swy_idx] += w * f;
            sm[swxx_idx] += w * t * t;
            sm[swxy_idx] += w * t * f;
            sm[swxxx_idx] += w * t * t * t;
            sm[swxxy_idx] += w * t * t * f;
            sm[swxxxx_idx] += w * t * t * t * t;
            sm[num_pts_idx] += 1.0f;

            // is this point in the transit window?
            if ((lc_idx >= itr_frst_idx) & (lc_idx <= itr_last_idx)) {
                // find the nearest model point index
                int model_idx = lrintf(( t - ts ) / duration * tm_size);
                // just in case we're out of bounds:
                if ((model_idx < 0) || (model_idx >= tm_size)) continue;
                float modval = sm[model_idx];

                // accumulate some additional values
                sm[stjwj_idx] += w * modval;
                sm[sttjwj_idx] += w * modval * modval;
                sm[stjwjxj_idx] += w * modval * t;
                sm[stjwjyj_idx] += w * modval * f;
                sm[stjwjxxj_idx] += w * modval * t * t;
            }
        }

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // sum reduction of the additional arrays in shared memory
        warpSumReductionf(sm, threadIdx.x, sw_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, swx_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, swy_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, swxx_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, swxy_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, swxxx_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, swxxy_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, swxxxx_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, stjwj_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, sttjwj_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, stjwjxj_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, stjwjyj_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, stjwjxxj_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, num_pts_idx - threadIdx.x);

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // pull these values out
        float sw = sm[sw_idx - threadIdx.x];
        float swx = sm[swx_idx - threadIdx.x];
        float swy = sm[swy_idx - threadIdx.x];
        float swxx = sm[swxx_idx - threadIdx.x];
        float swxy = sm[swxy_idx - threadIdx.x];
        float swxxx = sm[swxxx_idx - threadIdx.x];
        float swxxy = sm[swxxy_idx - threadIdx.x];
        float swxxxx = sm[swxxxx_idx - threadIdx.x];
        float stjwj = sm[stjwj_idx - threadIdx.x];
        float sttjwj = sm[sttjwj_idx - threadIdx.x];
        float stjwjxj = sm[stjwjxj_idx - threadIdx.x];
        float stjwjyj = sm[stjwjyj_idx - threadIdx.x];
        float stjwjxxj = sm[stjwjxxj_idx - threadIdx.x];
        float num_pts = sm[num_pts_idx - threadIdx.x];

        // nothing more to do if there were insufficient valid data in the window
        if (num_pts < 10){
            BIC_ratio[arr2d_ptr] = nanf(0);
            continue;
        };

        // calculate the least squares parameters for the transit model
        double B1_tr = (stjwj*stjwj*swxx*swxxy - stjwj*stjwj*swxxx*swxy - 2*stjwj*stjwjxj*swx*swxxy + stjwj*stjwjxj*swxx*swxy + stjwj*stjwjxj*swxxx*swy + stjwj*stjwjxxj*swx*swxy - stjwj*stjwjxxj*swxx*swy + stjwj*stjwjyj*swx*swxxx - stjwj*stjwjyj*swxx*swxx + stjwjxj*stjwjxj*sw*swxxy - stjwjxj*stjwjxj*swxx*swy - stjwjxj*stjwjxxj*sw*swxy + stjwjxj*stjwjxxj*swx*swy - stjwjxj*stjwjyj*sw*swxxx + stjwjxj*stjwjyj*swx*swxx + stjwjxxj*stjwjyj*sw*swxx - stjwjxxj*stjwjyj*swx*swx - sttjwj*sw*swxx*swxxy + sttjwj*sw*swxxx*swxy + sttjwj*swx*swx*swxxy - sttjwj*swx*swxx*swxy - sttjwj*swx*swxxx*swy + sttjwj*swxx*swxx*swy)/(stjwj*stjwj*swxx*swxxxx - stjwj*stjwj*swxxx*swxxx - 2*stjwj*stjwjxj*swx*swxxxx + 2*stjwj*stjwjxj*swxx*swxxx + 2*stjwj*stjwjxxj*swx*swxxx - 2*stjwj*stjwjxxj*swxx*swxx + stjwjxj*stjwjxj*sw*swxxxx - stjwjxj*stjwjxj*swxx*swxx - 2*stjwjxj*stjwjxxj*sw*swxxx + 2*stjwjxj*stjwjxxj*swx*swxx + stjwjxxj*stjwjxxj*sw*swxx - stjwjxxj*stjwjxxj*swx*swx - sttjwj*sw*swxx*swxxxx + sttjwj*sw*swxxx*swxxx + sttjwj*swx*swx*swxxxx - 2*sttjwj*swx*swxx*swxxx + sttjwj*swxx*swxx*swxx);
        double B2_tr = (-stjwj*stjwj*swxxx*swxxy + stjwj*stjwj*swxxxx*swxy + stjwj*stjwjxj*swxx*swxxy - stjwj*stjwjxj*swxxxx*swy + stjwj*stjwjxxj*swx*swxxy - 2*stjwj*stjwjxxj*swxx*swxy + stjwj*stjwjxxj*swxxx*swy - stjwj*stjwjyj*swx*swxxxx + stjwj*stjwjyj*swxx*swxxx - stjwjxj*stjwjxxj*sw*swxxy + stjwjxj*stjwjxxj*swxx*swy + stjwjxj*stjwjyj*sw*swxxxx - stjwjxj*stjwjyj*swxx*swxx + stjwjxxj*stjwjxxj*sw*swxy - stjwjxxj*stjwjxxj*swx*swy - stjwjxxj*stjwjyj*sw*swxxx + stjwjxxj*stjwjyj*swx*swxx + sttjwj*sw*swxxx*swxxy - sttjwj*sw*swxxxx*swxy - sttjwj*swx*swxx*swxxy + sttjwj*swx*swxxxx*swy + sttjwj*swxx*swxx*swxy - sttjwj*swxx*swxxx*swy)/(stjwj*stjwj*swxx*swxxxx - stjwj*stjwj*swxxx*swxxx - 2*stjwj*stjwjxj*swx*swxxxx + 2*stjwj*stjwjxj*swxx*swxxx + 2*stjwj*stjwjxxj*swx*swxxx - 2*stjwj*stjwjxxj*swxx*swxx + stjwjxj*stjwjxj*sw*swxxxx - stjwjxj*stjwjxj*swxx*swxx - 2*stjwjxj*stjwjxxj*sw*swxxx + 2*stjwjxj*stjwjxxj*swx*swxx + stjwjxxj*stjwjxxj*sw*swxx - stjwjxxj*stjwjxxj*swx*swx - sttjwj*sw*swxx*swxxxx + sttjwj*sw*swxxx*swxxx + sttjwj*swx*swx*swxxxx - 2*sttjwj*swx*swxx*swxxx + sttjwj*swxx*swxx*swxx);
        double B3_tr = (stjwj*stjwjxj*swxxx*swxxy - stjwj*stjwjxj*swxxxx*swxy - stjwj*stjwjxxj*swxx*swxxy + stjwj*stjwjxxj*swxxx*swxy + stjwj*stjwjyj*swxx*swxxxx - stjwj*stjwjyj*swxxx*swxxx - stjwjxj*stjwjxj*swxx*swxxy + stjwjxj*stjwjxj*swxxxx*swy + stjwjxj*stjwjxxj*swx*swxxy + stjwjxj*stjwjxxj*swxx*swxy - 2*stjwjxj*stjwjxxj*swxxx*swy - stjwjxj*stjwjyj*swx*swxxxx + stjwjxj*stjwjyj*swxx*swxxx - stjwjxxj*stjwjxxj*swx*swxy + stjwjxxj*stjwjxxj*swxx*swy + stjwjxxj*stjwjyj*swx*swxxx - stjwjxxj*stjwjyj*swxx*swxx - sttjwj*swx*swxxx*swxxy + sttjwj*swx*swxxxx*swxy + sttjwj*swxx*swxx*swxxy - sttjwj*swxx*swxxx*swxy - sttjwj*swxx*swxxxx*swy + sttjwj*swxxx*swxxx*swy)/(stjwj*stjwj*swxx*swxxxx - stjwj*stjwj*swxxx*swxxx - 2*stjwj*stjwjxj*swx*swxxxx + 2*stjwj*stjwjxj*swxx*swxxx + 2*stjwj*stjwjxxj*swx*swxxx - 2*stjwj*stjwjxxj*swxx*swxx + stjwjxj*stjwjxj*sw*swxxxx - stjwjxj*stjwjxj*swxx*swxx - 2*stjwjxj*stjwjxxj*sw*swxxx + 2*stjwjxj*stjwjxxj*swx*swxx + stjwjxxj*stjwjxxj*sw*swxx - stjwjxxj*stjwjxxj*swx*swx - sttjwj*sw*swxx*swxxxx + sttjwj*sw*swxxx*swxxx + sttjwj*swx*swx*swxxxx - 2*sttjwj*swx*swxx*swxxx + sttjwj*swxx*swxx*swxx);
        double B4_tr = (stjwj*swx*swxxx*swxxy - stjwj*swx*swxxxx*swxy - stjwj*swxx*swxx*swxxy + stjwj*swxx*swxxx*swxy + stjwj*swxx*swxxxx*swy - stjwj*swxxx*swxxx*swy - stjwjxj*sw*swxxx*swxxy + stjwjxj*sw*swxxxx*swxy + stjwjxj*swx*swxx*swxxy - stjwjxj*swx*swxxxx*swy - stjwjxj*swxx*swxx*swxy + stjwjxj*swxx*swxxx*swy + stjwjxxj*sw*swxx*swxxy - stjwjxxj*sw*swxxx*swxy - stjwjxxj*swx*swx*swxxy + stjwjxxj*swx*swxx*swxy + stjwjxxj*swx*swxxx*swy - stjwjxxj*swxx*swxx*swy - stjwjyj*sw*swxx*swxxxx + stjwjyj*sw*swxxx*swxxx + stjwjyj*swx*swx*swxxxx - 2*stjwjyj*swx*swxx*swxxx + stjwjyj*swxx*swxx*swxx)/(stjwj*stjwj*swxx*swxxxx - stjwj*stjwj*swxxx*swxxx - 2*stjwj*stjwjxj*swx*swxxxx + 2*stjwj*stjwjxj*swxx*swxxx + 2*stjwj*stjwjxxj*swx*swxxx - 2*stjwj*stjwjxxj*swxx*swxx + stjwjxj*stjwjxj*sw*swxxxx - stjwjxj*stjwjxj*swxx*swxx - 2*stjwjxj*stjwjxxj*sw*swxxx + 2*stjwjxj*stjwjxxj*swx*swxx + stjwjxxj*stjwjxxj*sw*swxx - stjwjxxj*stjwjxxj*swx*swx - sttjwj*sw*swxx*swxxxx + sttjwj*sw*swxxx*swxxx + sttjwj*swx*swx*swxxxx - 2*sttjwj*swx*swxx*swxxx + sttjwj*swxx*swxx*swxx);

        // require some minimum depth
        // todo user setting for the minimum depth
        if (B4_tr < 10 * 1e-6) {
            BIC_ratio[arr2d_ptr] = nanf(0);
            continue;
        }

        // calculate the least squares parameters for the non transit model
        double B1_nt = (sw*swxx*swxxy - sw*swxxx*swxy - swx*swx*swxxy + swx*swxx*swxy + swx*swxxx*swy - swxx*swxx*swy)/(sw*swxx*swxxxx - sw*swxxx*swxxx - swx*swx*swxxxx + 2*swx*swxx*swxxx - swxx*swxx*swxx);
        double B2_nt = (-sw*swxxx*swxxy + sw*swxxxx*swxy + swx*swxx*swxxy - swx*swxxxx*swy - swxx*swxx*swxy + swxx*swxxx*swy)/(sw*swxx*swxxxx - sw*swxxx*swxxx - swx*swx*swxxxx + 2*swx*swxx*swxxx - swxx*swxx*swxx);
        double B3_nt = (swx*swxxx*swxxy - swx*swxxxx*swxy - swxx*swxx*swxxy + swxx*swxxx*swxy + swxx*swxxxx*swy - swxxx*swxxx*swy)/(sw*swxx*swxxxx - sw*swxxx*swxxx - swx*swx*swxxxx + 2*swx*swxx*swxxx - swxx*swxx*swxx);

        // now loop through the detrending window again and calculate the log-likelihood
        for (int i = 0; i <= dtr_size; i += blockDim.x){
            int lc_idx = dtr_frst_idx + i + threadIdx.x;
            if (lc_idx >= lc_size) break;

            // skip if the light curve point has infinite error (i.e. zero weight)
            float w = wght[lc_idx];
            if (w == 0.0f) continue;

            // grab the values of time and flux to make the following code more readable (assume the compiler is smart)
            float t = time[lc_idx];
            float f = flux[lc_idx];

            // compute the best fit models for this point
            float tr_flux = B1_tr * t * t + B2_tr * t + B3_tr;  // transit
            float nt_flux = B1_nt * t * t + B2_nt * t + B3_nt;  // non-transit

            // is this point in the transit window?
            if ((lc_idx >= itr_frst_idx) & (lc_idx <= itr_last_idx)) {
                // find the nearest model point index
                int model_idx = lrintf(( t - ts ) / duration * tm_size);
                // just in case we're out of bounds:
                if ((model_idx < 0) || (model_idx >= tm_size)) continue;
                float modval = sm[model_idx];

                // incorporate the transit model
                tr_flux += B4_tr * modval;
            }

            // log-likelihood of in-transit points
            float tr_resid = tr_flux - f;
            float nt_resid = nt_flux - f;
            float e_term = - 0.5f * log(2 * M_PI / w);
            sm[BIC_tr_idx] += (-0.5f * tr_resid * tr_resid * w) + e_term;
            sm[BIC_nt_idx] += (-0.5f * nt_resid * nt_resid * w) + e_term;
        }

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // sum reduction of the BIC arrays in shared memory
        warpSumReductionf(sm, threadIdx.x, BIC_tr_idx - threadIdx.x);
        warpSumReductionf(sm, threadIdx.x, BIC_nt_idx - threadIdx.x);

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // store the BIC ratio in the output array
        float BIC_tr = 5 * log(num_pts) - 2 * sm[BIC_tr_idx - threadIdx.x];
        float BIC_nt = 4 * log(num_pts) - 2 * sm[BIC_nt_idx - threadIdx.x];
        // output
//         float tt = ts + 0.5*duration;B1_tr*tt*tt + B2_tr*tt + B3_tr  ;//BIC_nt - BIC_tr;
        BIC_ratio[arr2d_ptr] = BIC_nt - BIC_tr;
    }
}

// monotransit search
__global__ void monotransit_search(
    const float * time,  // offset time array
    const float * flux,  // offset flux array
    const float * wght,  // offset flux weight array
    const float cadence,  // the cadence of the light curve
    const int lc_size,  // number of light curve elements
    const float * tmodel,  // offset flux transit model array
    const int tm_size,  // number of transit model elements
    const float * durations,  // the duration array
    const int n_durations,  // the number of durations
    const float ts_stride_length,  // the number of start times per duration
    const int ts_stride_count,  // number of start time strides
    float * like_ratio,  // the likelihood ratio array (to be filled)
    float * depth,  // the depth array (to be filled)
    float * vdepth  // the depth variance array (to be filled)
){

    // open the array in shared memory
    extern __shared__ float sm[];

    // read the transit model into shared memory
    for (int i = 0; i < tm_size; i += blockDim.x){
        int sm_idx = i + threadIdx.x;
        if (sm_idx >= tm_size) break;
        sm[sm_idx] = tmodel[sm_idx];
    }

    // index of this thread into the second and third shared memory arrays
    const int sm1_ptr = threadIdx.x + tm_size;
    const int sm2_ptr = sm1_ptr + blockDim.x;

    // duration index
    const int dur_id = blockIdx.y;

    // grab the duration
    const float duration = durations[dur_id];

    // Stride through the start time steps
    // Each block reads the transit model from global to shared memory once.
    // Striding through the start time steps means each block computes the likelihood
    // ratio of multiple start time steps, but it still only reads the transit model
    // once, so the total number of reads from global memory is reduced.
    // Testing indicates this optimisation halves the compute time for a 2yr
    // light curve at 10 min cadence.
    for (int s = 0; s < ts_stride_count; s += gridDim.x) {
        // ts number
        int ts_num = blockIdx.x + s;
        if (ts_num >= ts_stride_count) return;

        // 2d output array pointer
        int arr2d_ptr = ts_num + ts_stride_count * dur_id;

        // calculate ts
        float ts = ts_num * ts_stride_length;

        // zero out the second and third arrays in shared memory
        sm[sm1_ptr] = 0.0f;
        sm[sm2_ptr] = 0.0f;

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // compute the index of the first and last in-transit points
        int itr_frst_idx = lrintf(ceilf(ts / cadence)) ;
        int itr_last_idx = lrintf(floorf((ts+duration) / cadence));
        // clip last index to end of light curve
        itr_last_idx = min(itr_last_idx, lc_size-1);
        // width of the transit window
        int itr_size = itr_last_idx - itr_frst_idx + 1;

        // loop over the light curve in the transit window
        for (int i = 0; i <= itr_size; i += blockDim.x){
            int lc_idx = itr_frst_idx + i + threadIdx.x;
            if (lc_idx >= lc_size) break;

            // skip if the light curve point has infinite error (i.e. zero weight)
            float w = wght[lc_idx];
            if (w == 0.0f) continue;

            // find the nearest model point index
            int model_idx = lrintf(( time[lc_idx] - ts ) / duration * tm_size);
            // just in case we're out of bounds:
            if ((model_idx < 0) || (model_idx >= tm_size)) continue;
            float modval = sm[model_idx];

//             // get model flux using linear interpolation
//             // compute true model index including fraction (i.e. x)
//             float mod_idx = (time[lc_idx] - ts) / duration * tm_size;
//             // just in case we're out of bounds:
//             if ((mod_idx < 0) || (mod_idx >= tm_size)) continue;
//             // cast to integer (i.e. round down to get x0)
//             int mod_idx0 = __float2int_rd(mod_idx);
//             // values of model points either side of this point (i.e. y0, y1)
//             float modval0 = sm[max(0, mod_idx0)];
//             float modval1 = sm[min(tm_size, mod_idx0+1)];
//             // perform the linear interpolation to obtain the model flux
//             float modval = modval0 + (mod_idx - mod_idx0) * (modval1 - modval0);

            // transit depth implied by this light curve point
            float local_depth = flux[lc_idx] / modval;
            // weight of this light curve point
            float local_weight = modval * modval * w;

            // accumulate the depth and weight
            sm[sm1_ptr] += local_depth * local_weight;
            sm[sm2_ptr] += local_weight;
        }

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // sum reduction of the second and third arrays in shared memory
        warpSumReductionf(sm, threadIdx.x, tm_size);
        warpSumReductionf(sm, threadIdx.x, tm_size + blockDim.x);

        // calculate the maximum likelihood transit depth
        float wav_depth = sm[tm_size] / sm[tm_size + blockDim.x];
        float var_depth = 1.0f / sm[tm_size + blockDim.x];
        // send the depth to the output arrays
        depth[arr2d_ptr] = wav_depth;
        vdepth[arr2d_ptr] = var_depth;

        // nothing more to do if there were no valid data in the window
        if (isnan(wav_depth)){
            like_ratio[arr2d_ptr] = nanf(0);
            continue;
        };

        // zero out the non transit-model portion of the shared memory again
        sm[sm1_ptr] = 0.0f;
        sm[sm2_ptr] = 0.0f;

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // now loop through the transit window again and calculate the log-likelihood
        for (int i = 0; i <= itr_size; i += blockDim.x){
            int lc_idx = itr_frst_idx + i + threadIdx.x;
            if (lc_idx > lc_size) break;

            // skip if the light curve point has infinite error (i.e. zero weight)
            float w = wght[lc_idx];
            if (w == 0.0f) continue;

            // find the nearest model point index
            int model_idx = lrintf(( time[lc_idx] - ts ) / duration * tm_size);
            // just in case we're out of bounds:
            if ((model_idx < 0) || (model_idx >= tm_size)) continue;
            float modval = sm[model_idx];

//             // get model flux using linear interpolation
//             // compute true model index including fraction (i.e. x)
//             float mod_idx = (time[lc_idx] - ts) / duration * tm_size;
//             // just in case we're out of bounds:
//             if ((mod_idx < 0) || (mod_idx >= tm_size)) continue;
//             // cast to integer (i.e. round down to get x0)
//             int mod_idx0 = __float2int_rd(mod_idx);
//             // values of model points either side of this point (i.e. y0, y1)
//             float modval0 = sm[max(0, mod_idx0)];
//             float modval1 = sm[min(tm_size, mod_idx0+1)];
//             // perform the linear interpolation to obtain the model flux
//             float modval = modval0 + (mod_idx - mod_idx0) * (modval1 - modval0);

            // log-likelihood of in-transit points
            float resid = modval * wav_depth - flux[lc_idx];
            sm[sm1_ptr] += (-0.5f * resid * resid * w);

            // the second part of the log-likelihood is:
            //     -0.5 * log(2 * pi * error^2)
            // but it's unnecessary to add it only to subtract it again,
            // might as well avoid that expensive log operation!

            // subtract constant flux log-likelihood of in-transit points
            // to get the likelihood ratio
            sm[sm1_ptr] -= (-0.5f * flux[lc_idx] * flux[lc_idx] * w);
        }

        // syncthreads needed in cases where block size > 32
        __syncthreads();

        // sum reduction of the second array in shared memory
        warpSumReductionf(sm, threadIdx.x, tm_size);

        // store the likelihood ratio in the output array
        like_ratio[arr2d_ptr] = sm[tm_size];
    }
}

// light curve resampling - stage 1
__global__ void resample_k1(
    const double * time,  // input light curve observation time array
    const double cadence,  // desired output cadence
    const double * flux,  // input light curve offset flux array
    const double * ferr,  // input light curve offset flux error array
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
    double * rflux,  // array of weighted average flux relative to baseline
    double * eflux,  // array of error on weighted average flux relative to baseline
    const int n_elem
){
    const int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_idx >= n_elem) return;

    // note! in the call from python, eflux and sum_w point to the same array...

    // compute inverse variance weighted relative flux
    rflux[x_idx] = 1.0 - sum_fw[x_idx] / sum_w[x_idx];

    // compute error on above
    eflux[x_idx] = rsqrt(sum_w[x_idx]);
}

// period search - kernel 1 (joint likelihood-ratio compute and first stage reduction)
__global__ void periodic_search_k1(
    const float period_strides,  // period in strides
    const float * in_like_ratio,  // the previously computed likelihood ratios
    const float * in_depth,  // the previously computed max-likelihood depths
    const float * in_var_depth,  // the previously computed max-likelihood depth variance
    const int long_ts_count,  // start time stride count across whole light curve
    const int duration_idx_first,  // the index of the first duration to check
    const int duration_idx_last,  // the index of the last duration to check
    const int max_transit_count,  // the maximum possible number of transits for this period
    float * lrat_out,  // the temporary max likelihood ratio array (to be filled)
    float * depth_out,  // the temporary depth array (to be filled)
    float * vdepth_out,  // the temporary depth variance array (to be filled)
    int * d_idx_out,  // the temporary duration index array (to be filled)
    int * ts_idx_out  // the temporary start time index array (to be filled)
){
    // variable declarations
    bool null = false;

    // open the array in shared memory
    extern __shared__ float sm[];
    // pointers for the array in shared memory - split it in half
    int sm_ptr_lr = threadIdx.x;               // likelihood ratio
    int sm_ptr_id = threadIdx.x + blockDim.x;  // thread index

    // nullify the arrays in shared memory
    sm[sm_ptr_lr] = nanf(0);
    sm[sm_ptr_id] = nanf(0);

    // start time and duration indices
    const int ts_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dur_idx = duration_idx_first + blockIdx.y * blockDim.y + threadIdx.y;
    // nullify this thread if we're out of bounds
    // might as well still allow all the processing to run because warp divergence
    // we can't return yet as we need the threads to participate in the reduction
    // operation later
    if ((ts_idx >= period_strides) || (dur_idx > duration_idx_last)){
        null = true;
    };

    // output array pointer
    const int out2d_ptr = blockIdx.x + gridDim.x * blockIdx.y;

    // accumulators
    float _sum_dw = 0.0;
    float _sum_w = 0.0;
    int n_transits = 0;
    // loop through the input arrays to determine the maximum likelihood depth
    for (int i = 0; i < max_transit_count; i++){
        // compute the start time index of this iteration
        int _ts_idx = ts_idx + lrintf(period_strides * i);
        if (_ts_idx >= long_ts_count) break;  // exit the loop if out of bounds

        // pointer into 2d input arrays
        int in2d_ptr = _ts_idx + long_ts_count * dur_idx;

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

    // nullify this thread if there were no transits
    if (n_transits == 0){
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

        // compute the start time index of this iteration
        int _ts_idx = ts_idx + lrintf(period_strides * i);
        if (_ts_idx >= long_ts_count) break;  // exit the loop if out of bounds

        // pointer into 2d input arrays
        int in2d_ptr = _ts_idx + long_ts_count * dur_idx;

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
        sm[sm_ptr_lr] = _sum_lrats_sgl - 0.5 * _sum_logs;
        sm[sm_ptr_id] = 1.0f * threadIdx.x;
    }
    // if you were to add this to the constant model log-likelihood you
    // would obtain the joint-likelihood

    // now do the block level max reduction operation, also recording the pointers
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (threadIdx.x < s) {
            if ((isnan(sm[sm_ptr_lr])) \
                || ((sm[sm_ptr_lr + s] > sm[sm_ptr_lr]) \
                    && (!isnan(sm[sm_ptr_lr+s])))) {
                sm[sm_ptr_lr] = sm[sm_ptr_lr+s];
                sm[sm_ptr_id] = sm[sm_ptr_id+s];
            }
        }
        __syncthreads();
    }

    // only the max-likelihood ratio thread records its results
    const int best_tid = lrintf(sm[blockDim.x]);
    if (threadIdx.x == best_tid) {
        lrat_out[out2d_ptr] = sm[sm_ptr_lr];
        depth_out[out2d_ptr] = wav_depth;
        vdepth_out[out2d_ptr] = var_depth;
        d_idx_out[out2d_ptr] = dur_idx;
        ts_idx_out[out2d_ptr] = ts_idx;
    }
}

// periodic search - kernel 2 (second-stage reduction operation)
__global__ void periodic_search_k2(
    const float * lrat_in,  // max likelihood ratio array
    const float * depth_in,  // depth array
    const float * vdepth_in,  // depth variance array
    const int * d_idx_in,  // duration index array
    const int * ts_idx_in,  // start time index array
    float * lrat_out,  // max likelihood ratio array (single element - to be filled)
    float * depth_out,  // depth array (single element - to be filled)
    float * vdepth_out,  // depth variance array (single element - to be filled)
    int * d_idx_out,  // duration index array (single element - to be filled)
    int * ts_idx_out,  // start time index array (single element - to be filled)
    const int in_arr_len  // length of input arrays
){
    // open the array in shared memory
    extern __shared__ float sm[];
    // pointers for the 2 arrays in shared memory
    int sm_ptr_lr = threadIdx.x;             // likelihood ratio
    int sm_ptr_id = threadIdx.x + blockDim.x;  // thread index

    // nullify the arrays in shared memory
    sm[sm_ptr_lr] = nanf(0);
    sm[sm_ptr_id] = nanf(0);

    // cycle through the likelihood ratio input array
    // record the max and index in shared mem
    for (int i = 0 ; i <= in_arr_len ; i += blockDim.x){
        int idx = i + threadIdx.x;
        if (idx >= in_arr_len) break;
        if ((isnan(sm[sm_ptr_lr])) || ((lrat_in[idx] > sm[sm_ptr_lr]) && (!isnan(lrat_in[idx])))){
            sm[sm_ptr_lr] = lrat_in[idx];
            sm[sm_ptr_id] = 1.0f * idx;
        }
    }
    __syncthreads();

    // final reduction through shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (threadIdx.x < s) {
            if ((isnan(sm[sm_ptr_lr])) || ((sm[sm_ptr_lr + s] > sm[sm_ptr_lr]) && (!isnan(sm[sm_ptr_lr+s])))) {
                sm[sm_ptr_lr] = sm[sm_ptr_lr+s];
                sm[sm_ptr_id] = sm[sm_ptr_id+s];
            }
        }
        __syncthreads();
    }

    // record the maximum likelihood parameters
    const int best_idx = lrintf(sm[blockDim.x]);
    lrat_out[0] = lrat_in[best_idx];
    depth_out[0] = depth_in[best_idx];
    vdepth_out[0] = vdepth_in[best_idx];
    d_idx_out[0] = d_idx_in[best_idx];
    ts_idx_out[0] = ts_idx_in[best_idx];

}
