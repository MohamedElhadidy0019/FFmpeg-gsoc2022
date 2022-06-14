/*
 * Copyright (c) 2022, Dummy
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "cuda/vector_helpers.cuh"

extern "C"
{

    __device__ float minnn(float a, float b)
    {
        return a < b ? a : b;
    }
    __device__ float maxxx(float a, float b)
    {
        return a > b ? a : b;
    }
    __device__ float root(float n)
    {
        // Max and minnn are used to take into account numbers less than 1
        float lo = minnn(1.0, n), hi = maxxx(1.0, n), mid;

        // Update the bounds to be off the target by a factor of 10
        while (100 * lo * lo < n)
            lo *= 10;
        while (0.01 * hi * hi > n)
            hi *= 0.1;

        for (int i = 0; i < 100; i++)
        {
            mid = (lo + hi) / 2;
            if (mid * mid == n)
                return mid;
            if (mid * mid > n)
                hi = mid;
            else
                lo = mid;
        }
        return mid;
    }

    __device__ static inline bool get_alpha_value(cudaTextureObject_t src_tex,
                            cudaTextureObject_t src_tex_V,
                            int width_uv, int height_uv,
                            int x, int y, float2 chromakey_uv,
                            float similarity, bool is_uchar2)
    {

        int window_size = 3;
        int start_r = x - window_size / 2;
        int start_c = y - window_size / 2;

        int counter = 0;
        float diff = 0.0f;
        float du, dv;

        for (int i = 0; i < window_size; i++)
        {
            for (int j = 0; j < window_size; j++)
            {
                int r = start_r + i;
                int c = start_c + j;
                bool flag = (r >= 0 && r < width_uv && c >= 0 && c < height_uv);

                if (!flag)
                    continue;

                float u_value, v_value;
                if (is_uchar2)
                {
                    float2 temp_uv = tex2D<float2>(src_tex, r, c);
                    u_value = temp_uv.x;
                    v_value = temp_uv.y;
                }
                else
                {
                    u_value = tex2D<float>(src_tex, r, c);
                    v_value = tex2D<float>(src_tex_V, r, c);
                }

                du = (u_value * 255.0f) - chromakey_uv.x;
                dv = (v_value * 255.0f) - chromakey_uv.y;
                diff += root((du * du + dv * dv) / (255.0f * 255.0f * 2.f));
                counter++;
            }
        }

        if (counter > 0)
        {
            diff = diff / counter;
        }
        else
        {
            diff /= 9.0f;
        }

        return diff < similarity ? 1 : 0;
    }

    /*
    function description : function that genrates the alpha channel for a chroma keyed picture

    input :
        src_tex_Y: Y channel of source image normallized to [0,1]
        src_tex_U: U channel of source image normallized to [0,1]
        src_tex_V: V channel of source image normallized to [0,1]
        dst_Y: Y channel of output convolved image [0,255]
        dst_U: U channel of output convolved image [0,255] ,always zero as the Y channel only matters in edge detection
        dst_V: V channel of output convolved image [0,255] ,always zero as the Y channel only matters in edge detection
        width: width of sourceY image
        height: height of sourceY image
        pitch: pitch of sourceY image
        width_uv: width of sourceU,V image
        height_uv: height of sourceU,V image
        pitch_uv: pitch of sourceU,V image
    */

    __global__ void Process_uchar(cudaTextureObject_t src_tex_Y, cudaTextureObject_t src_tex_U, cudaTextureObject_t src_tex_V,
                                  uchar *dst_Y, uchar *dst_U, uchar *dst_V,uchar *dst_A,
                                  int width, int height, int pitch,
                                  int width_uv, int height_uv, int pitch_uv)
    {

        int window_size = 3;                           // size of window
        int x = blockIdx.x * blockDim.x + threadIdx.x; // x coordinate of current pixel
        int y = blockIdx.y * blockDim.y + threadIdx.y; // y coordinate of current pixel

        if (y >= height || x >= width)
            return;
        dst_Y[y*pitch + x] = tex2D<float>(src_tex_Y, x, y) * 255;

        if (y >= height_uv || x >= width_uv)
            return;
        dst_U[y*pitch_uv + x] = tex2D<float>(src_tex_U, x, y) * 255;
        dst_V[y*pitch_uv + x] = tex2D<float>(src_tex_V, x, y) * 255;

        int y_index = y * pitch + x; // index of current pixel in sourceY , access the 1d array as a 2d one

    
        // green color
        float u_chroma = 48.0f;
        float v_chroma = 45.0f;
        float similarity = 0.20f;
        bool alpha_value=get_alpha_value(src_tex_U,src_tex_V,width_uv,height_uv,x,y,make_float2(u_chroma,v_chroma),similarity,true);


        int u_index, v_index;
        v_index = u_index = y * pitch_uv + x;
        int new_size = 2;

        if (!alpha_value) // it is chroma
        {
            dst_A[u_index] = 255;
        }
        else
        { 
            dst_A[u_index] = 0;
        }
    }

    // function to prtotoype chroma keing

    __global__ void Process_uchar2(cudaTextureObject_t src_tex_Y, cudaTextureObject_t src_tex_UV, cudaTextureObject_t unused1,
                                   uchar *dst_Y, uchar *dst_U, uchar *dst_V,uchar *dst_A,
                                   int width, int height, int pitch,
                                   int width_uv, int height_uv, int pitch_uv)
    {

        int window_size = 3;                           // size of window
        int x = blockIdx.x * blockDim.x + threadIdx.x; // x coordinate of current pixel
        int y = blockIdx.y * blockDim.y + threadIdx.y; // y coordinate of current pixel

        if (y >= height || x >= width)
            return;
        dst_Y[y*pitch + x] = tex2D<float>(src_tex_Y, x, y) * 255;

        if (y >= height_uv || x >= width_uv)
            return;

        int y_index = y * pitch + x; // index of current pixel in sourceY , access the 1d array as a 2d one


        // green color
        float u_chroma = 48.0f;
        float v_chroma = 45.0f;
        float similarity = 0.22f;

        bool alpha_value=!get_alpha_value(src_tex_UV,unused1,width_uv,height_uv,x,y,make_float2(u_chroma,v_chroma),similarity,true);
        
        


        int u_index, uv_index;
        uv_index = u_index = y * pitch_uv + x;

        dst_U[y*pitch_uv + x] = tex2D<float2>(src_tex_UV, x, y).x * 255;
        dst_V[y*pitch_uv + x] = tex2D<float2>(src_tex_UV, x, y).y * 255;
        if (!alpha_value) // it is chroma
        {
            dst_A[uv_index] = 255;
        }
        else // it is not chroma
        {
            dst_A[uv_index] = 0;
        }
    }
}
