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

    __device__ static inline void change_alpha_channel(cudaTextureObject_t &src_tex,
                            cudaTextureObject_t &src_tex_V,uchar *dst_A,
                            int &width_uv, int &height_uv,int &width,int &height,int &pitch,
                            int &x, int &y, float2 chromakey_uv,
                            float &similarity,float &blend, bool is_uchar2,uchar &resize_ratio)
    {

        uchar window_size = 3;
        int start_r = x - window_size / 2;
        int start_c = y - window_size / 2;

        uchar counter = 0;
        float diff = 0.0f;
        float du, dv;

        for (uchar i = 0; i < window_size; i++)
        {
            for (uchar j = 0; j < window_size; j++)
            {
                int r = start_r + i;
                int c = start_c + j;
                bool check_flag = (r >= 0 && r < width_uv && c >= 0 && c < height_uv);

                if (!check_flag)
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
                diff += sqrtf((du * du + dv * dv) / (255.0f * 255.0f * 2.f));
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

        uchar  alpha_value;
        if(blend>0.0001f){
            alpha_value=__saturatef((diff - similarity) / blend)*255;
        }else{
            alpha_value=(diff < similarity ? 0 : 1)*255;
        }
        
       
        for (uchar k = 0; k < resize_ratio; k++)
        {
            for (uchar l = 0; l < resize_ratio; l++)
            {
                int x_resize = x * resize_ratio + k;
                int y_resize = y * resize_ratio + l;
                int a_channel_resize = y_resize * pitch + x_resize;
                if (y_resize >= height || x_resize >= width)
                    continue;
                dst_A[a_channel_resize] = alpha_value;
            }
        }


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
                                  int width_uv, int height_uv, int pitch_uv,
                            float u_key,float v_key, float similarity,
                            float blend)
    {

        uchar resize_ratio = 2;                           // size of window
        int x = blockIdx.x * blockDim.x + threadIdx.x; // x coordinate of current pixel
        int y = blockIdx.y * blockDim.y + threadIdx.y; // y coordinate of current pixel

        if (y >= height || x >= width)
            return;
        dst_Y[y * pitch + x] = tex2D<float>(src_tex_Y, x, y)*255;
        if (y >= height_uv || x >= width_uv)
            return;
        
        int uv_index = y * pitch_uv + x;
        dst_U[uv_index]=tex2D<float>(src_tex_U,x,y)*255;
        dst_V[uv_index]=tex2D<float>(src_tex_V,x,y)*255;



        change_alpha_channel(src_tex_U,src_tex_V,
                            dst_A,width_uv,height_uv,
                            width,height,
                            pitch,x,y,make_float2(u_key,v_key),
                            similarity,blend,false,resize_ratio);

    }

    // function to prtotoype chroma keing

    __global__ void Process_uchar2(cudaTextureObject_t src_tex_Y, cudaTextureObject_t src_tex_UV, cudaTextureObject_t unused1,
                                   uchar *dst_Y, uchar *dst_U, uchar *dst_V,uchar *dst_A,
                                   int width, int height, int pitch,
                                   int width_uv, int height_uv, int pitch_uv,
                            float u_key,float v_key, float similarity,
                            float blend)
    {

        uchar resize_ratio = 2;                           // size of window
        int x = blockIdx.x * blockDim.x + threadIdx.x; // x coordinate of current pixel
        int y = blockIdx.y * blockDim.y + threadIdx.y; // y coordinate of current pixel

        if (y >= height || x >= width)
            return;
        dst_Y[y * pitch + x] = tex2D<float>(src_tex_Y, x, y)*255;

        if (y >= height_uv || x >= width_uv)
            return;
        int uv_index= y * pitch_uv + x;
        float2 uv_temp=tex2D<float2>(src_tex_UV,x,y);
        dst_U[uv_index]=uv_temp.x*255;
        dst_V[uv_index]=uv_temp.y*255;

        change_alpha_channel(src_tex_UV,unused1,
                                dst_A,width_uv,height_uv,
                                width,height,pitch,
                                x,y,make_float2(u_key,v_key),
                                similarity,blend,
                                true,resize_ratio);
        
        
    }
}
