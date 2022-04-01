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

extern "C" {





/*
function description : function convolves the edge detector laplacian operator with the image

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
                              uchar *dst_Y, uchar *dst_U, uchar *dst_V,
                              int width, int height, int pitch,
                              int width_uv, int height_uv, int pitch_uv)
{
    float conv_kernel[] = {0, 1, 0,  
                    1, -4, 1,
                    0, 1, 0};

    int kernel_size = 3; //size of convolution kernel (kernel dimesnion is size * size)
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x coordinate of current pixel
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y coordinate of current pixel
    if (y >= height || x >= width)                 // check if out of image
        return;
    int y_index = y * pitch + x; // index of current pixel in sourceY , access the 1d array as a 2d one

    int start_r = x - kernel_size / 2;
    int start_c = y - kernel_size / 2;
    int temp = 0; 
    //loop for applying convolution kernel to each pixel
    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            int r = start_r + i;
            int c = start_c + j;
            bool flag = r >= 0 && r < width && c >= 0 && c < height;
            if (flag)
            {
                // multiply by 255 as the input kernel and tex  is in [0,1]
                temp += conv_kernel[i * kernel_size + j] * tex2D<float>(src_tex_Y, r, c) * 255;
            }
        }
    }
    dst_Y[y_index] = temp; // put the result of convolution of the pixel in output image


    if (y >= height_uv || x >= width_uv)
        return;

    int u_index, v_index;
    v_index = u_index = y * pitch_uv + x;

    //make the UV channels black 
    dst_U[u_index] = 128;
    dst_V[v_index] = 128;
    
}






/*

function convolves the edge detector laplacian operator with the image

input: 
    src_tex_Y: Y channel of source image normallized to [0,1]
    src_tex_UV : UV channel of source image normallized to [0,1] , it is like a tuple that has U and V channels
    dst_Y: Y channel of output convolved image [0,255]
    dst_UV : UV channel output , it is always zero as Y channel only matters in edge detection
    unused2: unused parameter
    width: width of sourceY image
    height: height of sourceY image
    pitch: pitch of sourceY image , the linesize 
    width_uv: width of sourceU,V image
    height_uv: height of sourceU,V image
    pitch_uv: pitch of sourceU,V image , the linesize

*/

__global__ void Process_uchar2(cudaTextureObject_t src_tex_Y, cudaTextureObject_t src_tex_UV, cudaTextureObject_t unused1,
                               uchar *dst_Y, uchar2 *dst_UV, uchar *unused2,
                               int width, int height, int pitch,
                               int width_uv, int height_uv, int pitch_uv)
{

    float conv_kernel[] = {0, 1, 0,  
                    1, -4, 1,
                    0, 1, 0};

    int kernel_size = 3; //size of convolution kernel (kernel dimesnion is size * size)
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x coordinate of current pixel
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y coordinate of current pixel
    if (y >= height || x >= width)                 // check if out of image
        return;
    int y_index = y * pitch + x; // index of current pixel in sourceY , access the 1d array as a 2d one

    int start_r = x - kernel_size / 2;
    int start_c = y - kernel_size / 2;
    int temp = 0; 
    //loop for applying convolution kernel to each pixel
    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            int r = start_r + i;
            int c = start_c + j;
            bool flag = r >= 0 && r < width && c >= 0 && c < height;
            if (flag)
            {
                // multiply by 255 as the input kernel and tex  is in [0,1]
                temp += conv_kernel[i * kernel_size + j] * tex2D<float>(src_tex_Y, r, c) * 255;
            }
        }
    }
    dst_Y[y_index] = temp; // put the result of convolution of the pixel in output image


    if (y >= height_uv || x >= width_uv)
        return;

    //make the UV channels black
    dst_UV[y*pitch_uv + x] = make_uchar2(128, 128);

}

 }
