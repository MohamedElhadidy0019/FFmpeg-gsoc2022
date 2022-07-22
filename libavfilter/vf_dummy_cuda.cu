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


//make a function

// __device__ static inline float distance(int x, int y, int i, int j) {
//     return float(sqrtf(powf(x - i, 2) + powf(y - j, 2)));
// }

// __device__ static inline float guassian(float x, float sigma)
// {
//     float val=x;
//     const float PI = 3.14159;
//     float ans;
//     //ans= (1/(2*PI*powf(sigma,2)))   * expf((-1/2) * (  powf(val,2) / powf(sigma,2) ) );
//     //return ans;
//     float first_term= (1/(sigma * sqrtf(2*PI)) );
//     float second_term= expf( -powf(x,2) / (2* powf(sigma,2)));
//     return first_term * second_term;
//     //return expf(-(powf(x, 2))/(2 * powf(sigma, 2))) / (2 * PI * powf(sigma, 2));
// }

__device__ static inline float calculate_w(int x, int y, int r,int c,
                                    float pixel_value, float neighbor_value,
                                    float sigma_space, float sigma_color)
{
    float first_term,second_term,w;
    first_term=  (powf(x-r,2) + powf(y-c,2))  / (2*sigma_space*sigma_space);
    second_term= powf(pixel_value - neighbor_value,2) / (2*sigma_color*sigma_color);
    w= expf(-first_term - second_term);
    return w;
}



__device__ static inline float apply_bilateral(cudaTextureObject_t tex,int width,int height,int x, int y, float sigma_space, float sigma_color, int window_size)
{
    float current_pixel=tex2D<float>(tex, x, y)*255;
    int start_r = x - window_size / 2;
    int start_c = y - window_size / 2;
    float neighbor_pixel=0;
    float Wp=0.f;
    float new_pixel_value=0.f;
    float w=0.f;
    for(int i=0;i<window_size;i++)
    {
        for(int j=0;j<window_size;j++)
        {
            int r=start_r+i;
            int c=start_c+j;
            bool in_bounds=r>=0 && r<width && c>=0 && c<height;
            if(in_bounds)
            {
                neighbor_pixel=tex2D<float>(tex, r, c)*255;
                float dist=distance(x,y,r,c);
                //float guassian_space=guassian(dist,sigma_space);
                //float guassian_color=guassian(abs(current_pixel-neighbor_pixel),sigma_color);
                //w=guassian_space*guassian_color;
                w=calculate_w(x,y,r,c,current_pixel,neighbor_pixel,sigma_space,sigma_color);
                Wp+=w;
                new_pixel_value+=w*neighbor_pixel;
            }
        }
    }

    return new_pixel_value/Wp;
}



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

    float sigmaSpace=10.0f;
    float sigmaColor=100.0f;

    int window_size = 9;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || x >= width)
        return;
    int y_index = y * pitch + x; // index of current pixel in sourceY , access the 1d array as a 2d one

    float new_Y=apply_bilateral(src_tex_Y,width,height,x,y,sigmaSpace,sigmaColor,window_size);
    dst_Y[y_index] = (uchar)llrintf(new_Y);



    if (y >= height_uv || x >= width_uv)
        return;

    int u_index, v_index;
    v_index = u_index = y * pitch_uv + x;
    float new_U=apply_bilateral(src_tex_U,width_uv,height_uv,x,y,sigmaSpace,sigmaColor,window_size);
    float new_V=apply_bilateral(src_tex_V,width_uv,height_uv,x,y,sigmaSpace,sigmaColor,window_size);


    dst_U[u_index] = (uchar)llrintf(new_U);
    dst_V[v_index] = (uchar)llrintf(new_V);

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
