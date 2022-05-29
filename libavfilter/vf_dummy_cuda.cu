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




// __device__ int julia( int x, int y ) {
//  const float scale = 1.5;
//  float jx = scale * (float)(DIM/2 - x)/(DIM/2);
//  float jy = scale * (float)(DIM/2 - y)/(DIM/2);
//  cuComplex c(-0.8, 0.156);
//  cuComplex a(jx, jy);
//  int i = 0;
//  for (i=0; i<200; i++) {
//  a = a * a + c;
//  if (a.magnitude2() > 1000)
//  return 0;
//  }
//  return 1; 
// }

__device__ double minnn(double a, double b) {
  return a < b ? a : b;
}
__device__ double maxxx(double a, double b) {
  return a > b ? a : b;
}
__device__ double root(double n){
  // Max and minnn are used to take into account numbers less than 1
  double lo = minnn(1.0, n), hi = maxxx(1.0, n), mid;

  // Update the bounds to be off the target by a factor of 10
  while(100 * lo * lo < n) lo *= 10;
  while(0.01 * hi * hi > n) hi *= 0.1;

  for(int i = 0 ; i < 100 ; i++){
      mid = (lo+hi)/2;
      if(mid*mid == n) return mid;
      if(mid*mid > n) hi = mid;
      else lo = mid;
  }
  return mid;
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
   

    int window_size = 3; //size of window that we take its sum
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x coordinate of current pixel
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y coordinate of current pixel
   
    if (y >= height || x >= width)
        return;
    if (y >= height_uv || x >= width_uv)
        return;


    int y_index = y * pitch + x; // index of current pixel in sourceY , access the 1d array as a 2d one

    int start_r = x - window_size / 2;
    int start_c = y - window_size / 2;
    int temp = 0; 

   
    //green color
    float u_chroma=48.0/255.0;
    float v_chroma=45.0/255.0;

    //yuv
    //             y        u           v
    // kak_yuv= 133.8380, -14.7101, -41.9539
    int counter=0;
    double diff=0.0;
    float du,dv;
    for (int i = 0; i < window_size; i++)
    {
        for (int j = 0; j < window_size; j++)
        {
            int r = start_r + i;
            int c = start_c + j;
            bool flag = r >= 0 && r < width && c >= 0 && c < height;
            if (flag)
            {
                du=tex2D<float>(src_tex_U, r, c) - u_chroma;
                dv=tex2D<float>(src_tex_V, r, c) - v_chroma;
                diff += root((du * du + dv * dv) / (255.0 * 255.0 * 2) );
                counter++;
            }
        }
    }
    diff/=float(counter);
    
    
    int u_index, v_index;
    v_index = u_index = y * pitch_uv + x;
    float similarity = 0.1;

   
    if(diff > similarity)
    {
        //blue
    dst_Y[y_index] = 43; // put the result of convolution of the pixel in output imag
    //make the UV channels blue 
    dst_U[u_index] = 245;
    dst_V[v_index] = 128;
    
    }
    else{
        //red
        dst_Y[y_index] = 78;//tex2D<float>(src_tex_Y, x, y)*255;
        dst_U[u_index] = 55;//tex2D<float>(src_tex_U, x, y)*255;
        dst_V[v_index] = 254;//tex2D<float>(src_tex_V, x, y)*255;
    }
    
    //dst_Y[y_index] = temp; // put the result of convolution of the pixel in output image


    
}






//function to prtotoype chroma keing

__global__ void Process_uchar2(cudaTextureObject_t src_tex_Y, cudaTextureObject_t src_tex_UV, cudaTextureObject_t unused1,
                               uchar *dst_Y, uchar2 *dst_UV, uchar *unused2,
                               int width, int height, int pitch,
                               int width_uv, int height_uv, int pitch_uv)
{

  
    int window_size = 3; //size of convolution kernel (kernel dimesnion is size * size)
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x coordinate of current pixel
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y coordinate of current pixel
    
    if (y >= height || x >= width)
        return;
    if (y >= height_uv || x >= width_uv)
        return;


    int y_index = y * pitch + x; // index of current pixel in sourceY , access the 1d array as a 2d one

    int start_r = x - window_size / 2;
    int start_c = y - window_size / 2;
    int temp = 0; 

    
    //green color
    float u_chroma=48.0;
    float v_chroma=45.0;


  
    int counter=0;
    double diff=0.0;
    float du,dv;

    //this loop covers the eight neghbour of the pixel
    for (int i = 0; i < window_size; i++)
    {
        for (int j = 0; j < window_size; j++)
        {
            int r = start_r + i;
            int c = start_c + j;
            bool flag = r >= 0 && r < width && c >= 0 && c < height;
            if (flag)
            {
                uchar2 uv=tex2D<uchar2>(src_tex_UV, r, c)*255;

                du=uv.x - u_chroma;
                dv=uv.y - v_chroma;
                diff += root(  (du * du + dv * dv) / (255.0 * 255.0 * 2))  ;
                counter++;
            }
        }
    }
    diff/=float(counter);
    
    
    int u_index, v_index;
    v_index = u_index = y * pitch_uv + x;

    
    float similarity = 0.22;

   
    if(diff < similarity)
    {
    
        //black
        dst_Y[y_index] = 0; 
        //make the UV channels black 
        dst_UV[u_index] = make_uchar2(128,128);
    
    }
    else
    {
        //white
    dst_Y[y_index] = 255; // put the result of convolution of the pixel in output imag
    //make the UV channels white 
    dst_UV[u_index] = make_uchar2(128,128);

   
    

    }

}

}
