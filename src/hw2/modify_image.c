#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

/******************************** Resizing *****************************
  To resize we'll need some interpolation methods and a function to create
  a new image and fill it in with our interpolation methods.
************************************************************************/

float nn_interpolate(image im, float x, float y, int c)
{
    // TODO
    /***********************************************************************
      This function performs nearest-neighbor interpolation on image "im"
      given a floating column value "x", row value "y" and integer channel "c",
      and returns the interpolated value.
    ************************************************************************/
    int nn_x = round(x);
    int nn_y = round(y);
    float f = get_pixel(im, nn_x, nn_y, c);
    return f;
}

image nn_resize(image im, int w, int h)
{
    // TODO Fill in (also fix the return line)
    /***********************************************************************
      This function uses nearest-neighbor interpolation on image "im" to a new
      image of size "w x h"
    ************************************************************************/
    image resized_image = make_image(w, h, im.c);
    
    // Coordinate transfer resized_x = Ax * x + B_x; resized_y = A_y * y + B_y
    float A_x, B_x, A_y, B_y;
    A_x = (float)im.w / (float)w;
    B_x = -0.5 + A_x * 0.5;
    A_y = (float)im.h / (float)h;
    B_y = -0.5 + A_y * 0.5;

    // Interpolate
    float interpolated_f;
    for(int k = 0; k < im.c; k++){
        for(int i = 0; i < h; i++){
            for(int j = 0; j < w; j++){
                interpolated_f = nn_interpolate(im, A_x * j + B_x, A_y * i + B_y, k);
                set_pixel(resized_image, j, i, k, interpolated_f);
            }
        }
    }

    return resized_image;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    // TODO
    /***********************************************************************
      This function performs bilinear interpolation on image "im" given
      a floating column value "x", row value "y" and integer channel "c".
      It interpolates and returns the interpolated value.
    ************************************************************************/
    float fl_x, cl_x, fl_y, cl_y;
    fl_x = floor(x);
    fl_y = floor(y);
    cl_x = floor(x) + 1;
    cl_y = floor(y) + 1;

    // Get pixels
    float p11, p12, p21, p22;
    p11 = get_pixel(im, fl_x, fl_y, c);
    p12 = get_pixel(im, fl_x, cl_y, c);
    p21 = get_pixel(im, cl_x, fl_y, c);
    p22 = get_pixel(im, cl_x, cl_y, c);

    // Interpolate along x axis
    float fl_in_x, cl_in_x;
    fl_in_x = p11 * (cl_x - x) + p21 * (x - fl_x);
    cl_in_x = p12 * (cl_x - x) + p22 * (x - fl_x);
    /*
    if (cl_x == x && fl_x == x){
        fl_in_x = p11;
        cl_in_x = p12;
    }else{
        fl_in_x = p11 * (cl_x - x) + p21 * (x - fl_x);
        cl_in_x = p12 * (cl_x - x) + p22 * (x - fl_x);
    }*/

    // Interpolate along y axis
    float f;
    f = fl_in_x * (cl_y - y) + cl_in_x * (y - fl_y);
    /*
    if (cl_y == y && fl_y == y) f = fl_in_x;
    else f = fl_in_x * (cl_y - y) + cl_in_x * (y - fl_y);*/
    return f;
}

image bilinear_resize(image im, int w, int h)
{
    // TODO
    /***********************************************************************
      This function uses bilinear interpolation on image "im" to a new image
      of size "w x h". Algorithm is same as nearest-neighbor interpolation.
    ************************************************************************/
    image resized_image = make_image(w, h, im.c);

    // Coordinate transfer
    float A_x, B_x, A_y, B_y;
    A_x = (float)im.w / (float)w;
    B_x = -0.5 + A_x * 0.5;
    A_y = (float)im.h / (float)h;
    B_y = -0.5 + A_y * 0.5;

    // Interpolate
    float interpolated_f;
    for(int k = 0; k < im.c; k++){
        for(int i = 0; i < h; i++){
            for(int j = 0; j < w; j++){
                interpolated_f = bilinear_interpolate(im, A_x * j + B_x, A_y * i + B_y, k);
                set_pixel(resized_image, j, i, k, interpolated_f);
            }
        }
    }
    return resized_image;
}


/********************** Filtering: Box filter ***************************
  We want to create a box filter. We will only use square box filters.
************************************************************************/

void l1_normalize(image im)
{
    // TODO
    /***********************************************************************
      This function divides each value in image "im" by the sum of all the
      values in the image and modifies the image in place.
    ************************************************************************/

    // Get sum of all image pixel values
    float sum, f;
    sum = 0;
    for(int k = 0; k < im.c; k++){
        for(int i = 0; i < im.h; i++){
            for(int j = 0; j < im.w; j++){
                sum +=get_pixel(im, j, i, k);
            }
        }
    }

    // Normalize
    for(int k = 0; k < im.c; k++){
        for(int i = 0; i < im.h; i++){
            for(int j = 0; j < im.w; j++){
                f = get_pixel(im, j, i, k);
                f = f/sum;
                set_pixel(im, j, i, k, f);
            }
        }
    }
}

image make_box_filter(int w)
{
    // TODO
    /***********************************************************************
      This function makes a square filter of size "w x w". Make an image of
      width = height = w and number of channels = 1, with all entries equal
      to 1. Then use "l1_normalize" to normalize your filter.
    ************************************************************************/
    // Make a filter
    image im = make_image(w, w, 1);
    for(int k = 0; k < im.c; k++){
        for(int i = 0; i < im.h; i++){
            for(int j = 0; j < im.w; j++){
                set_pixel(im, j, i, k, 1.0);
            }
        }
    }

    // Normalize
    l1_normalize(im);
    return im;
}

image convolve_image(image im, image filter, int preserve)
{
    // TODO
    /***********************************************************************
      This function convolves the image "im" with the "filter". The value
      of preserve is 1 if the number of input image channels need to be 
      preserved. Check the detailed algorithm given in the README.  
    ************************************************************************/
    // Check filter channel
    assert(filter.c == 1 || filter.c == im.c);
    int s_x = (filter.w - 1) / 2;
    int s_y = (filter.h - 1) / 2;

    // Loop over the image to do convolution
    image filtered_im = make_image(im.w, im.h, im.c);
    int filter_k;
    float filtered_p;
    for(int kk = 0; kk < im.c; kk++){
        // Filter has only 1 channel
        if(filter.c == 1) filter_k = 0;
        else filter_k = kk;

        for(int ii = 0; ii <= im.h; ii++){
            for(int jj = 0; jj <= im.w; jj++){
                // Convolution
                filtered_p = 0;
                for (int fi = 0; fi < filter.h; fi++){
                    for (int fj = 0; fj < filter.w; fj ++){
                        filtered_p += get_pixel(im, jj - s_x + fj, ii - s_y + fi, kk) * get_pixel(filter, fj, fi, filter_k);
                    }
                }
                set_pixel(filtered_im, jj, ii, kk, filtered_p);
            }
        }

    }
    // Presearve
    if (preserve == 1) {
       return filtered_im;
    }else{
        image preserve_im = make_image(filtered_im.w, filtered_im.h, 1);
        float sum_c; // float!!!! not int!!!
        for (int ii = 0; ii < filtered_im.h; ii++){
            for (int jj = 0; jj < filtered_im.w; jj++){
                sum_c = 0;
                for (int kk = 0; kk < filtered_im.c; kk++){
                    sum_c += get_pixel(filtered_im, jj, ii, kk); 
                }
                set_pixel(preserve_im, jj, ii, 0, sum_c);
            }
        }
        return preserve_im;
    }
    return filtered_im;
}

image make_highpass_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with highpass filter values using image.data[]
    ************************************************************************/
     // Make a filter
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, 0);
    set_pixel(filter, 1, 0, 0, -1);
    set_pixel(filter, 2, 0, 0, 0);
    set_pixel(filter, 0, 1, 0, -1);
    set_pixel(filter, 1, 1, 0, 4);
    set_pixel(filter, 2, 1, 0, -1);
    set_pixel(filter, 0, 2, 0, 0);
    set_pixel(filter, 1, 2, 0, -1);
    set_pixel(filter, 2, 2, 0, 0);
    return filter;
}

image make_sharpen_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with sharpen filter values using image.data[]
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, 0);
    set_pixel(filter, 1, 0, 0, -1);
    set_pixel(filter, 2, 0, 0, 0);
    set_pixel(filter, 0, 1, 0, -1);
    set_pixel(filter, 1, 1, 0, 5);
    set_pixel(filter, 2, 1, 0, -1);
    set_pixel(filter, 0, 2, 0, 0);
    set_pixel(filter, 1, 2, 0, -1);
    set_pixel(filter, 2, 2, 0, 0);
    return filter;
}

image make_emboss_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with emboss filter values using image.data[]
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, -2);
    set_pixel(filter, 1, 0, 0, -1);
    set_pixel(filter, 2, 0, 0, 0);
    set_pixel(filter, 0, 1, 0, -1);
    set_pixel(filter, 1, 1, 0, 1);
    set_pixel(filter, 2, 1, 0, 1);
    set_pixel(filter, 0, 2, 0, 0);
    set_pixel(filter, 1, 2, 0, 1);
    set_pixel(filter, 2, 2, 0, 2);
    return filter;
}

// Question 2.3.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: We should use preserve when we run box filters, gaussian filters, and emboss filters. Because after running the filter on a RGB image, we still want to get a RGB image. Therefore, we preserve the same number of channels as the input.
// We should not use preserve when we run high pass filters and sobel filters. Because we use these filters to detect high frequency signals in the image like edges, and we only need one channel to merge information from all channels and display where the edges are.

// Question 2.3.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer corrected:  we need to clamp pixel values for all filters for postprocessing.
// Answer: The box filter outputs a blurred image. It's a low pass filter. After box filtering, we can down sample the image with a large ratio and the image can appear smooth.
// The gaussian filter computes a smoothed image. We use gaussian filters to remove white noise. It is the preprocessing tool for common image algorithms.
// The sobel filter computes the first order derivative of the image. The derivatives are in either X or Y direction. We use sobel in canny edge detection. After computing the derivative magnitude, we apply non-max suppression, thresholding, and connecting to futher identify edges.
// The high-pass filter computes the second order derivative of the image. The edge aeras are 0.  
// The sharpen filter enhances the edges of an image by combining edges computed by laplacian filters and the original image.
// The emboss filter combines edges with a certain direction and the original image to display the emboss effect.

image make_gaussian_filter(float sigma)
{
    // TODO
    /***********************************************************************
      sigma: a float number for the Gaussian.
      Create a Gaussian filter with the given sigma. Note that the kernel size 
      is the next highest odd integer from 6 x sigma. Return the Gaussian filter.
    ************************************************************************/
    int size = ceil(6 * sigma);
    if (size % 2 == 0) size++;
    image filter = make_image(size, size, 1);
    // Make a gaussian filter
    float f;
    int s = (size - 1) / 2;
    for (int i = 0; i < filter.h; i++){
        for (int j = 0; j < filter.w; j++){
            f = exp(((j - s) * (j - s) + (i - s) * (i - s)) / pow(sigma, 2) / -2.0) / TWOPI / pow(sigma, 2);  
            set_pixel(filter, j, i, 0, f);
        }
    }
    // Let sum to be 1
    l1_normalize(filter);
    return filter;
}

image add_image(image a, image b)
{
    // TODO
    /***********************************************************************
      The input images a and image b have the same height, width, and channels.
      Sum the given two images and return the result, which should also have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/
     assert(a.w == b.w && a.h == b.h && a.c == b.c);
    image sum = make_image(a.w, a.h, a.c);
    float fa, fb;
    for (int i = 0; i < sum.h; i++){
        for (int j = 0; j < sum.w; j++){
            for (int k = 0; k < sum.c; k++){
                fa = get_pixel(a, j, i, k);
                fb = get_pixel(b, j, i, k);
                set_pixel(sum, j, i, k, fa + fb);
            }
        }
    }
    return sum;
}

image sub_image(image a, image b)
{
    // TODO
    /***********************************************************************
      The input image a and image b have the same height, width, and channels.
      Subtract the given two images and return the result, which should have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/
    assert(a.w == b.w && a.h == b.h && a.c == b.c);
    image sub = make_image(a.w, a.h, a.c);
    float fa, fb;
    for (int i = 0; i < sub.h; i++){
        for (int j = 0; j < sub.w; j++){
            for (int k = 0; k < sub.c; k++){
                fa = get_pixel(a, j, i, k);
                fb = get_pixel(b, j, i, k);
                set_pixel(sub, j, i, k, fa - fb);
            }
        }
    }
    return sub;
}

image make_gx_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 Sobel Gx filter and return it
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, -1);
    set_pixel(filter, 1, 0, 0, 0);
    set_pixel(filter, 2, 0, 0, 1);
    set_pixel(filter, 0, 1, 0, -2);
    set_pixel(filter, 1, 1, 0, 0);
    set_pixel(filter, 2, 1, 0, 2);
    set_pixel(filter, 0, 2, 0, -1);
    set_pixel(filter, 1, 2, 0, 0);
    set_pixel(filter, 2, 2, 0, 1);
    return filter;
}

image make_gy_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 Sobel Gy filter and return it
    ************************************************************************/
   image filter = make_image(3, 3, 1);
    set_pixel(filter, 0, 0, 0, -1);
    set_pixel(filter, 1, 0, 0, -2);
    set_pixel(filter, 2, 0, 0, -1);
    set_pixel(filter, 0, 1, 0, 0);
    set_pixel(filter, 1, 1, 0, 0);
    set_pixel(filter, 2, 1, 0, 0);
    set_pixel(filter, 0, 2, 0, 1);
    set_pixel(filter, 1, 2, 0, 2);
    set_pixel(filter, 2, 2, 0, 1);
    return filter;
}

void feature_normalize(image im)
{
    // TODO
    /***********************************************************************
      Calculate minimum and maximum pixel values. Normalize the image by
      subtracting the minimum and dividing by the max-min difference.
    ************************************************************************/
    float min = get_pixel(im, 0, 0, 0);
    float max = get_pixel(im, 0, 0, 0);
    float f;

    for (int k = 0; k < im.c; k++){
        for (int i = 0; i < im.h; i++){
            for (int j = 0; j < im.w; j++){
                f = get_pixel(im, j, i, k);
                if (f < min) min = f;
                if (f > max) max = f;
            }
        }
    }

    // Rearange
    float range = max - min;
    for (int k = 0; k < im.c; k++){
        for (int i = 0; i < im.h; i++){
            for (int j = 0; j < im.w; j++){
                if(range - 0 < 1e-10){
                    set_pixel(im, j, i, k, 0);
                }else{
                    f = (get_pixel(im, j, i, k) - min) / range;
                    set_pixel(im, j, i, k, f);
                }
            }
        }
    }
}

image *sobel_image(image im)
{
    // TODO
    /***********************************************************************
      Apply Sobel filter to the input image "im", get the magnitude as sobelimg[0]
      and gradient as sobelimg[1], and return the result.
    ************************************************************************/
    // Sobel x and y
    image filter_x = make_gx_filter();
    image im_x = convolve_image(im, filter_x, 0);
    image filter_y = make_gy_filter();
    image im_y = convolve_image(im, filter_y, 0);

    // Magnitude and direction
    image mag = make_image(im.w, im.h, 1);
    image dir = make_image(im.w, im.h, 1);
    float f, d, fx, fy;
    for (int i = 0; i < im.h; i++){
        for (int j = 0; j < im.w; j++){
            fx = get_pixel(im_x, j, i, 0);
            fy = get_pixel(im_y, j, i, 0);
            f = pow(pow(fx, 2) + pow(fy, 2), 0.5);
            d = atan2(fy, fx);
            set_pixel(mag, j, i, 0, f);
            set_pixel(dir, j, i, 0, d);
        }
    }
    image* sob_im = calloc(2, sizeof(image));
    sob_im[0] = mag;
    sob_im[1] = dir;
    return sob_im;
}

image colorize_sobel(image im)
{
  // TODO
  /***********************************************************************
    Create a colorized version of the edges in image "im" using the 
    algorithm described in the README.
  ************************************************************************/
  // Sobel
    image* sob_im = sobel_image(im);
    // Feature normalization
    feature_normalize(sob_im[1]);
    // False color
    image color_sob = make_image(im.w, im.h, im.c);
    float h, s;
    for (int i = 0; i < im.h; i++){
        for (int j = 0; j < im.w; j++){
            h = get_pixel(sob_im[1], j, i, 0);
            s = get_pixel(sob_im[0], j, i, 0);
            set_pixel(color_sob, j, i, 0, h);
            set_pixel(color_sob, j, i, 1, s);
            set_pixel(color_sob, j, i, 2, h);
        }
    }
    // hsv to rgb
    hsv_to_rgb(color_sob);
    return color_sob;
}

// EXTRA CREDIT: Median filter
int cmpfunc (const void * a, const void * b)
{
   return ( *(float*)a - *(float*)b )>0?1:-1;
}

image apply_median_filter(image im, int kernel_size)
{
  float* arr = malloc(sizeof(float)*kernel_size*kernel_size);
  image filtered_im = make_image(im.w, im.h, im.c);
  for (int k = 0; k < im.c; k++){
    for (int j = 0; j < im.h; j++){
      for (int i = 0; i < im.w; i++){

        // Median filter
        for (int kernel_i = 0; kernel_i < kernel_size; kernel_i++){
          for (int kernel_j = 0; kernel_j < kernel_size; kernel_j++){
            arr[kernel_i * kernel_size + kernel_j] = get_pixel(im, i-(kernel_size-1)/2+kernel_j, j-(kernel_size-1)/2+kernel_i, k);
            qsort(arr, kernel_size*kernel_size, sizeof(float), cmpfunc);
            set_pixel(filtered_im, i, j, k, arr[(kernel_size-1)/2]);
          }
        }
      }
    }
  }
  return filtered_im;
}


// SUPER EXTRA CREDIT: Bilateral filter


image apply_bilateral_filter(image im, float sigma1, float sigma2)
{
  image gauss_1 = make_gaussian_filter(sigma1);
  image filtered_im = make_image(im.w, im.h, im.c);
  for(int k = 0; k < im.c; k++){
    for(int j = 0; j < im.h; j++){
      for(int i = 0; i < im.w; i++){
        // Make a filter
        image filter = make_image(gauss_1.w, gauss_1.h, 1);
        float f0 = get_pixel(im, i, j, k);
        for (int fj = 0; fj < filter.h; fj++){
          for (int fi = 0; fi < filter.w; fi++){
            float f1 = get_pixel(im, i-(filter.w - 1)/2+fi, j-(filter.h - 1)/2+fj, k);
            float v = exp((f1 - f0) * (f1 - f0) / pow(sigma2, 2) / -2.0) / TWOPI / pow(sigma2, 2);  
            float v0 = get_pixel(gauss_1, fi, fj, 0);
            set_pixel(filter, fi, fj, 0, v*v0);
          }
        }
        l1_normalize(filter);

        // Convolution
        float filtered_p = 0;
        for (int fj = 0; fj < filter.h; fj++){
          for (int fi = 0; fi < filter.w; fi ++){
              filtered_p += get_pixel(im, i-(filter.w - 1)/2+fi, j-(filter.h - 1)/2+fj, k) * get_pixel(filter, fi, fj, 0);
          }
        }
        set_pixel(filtered_im, i, j, k, filtered_p);        
      }
    }
  }
  return filtered_im;
}

