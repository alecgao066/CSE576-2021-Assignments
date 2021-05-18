#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#include "matrix.h"
#include <time.h>

// Frees an array of descriptors.
// descriptor *d: the array.
// int n: number of elements in array.
void free_descriptors(descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(d[i].data);
    }
    free(d);
}

// Create a feature descriptor for an index in an image.
// image im: source image.
// int i: index in image for the pixel we want to describe.
// returns: descriptor for that index.
descriptor describe_index(image im, int i)
{
    int w = 5;
    descriptor d;
    d.p.x = i%im.w;
    d.p.y = i/im.w;
    d.data = calloc(w*w*im.c, sizeof(float));
    d.n = w*w*im.c;
    int c, dx, dy;
    int count = 0;
    // If you want you can experiment with other descriptors
    // This subtracts the central value from neighbors
    // to compensate some for exposure/lighting changes.
    for(c = 0; c < im.c; ++c){
        float cval = im.data[c*im.w*im.h + i];
        for(dx = -w/2; dx < (w+1)/2; ++dx){
            for(dy = -w/2; dy < (w+1)/2; ++dy){
                float val = get_pixel(im, i%im.w+dx, i/im.w+dy, c);
                d.data[count++] = cval - val;
            }
        }
    }
    return d;
}

// Marks the spot of a point in an image.
// image im: image to mark.
// ponit p: spot to mark in the image.
void mark_spot(image im, point p)
{
    int x = p.x;
    int y = p.y;
    int i;
    for(i = -9; i < 10; ++i){
        set_pixel(im, x+i, y, 0, 1);
        set_pixel(im, x, y+i, 0, 1);
        set_pixel(im, x+i, y, 1, 0);
        set_pixel(im, x, y+i, 1, 0);
        set_pixel(im, x+i, y, 2, 1);
        set_pixel(im, x, y+i, 2, 1);
    }
}

// Marks corners denoted by an array of descriptors.
// image im: image to mark.
// descriptor *d: corners in the image.
// int n: number of descriptors to mark.
void mark_corners(image im, descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        mark_spot(im, d[i].p);
    }
}

// Creates a 1d Gaussian filter.
// float sigma: standard deviation of Gaussian.
// returns: single row image of the filter.
image make_1d_gaussian(float sigma)
{
    // TODO: make separable 1d Gaussian.
    // Make a filter with a size of 6 * sigma
    int size = ceil(6 * sigma);
    if (size % 2 == 0) size++;
    image filter = make_image(size, 1, 1);
    // Make a gaussian filter
    float f;
    int s = (size - 1) / 2;
    for (int j = 0; j < filter.w; j++){
        f = exp(((j - s) * (j - s)) / pow(sigma, 2) / -2.0) / pow(TWOPI, 0.5) / sigma;  
        set_pixel(filter, j, 0, 0, f);
    }

    // Let sum to be 1
    l1_normalize(filter);
    return filter;
}

// Smooths an image using separable Gaussian filter.
// image im: image to smooth.
// float sigma: std dev. for Gaussian.
// returns: smoothed image.
image smooth_image(image im, float sigma)
{
    // TODO: use two convolutions with 1d gaussian filter.
    image gaussian_x = make_1d_gaussian(sigma);
    image gaussian_y = make_1d_gaussian(sigma);
    gaussian_y.h = gaussian_y.w;
    gaussian_y.w = 1;
    image filtered_x = convolve_image(im, gaussian_x, 1);
    image filtered_xy = convolve_image(filtered_x, gaussian_y, 1);
    return filtered_xy;
   
}

// Calculate the structure matrix of an image.
// image im: the input image.
// float sigma: std dev. to use for weighted sum.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          third channel is IxIy.
image structure_matrix(image im, float sigma)
{
    image S = make_image(im.w, im.h, 3);
    // TODO: calculate structure matrix for im.
    // Calculate Ix and Iy
    image filter_x = make_gx_filter();
    image Ix = convolve_image(im, filter_x, 0);
    image filter_y = make_gy_filter();
    image Iy = convolve_image(im, filter_y, 0);

    // Calculate measures
    float ix, iy;
    for(int i = 0; i < im.h; i++){
        for (int j = 0; j< im.w; j++){
            ix = get_pixel(Ix, j, i, 0);
            iy = get_pixel(Iy, j, i, 0);

            set_pixel(S, j, i, 0, ix*ix);
            set_pixel(S, j, i, 1, iy*iy);
            set_pixel(S, j, i, 2, ix*iy);
        }   
    }

    // Weightings
    image filter_S = smooth_image(S, sigma);
    return filter_S;
}

// Estimate the cornerness of each pixel given a structure matrix S.
// image S: structure matrix for an image.
// returns: a response map of cornerness calculations.
image cornerness_response(image S)
{
    image R = make_image(S.w, S.h, 1);
    // TODO: fill in R, "cornerness" for each pixel using the structure matrix.
    // We'll use formulation det(S) - alpha * trace(S)^2, alpha = .06.
    float a00, a01, a10, a11;
    float det, trace, cornerness;
    float alpha = 0.06;
    for(int i = 0; i < S.h; i++){
        for (int j = 0; j < S.w; j++){
            a00 = get_pixel(S, j, i, 0);
            a11 = get_pixel(S, j, i, 1);
            a01 = get_pixel(S, j, i, 2);
            a10 = get_pixel(S, j, i, 2);

            det = a00 * a11 - a01 * a10;
            trace = a00 + a11;
            cornerness = det - alpha * pow(trace, 2);
            set_pixel(R, j, i, 0, cornerness);
        }
    }
    return R;
}

// Perform non-max supression on an image of feature responses.
// image im: 1-channel image of feature responses.
// int w: distance to look for larger responses.
// returns: image with only local-maxima responses within w pixels.
image nms_image(image im, int w)
{
    image r = copy_image(im);
    // TODO: perform NMS on the response map.
    // for every pixel in the image:
    //     for neighbors within w:
    //         if neighbor response greater than pixel response:
    //             set response to be very low (I use -999999 [why not 0??])
    float f, fw;
    int flag = 0;
    for (int i = 0; i < r.h; i++){
        for (int j = 0; j < r.w; j++){
            f = get_pixel(im, j, i, 0);
            flag = 0;
            for(int wi = (i - w >= 0 ? i - w : 0); wi <= (i + w < im.h ? i+ w : im.h); wi++){
                for(int wj = (j - w >= 0 ? j - w : 0); wj <= (j + w < im.w ? j+ w : im.w); wj++){
                    fw = get_pixel(im, wj, wi, 0);
                    if (fw > f){
                        set_pixel(r, j, i, 0, -999999);
                        flag = 1;
                        break;
                    }
                }
                if (flag == 1) break;
            } 
        }
    }
    return r;
}

// Perform harris corner detection and extract features from the corners.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
// int *n: pointer to number of corners detected, should fill in.
// returns: array of descriptors of the corners in the image.
descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms, int *n)
{
    // Calculate structure matrix
    image S = structure_matrix(im, sigma);

    // Estimate cornerness
    image R = cornerness_response(S);

    // Run NMS on the responses
    image Rnms = nms_image(R, nms);

    //TODO: count number of responses over threshold
    int count = 0; // change this
    float f;
    for (int i = 0; i < im.h; i++){
        for (int j =0; j < im.w; j++){
            f = get_pixel(Rnms, j, i, 0);
            if (f > thresh) count++;
        }
    }
    
    *n = count; // <- set *n equal to number of corners in image.
    descriptor *d = calloc(count, sizeof(descriptor));
    //TODO: fill in array *d with descriptors of corners, use describe_index.
    int counti = 0;
    for (int i = 0; i < im.h * im.w; i++){
        f = get_pixel(Rnms, i % im.w, i / im.w, 0);
        if (f > thresh){
            d[counti] = describe_index(im, i);
            counti++;
        }
    }

    free_image(S);
    free_image(R);
    free_image(Rnms);
    return d;
   
}

// Find and draw corners on an image.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
void detect_and_draw_corners(image im, float sigma, float thresh, int nms)
{
    int n = 0;
    descriptor *d = harris_corner_detector(im, sigma, thresh, nms, &n);
    mark_corners(im, d, n);
}
