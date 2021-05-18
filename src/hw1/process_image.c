#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "../image.h"

float get_pixel(image im, int x, int y, int c)
{
    // TODO Fill this in
    // index = x+y*w+c*w*h
    int index;
    if (x >= im.w){
        x = im.w - 1;
    }else if (x < 0){
        x = 0;
    }
    if (y >= im.h){
        y = im.h - 1;
    }else if(y < 0){
        y = 0;
    }
    if (c >= im.c){
        c = im.c - 1;
    }else if(c < 0){
        c = 0;
    }
    index = x + y * im.w + c * im.h*im.w;
    return im.data[index]; 
}

void set_pixel(image im, int x, int y, int c, float v)
{
    // TODO Fill this in
    // set image[index] = v
    if (x >= 0 && x < im.w && y >= 0 && y < im.h && c >= 0 && c < im.c)
    {
        int index = x + y * im.w + c * im.w * im.h;
        im.data[index] = v;
    }
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    // TODO Fill this in
    int size = im.w * im.h * im.c;
    memcpy(copy.data, im.data, sizeof(float)*size); // copy a certain sized data from source to destination
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    // TODO Fill this in
    // y = 0.299 R' + 0.587 G' + .114 B' (luma)
    float r, g, b, y;
    for (int i = 0; i < im.h; i++){
        for(int j=0; j < im.w; j++){
            r = get_pixel(im, j, i, 0);
            g = get_pixel(im, j, i, 1);
            b = get_pixel(im, j, i, 2);
            y = 0.299 * r + 0.587 * g + 0.114 * b;
            set_pixel(gray, j, i, 0, y);
        }
    }
    return gray;
}

void shift_image(image im, int c, float v)
{
    // TODO Fill this in
    // im[c] = im[c] + v
    float x, y;
    for (int i = 0; i < im.h; i++){
        for (int j = 0; j < im.w; j++){
            x = get_pixel(im, j, i, c);
            y = x + v;
            set_pixel(im, j, i, c, y);
        }
    }
}

void clamp_image(image im)
{
    // TODO Fill this in
    // if v<0, then v=0; if v>1, then v=1.
    float v;
    for(int k = 0; k < im.c; k++){
        for (int i = 0; i < im.h; i++){
            for (int j = 0; j < im.w; j++){
                v = get_pixel(im, j, i, k);
                if (v < 0) set_pixel(im, j, i, k, 0);
                else if (v > 1) set_pixel(im, j, i, k, 1);
            }
        }
    }
}


// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    // TODO Fill this in
    // V = max(r,g,b)
    // m = min(r,g,b) S = V-m/V (V!=0)
    // Calculate H', then H
    float h, s, v;
    float r, g, b;
    float m, h0;
    for (int i = 0; i < im.h; i++){
        for (int j = 0; j < im.w; j++){
            r = get_pixel(im, j, i, 0);
            g = get_pixel(im, j, i, 1);
            b = get_pixel(im, j, i, 2);
            // value
            v = three_way_max(r, g, b);
            // saturation
            m = three_way_min(r, g, b);
            if (v == 0) s = 0;
            else{
                s = (v - m)/v;
            }
            // hue
            if (r == g && r == b) h0 = 0;
            else if(v == r) h0 = (g - b)/(v - m);
            else if(v == g) h0 = (b - r)/(v - m) + 2;
            else if(v == b) h0 = (r - g)/(v - m) + 4;

            if (h0 < 0) h = h0/6 + 1;
            else h = h0/6;

            // set pixels
            set_pixel(im, j, i, 0, h);
            set_pixel(im, j, i, 1, s);
            set_pixel(im, j, i, 2, v);
        }
    }
}

void hsv_to_rgb(image im)
{
    // TODO Fill this in
    float h, s, v;
    int nr = 5, ng = 3, nb = 1;
    float kr, kg, kb;
    float r, g, b;

    for (int i=0; i < im.h; i++){
        for (int j = 0; j < im.w; j++){
            // get pixels
            h = get_pixel(im, j, i, 0);
            s = get_pixel(im, j, i, 1);
            v = get_pixel(im, j, i, 2);

            /**
            kr = (nr + h * 6) - (nr + h * 6) / 6;
            kg = (ng + h * 6) - (ng + h * 6) / 6;
            kb = (nb + h * 6) - (nb + h * 6) / 6;
            **/

            kr = fmod((nr + h * 6), 6);
            kg = fmod((ng + h * 6), 6);
            kb = fmod((nb + h * 6), 6);

            r = v - v * s * fmax(0, three_way_min(kr, 4 - kr, 1));
            g = v - v * s * fmax(0, three_way_min(kg, 4 - kg, 1));
            b = v - v * s * fmax(0, three_way_min(kb, 4 - kb, 1));

            // set pixels
            set_pixel(im, j, i, 0, r);
            set_pixel(im, j, i, 1, g);
            set_pixel(im, j, i, 2, b);
        }
    }
}

void scale_image(image im, int c, float v)
{
    // im[i, j, c] = im[i, j, c] * v
    float x, y;
    for (int i = 0; i < im.h; i++){
        for (int j = 0; j < im.w; j++){
            x = get_pixel(im, j, i, c);
            y = x * v;
            set_pixel(im, j, i, c, y);
        }
    }
}

void rgb_to_hcl (image im)
{
    float r, g, b;
    float rg, gg, bg;
    float x, y, z;
    float uu, vv;
    float un = 0.2009;
    float vn = 0.4610;
    float u, v, l;
    float c, h;

    for (int i = 0; i < im.h; i++){
        for (int j = 0; j < im.w; j++){
            // get pixels
            r = get_pixel(im, j, i, 0);
            g = get_pixel(im, j, i, 1);
            b = get_pixel(im, j, i, 2);

            // gamma decompression
            if (r <= 0.04045) rg = 25.0 * r / 323;
            else rg = pow((200.0 * r + 11) / 211, 12.0 / 5);
            if (g <= 0.04045) gg = 25.0 * g / 323;
            else gg = pow((200.0 * g + 11) / 211, 12.0 / 5);
            if (b <= 0.04045) bg = 25.0 * b / 323;
            else bg = pow((200.0 * b + 11) / 211, 12.0 / 5);

            // to CIE XYZ
            x = 0.4123908 * rg + 0.35758434 * gg + 0.18048079 * bg;
            y = 0.21263901 * rg + 0.71516868 * gg + 0.07219232 * bg;
            z = 0.01933082 * rg + 0.11919478 * gg + 0.95053215 * bg;

            // to CIE LUV
            // yn = 1, un = 0.2009, vn = 0.4610
            if (y <= pow(6.0/29, 3)) l = pow(29.0/3, 3) * y;
            else l = pow(116.0 * y, 1.0 / 3) - 16;

            uu = 4 * x / (-2 * x + 12 * y + 3);
            vv = 9 * y / (-2 * x + 12 * y + 3);
            u = 13 * l * (uu - un);
            v = 13 * l * (vv - vn);

            // to hcl
            c = pow(pow(u, 2) + pow(v, 2), 0.5);
            h = atan2(v, u);

            // set pixels
            set_pixel(im, j, i, 0, h);
            set_pixel(im, j, i, 1, c);
            set_pixel(im, j, i, 0, l);
        }
    }
}

void hcl_to_rgb (image im)
{
    float r, g, b;
    float rg, gg, bg;
    float x, y, z;
    float uu, vv;
    float un = 0.2009;
    float vn = 0.4610;
    float u, v, l;
    float c, h;

    for (int i = 0; i < im.h; i++){
        for (int j = 0; j < im.w; j++){
            // get pixels
            h = get_pixel(im, j, i, 0);
            c = get_pixel(im, j, i, 1);
            l = get_pixel(im, j, i, 2);

            // to luv
            u = c * cos(h);
            v = c * sin(h);

            // to XYZ
            uu = u / 13 / l + un;
            vv = v / 13 / l + vn;

            if(l <= 8) y = l * pow(3.0 / 29, 3);
            else y = pow((l + 16.0) / 116, 3);
            x = y * 9 * u / 4 / v;
            z = y * (12 - 3 * u - 20 * v) / 4 / v;

            // to grb linear
            rg = 3.24096994 * x - 1.53738318 * y - 0.49861076 * z;
            gg = -0.96924364 * x + 1.8759675 * y + 0.04155506 * z;
            bg = 0.05563008 * x - 0.20397696 * y + 1.05697151 * z;

            // gamma compression
            if (rg <= 0.0031308) r = 12.92 * rg;
            else r = pow(1.055 * rg, 1/2.4) - 0.055;
            if (gg <= 0.0031308) g = 12.92 * gg;
            else g = pow(1.055 * gg, 1/2.4) - 0.055;
            if (bg <= 0.0031308) b = 12.92 * bg;
            else b = pow(1.055 * bg, 1/2.4) - 0.055;
            
            // set pixels
            set_pixel(im, j, i, 0, r);
            set_pixel(im, j, i, 1, g);
            set_pixel(im, j, i, 0, b);
        }
    }
}
