# CSE 576, 2021 Spring

This is my solutions to CSE 576 2021 assignments. It is an image processing/computer vision library in C with Python API. Key features include color space operations, image filtering, panorama stitching, optical flow and neuron networks.
## Course Introduction
In this repository you will find instructions on how to build your own image processing/computer vision library from (mostly) scratch. The work is divided out into different homework assignments, found in the `src/` directory.

To get started, make sure you have `git`, a C compiler, and `make` installed. Then run:

```
git clone https://github.com/UW-CSE-576-2021SP/Homework
cd Homework
make
```

and check to see that everything compiles correctly. We recommend using Linux or MacOS for the homework since installing `make` is easier. Linux uses GNU C++ compiler, while MacOS uses XCode C++ compiler. 

## Instructions for Mac Users
In MacOS, make sure you have the latest version of Xcode and perform `xcode-select --install`. 
If `make` still results in an error, try [this](https://github.com/frida/frida/issues/338#issuecomment-426777849) solution.

## Instructions for Windows Users
We do **NOT** recommend Windows OS for this assignment because C++ compilation is more complex under the Windows environment. However, if you only have Windows computers available, you can still manage your Python packages, C++ compiler, and Makefile with Anaconda.

Installation Steps:
1. Download [Anaconda](https://www.anaconda.com/distribution/) with Python 3.6+
2. Install Anaconda with "admin" rights: PLEASE select "All Users (requires admin privileges)", "Add Anaconda to the system PATH environment variable", and "Register Anaconda as the system Python 3.x".
3. Open "Command Prompt" (aka "cmd") with admin rights, then:
    - run the command `conda install -c msys2 m2-base m2w64-gcc` to install C++ compiler 
    - run the command `conda install -c conda-forge make cmake` to install Make.
6. Now, you can follow the same instructions as Mac/Linux users do. 

## Results
1. Color space operations
Raw image
<img src=".\figs\results\dog.jpg" width = "500"> 
a. RGB to gray scale
<img src=".\figs\results\rgb_2_gray.jpg" width = "500">
b. Saturation adjustment in HSV color space
<img src=".\figs\results\dog_saturated.jpg" width = "500">
2. Resize
Raw image
<img src=".\figs\results\dogsmall.jpg">
a. Resize - nearest neighbor
<img src=".\figs\results\dog4x-nn.jpg">
b. Resize - bilinear
<img src=".\figs\results\dog4x-bl.jpg">
3. Low-pass filtering
Raw image
<img src=".\figs\results\dog.jpg">
a. Box filter
<img src=".\figs\results\dog-box7.jpg">
b. Gaussian filter
<img src=".\figs\results\dog-gauss2.jpg">
4. Edge detection
Raw image
<img src=".\figs\results\dog.jpg">
Edge
<img src=".\figs\results\magnitude.jpg">
5. Denoising
Raw image
<img src=".\figs\results\landscape.jpg">
a. Median filter
<img src=".\figs\results\median.jpg">
b. Bilateral filter
<img src=".\figs\results\bilateral.jpg">
6. Panorama stitching
a. Harris corner point detection
<img src=".\figs\results\corners.jpg">
b. Matching
<img src=".\figs\results\inliers.jpg">
c. Affine transformation and stitching
<img src=".\figs\results\easy_panorama.jpg">
d. panorama stitching of multiple images
<img src=".\figs\results\rainier_panorama_5.jpg">
e. panorama stitching of multiple images with cylindrical projections
<img src=".\figs\results\field_panorama_5.jpg">
7. Optical flow
<img src=".\figs\results\lines.jpg">

8. Neuron networks

a. Handwritten digit identification (MNIST)

| Structure | Learning rate | Weight decay | Activation | Iteration |
| ----------- | ----------- | ----------- |----------- |----------- |
| 3-layer Neuron Net | 0.1 | 0.1 |LRELU |3000 |  

96.01% accuracy on the training set.  
95.54% accuracy on the training set.  

b. Image classification (CIFAR10)
| Model | Number of parameters | Training accuracy | Validation accuracy |
| ----------- | ----------- | ----------- |----------- |
| Neuron Net | 346373 | 76.26% | 77.16% |
| Convolutional Neuron Net | 8069 | 82.89% | 82.44% |
| CNN with color norm | 8069 | 88.38% | 85.84% |
| Deep CNN | 29077 | 89.92% | 87.08% |
| Deep CNN with data augumentation | 29077 | 86.97% | 87.40% |
