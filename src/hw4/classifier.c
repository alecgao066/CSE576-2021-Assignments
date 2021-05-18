#include <math.h>
#include <stdlib.h>
#include "image.h"
#include "matrix.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LOGISTIC){
                // 1/ 1 - e^x
                m.data[i][j] = 1 / (1 + exp(-x));
            } else if (a == RELU){
                // TODO
                if (x <= 0) m.data[i][j] = 0;
                else m.data[i][j] = x;
            } else if (a == LRELU){
                // TODO
                if (x <= 0) m.data[i][j] = 0.1 * x;
                else m.data[i][j] = x;
            } else if (a == SOFTMAX){
                // TODO
                m.data[i][j] = exp(x);
                sum += m.data[i][j];
            }
            //sum += m.data[i][j];
        }
        if (a == SOFTMAX) {
            // TODO: have to normalize by sum if we are using SOFTMAX
            for (int k = 0; k < m.cols; k++){
                m.data[i][k] = m.data[i][k] / sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            // TODO: multiply the correct element of d by the gradient
            if(a == LOGISTIC){
                // f'(x) = f(x) * (1 - f(x))
                x = x * (1 - x);
                d.data[i][j] = d.data[i][j] * x;
            } else if (a == RELU){
                // 0 or 1. d(x=0) = 0. 
                if (x <= 0) d.data[i][j] = 0;
                else d.data[i][j] = d.data[i][j] * 1;
            } else if (a == LRELU){
                // 0.1 or 1. d(x=0) = 0. 1
                if (x <= 0) d.data[i][j] = d.data[i][j] * 0.1;
                else d.data[i][j] = d.data[i][j] * 1;
            } else if (a == SOFTMAX){
                // 1
                d.data[i][j] = d.data[i][j] * 1;
            }
        }
    }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in)
{

    l->in = in;  // Save the input for backpropagation

    // TODO: fix this! multiply input by weights and apply activation function.
    //matrix out = make_matrix(in.rows, l->w.cols);
    if (in.cols != l->w.rows){
        in = transpose_matrix(in);
        l->w = transpose_matrix(l->w);
    }
    matrix out = matrix_mult_matrix(in, l->w);
    activate_matrix(out, l->activation);

    free_matrix(l->out);// free the old output
    l->out = out;       // Save the current output for gradient calculation
    return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta)
{
    // 1.4.1
    // delta is dL/dy
    // TODO: modify it in place to be dL/d(xw)
    gradient_matrix(l->out, l->activation, delta);

    // 1.4.2
    // TODO: then calculate dL/dw and save it in l->dw
    matrix in_t = transpose_matrix(l->in);
    free_matrix(l->dw);
    matrix dw = matrix_mult_matrix(in_t, delta); // replace this
    l->dw = dw;

    
    // 1.4.3
    // TODO: finally, calculate dL/dx and return it.
    matrix w_t = transpose_matrix(l->w);
    matrix dx = matrix_mult_matrix(delta, w_t); 
    return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay)
{
    // TODO:
    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    // save it to l->v
    matrix dw_t1 = axpy_matrix(-1 * decay, l->w, l->dw);
    matrix dw_t2 = axpy_matrix(momentum, l->v, dw_t1);
    free_matrix(l->v);
    l->v = dw_t2;
    // Update l->w
    matrix w_t = axpy_matrix(rate, dw_t2, l->w);
    free_matrix(l->w);
    l->w = w_t;
    // Remember to free any intermediate results to avoid memory leaks
    free_matrix(dw_t1);
}

// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation)
{
    layer l;
    l.in  = make_matrix(1,1);
    l.out = make_matrix(1,1);
    l.w   = random_matrix(input, output, sqrt(2./input));
    l.v   = make_matrix(input, output);
    l.dw  = make_matrix(input, output);
    l.activation = activation;
    return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X)
{
    int i;
    for(i = 0; i < m.n; ++i){
        X = forward_layer(m.layers + i, X);
    }
    return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL)
{
    matrix d = copy_matrix(dL);
    int i;
    for(i = m.n-1; i >= 0; --i){
        matrix prev = backward_layer(m.layers + i, d);
        free_matrix(d);
        d = prev;
    }
    free_matrix(d);
}

// Update the model weights
// model m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_model(model m, double rate, double momentum, double decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        update_layer(m.layers + i, rate, momentum, decay);
    }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    double max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// Calculate the accuracy of a model on some data d
// model m: model to run
// data d: data to run on
// returns: accuracy, number correct / total
double accuracy_model(model m, data d)
{
    matrix p = forward_model(m, d.X);
    int i;
    int correct = 0;
    for(i = 0; i < d.y.rows; ++i){
        if(max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols)) ++correct;
    }
    return (double)correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p)
{
    int i, j;
    double sum = 0;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            sum += -y.data[i][j]*log(p.data[i][j]);
        }
    }
    return sum/y.rows;
}


// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix p = forward_model(m, b.X);
        fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
        matrix dL = axpy_matrix(-1, p, b.y); // partial derivative of loss dL/dy
        backward_model(m, dL);
        update_model(m, rate/batch, momentum, decay);
        free_matrix(dL);
        free_data(b);
    }
}


// Questions 
//
// 2.1.1 What are the training and test accuracy values you get? Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?
// Training accuracy is 90.34 %. Testing accuracy is 90.76 %. The training accuracy can tell how well the model fits the training data. The testing accuracy can tell the generalization capbility of the model.
// Our current model has a similarly high training and testing accuracy, which indicates both its bias and variance are small. It doesn't have obvious overfitting or underfitting problem.
//
// 2.1.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?
// The trianing and testing accuracy is highest when the leaning rate equals 0.1. The accuracy goes down if the leaning rate is too small or too large. If the learning rate is samll, the model can't reach the minimum error by the end of iterations. If the learning rate is large, the model jumps around too much in each iteration, unable to converge to the minimum error.
//
// 2.1.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?
// The trianing and testing accuracy is highest when the weight decay equals 0.1. The accuracy goes down if the weight decay is too small or too large.
//
// 2.2.1 Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?
// LRELU has the best performance, which achieves an accuracy of 94.88%. RELU has a slightly lower accuracy of 94.62. Logistic achieves over 93% accuracy and Linear only reaches an accuracy of about 90%.
//
// 2.2.2 Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?
// The best learning rate is 0.1. The training accuracy is 95.97 %. The testing accuracy is 95.41 %.
//
// 2.2.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?
// Yes, when the decay is 0.01, training accuracy becomes 96.01%, and testing accuracy becomes 95.43%.
// Adding weight decay improves the accuracy, because the weight decay can make the model better generalization ability and prevents overfitting.
//
// 2.2.4 Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing error for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
// 0.1 is the best. The training accuracy is lower than smaller decay values but the testing accuracy is highest. 
// The model has more layers and more parameters, for a complex model, it is easy to over fit the data. Adopting a large weight decay can prevent overfitting and promotes generalizaton.
//
// 3.1.1 What is the best training accuracy and testing accuracy? Summarize all the hyperparameter combinations you tried.
// The training accuracy is 49.41%, and the testing accuracy is 47.37%.
// I tried learning rate 0.1, 0.01 and 0.001. The accuracy is highest when learnning rate is 0.01.
// The best accuracy is achieved when weight decay is 0.1, among 0.01, 0.1 and 1.
// I tried iteration number 3000 and 5000. 5000 iterations give a higher accuracy.
// I tried LOGiSTIC, RELU, and LRELU, and found LRELU has the highest accuracy.
