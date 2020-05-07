#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "math.h"
#include "float.h"

#include "CycleTimer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaNeural.h"

double trans_t = 0;
double cuda_forward_t = 0;
double cuda_backward_t = 0;
double cuda_sgd_t = 0;

struct GlobalConstants{
    int input_size;
    int output_size;
    int hidden_size;
    double learning_rate;
};

__constant__ GlobalConstants cuConstParams;


__device__ double addAtomic(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ __inline__ double dot(double* a, double* b, int row, int input_col, int col, int i, int j){

    double result = 0;
    int k;

    for(k = 0; k < input_col; ++k){
        result += a[i*input_col+k] * b[k*col +j];
    }
    
    return result;
}

__global__ void kernel_sgd(double* alpha, double* dalpha, double* beta, double* dbeta, 
    double* alpha_bias, double* dalpha_bias, double* beta_bias, double* dbeta_bias){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int input_size = cuConstParams.input_size;
    int hidden_size = cuConstParams.hidden_size;
    int output_size = cuConstParams.output_size;
    double learning_rate = cuConstParams.learning_rate;

    if(i < input_size * hidden_size && j == 0){
        alpha[i] -= learning_rate * dalpha[i];
    }

    if(i < hidden_size * output_size && j == 0){
        beta[i] -= learning_rate * dbeta[i];
    }

    if(i < hidden_size && j == 0){
        alpha_bias[i] -= learning_rate * dalpha_bias[i];
    }

    if(i < output_size && j == 0){
        beta_bias[i] -= learning_rate * dbeta_bias[i];
    }
}

__global__ void kernel_back(double* alpha, double* dalpha, double* beta, double* dbeta, 
    double* dalpha_bias, double* dbeta_bias, double* feature, double* z, double* derivative_value){

    int hidden_size = cuConstParams.hidden_size;
    int output_size = cuConstParams.output_size;
    int input_size = cuConstParams.input_size;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < output_size && j == 0){
        dbeta_bias[i] = derivative_value[i];
    }

    if(i < hidden_size && j < output_size){
        dbeta[i*output_size+j] = dot(z, derivative_value, hidden_size, 1, output_size, i, j);
    }

    if(i < hidden_size && j < 1){
        double tmp = dot(beta, derivative_value, hidden_size, output_size, 1, i, j);
        tmp = tmp* z[i] * (1-z[i]);
        dalpha_bias[i] = tmp;
    }
    __syncthreads();

    if(i < input_size && j < hidden_size){
        dalpha[i*hidden_size+j] = dot(feature, dalpha_bias, input_size, 1, hidden_size, i, j);
    }
}


__global__ void kernel_forward(double* feature, double* z, double* derivative_value, double* loss,
    int* predict_y, double* alpha, double* beta, double* alpha_bias, double* beta_bias, double* label){
    
    __shared__ double b_arr[1024];
    double sum = 0;

    int hidden_size = cuConstParams.hidden_size;
    int output_size = cuConstParams.output_size;
    int input_size = cuConstParams.input_size;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
        

    double a;
    if(i == 0 && j < hidden_size){
        a = dot(feature, alpha, 1, input_size, hidden_size, i, j);
        a += alpha_bias[j];
        z[j] = 1/(1+exp(-a));
    }
    __syncthreads();

    double b;

    if(i == 0 && j < output_size){
        if(j == 0)
            loss[0] = 0;
        b = dot(z, beta, 1, hidden_size, output_size, i, j);
        b += beta_bias[j];
        b_arr[j] = b;
        derivative_value[j] = exp(b);
    }

    __syncthreads();

    if(i == 0 && j < output_size){
        sum = 0;
        double max = FLT_MIN;

        for(int k = 0; k < output_size; ++k){
            sum += derivative_value[k];

            if(j == 0){
                if(b_arr[k] > max){
                    max = b_arr[k];
                    predict_y[0] = k;
                }
            }
        }
    }
    __syncthreads();

    if(i == 0 && j < output_size){
        derivative_value[j] /= sum;
        double l = label[j] * log(derivative_value[j]);
        addAtomic(&loss[0], -l);
        derivative_value[j] -= label[j];
    }
}


CudaNeural::CudaNeural(){
    output_model = NULL;

    hidden_size = 0;
    learning_rate = 0;
    
    device_loss = NULL;

    device_alpha = NULL;
    device_beta = NULL;
    device_dalpha = NULL;
    device_dbeta = NULL;

    device_alpha_bias = NULL;
    device_beta_bias = NULL;
    device_dalpha_bias = NULL;
    device_dbeta_bias = NULL;

    device_feature = NULL;
    device_z = NULL;
    device_derivative_value = NULL;

    device_label = NULL;
    device_predict_y = NULL;
}

CudaNeural::~CudaNeural(){
    if(output_model){
        delete output_model;
    }

    if(device_alpha){
        cudaFree(device_alpha);
        cudaFree(device_beta);
    }

    if(device_loss){
        cudaFree(device_loss);
        cudaFree(device_predict_y);
    }

    if(device_alpha_bias){
        cudaFree(device_alpha_bias);
        cudaFree(device_beta_bias);
    }

    if(device_derivative_value){
        cudaFree(device_feature);
        cudaFree(device_z);
        cudaFree(device_derivative_value);
    }

    if(device_dalpha){
        cudaFree(device_dalpha);
        cudaFree(device_dbeta);
        cudaFree(device_dalpha_bias);
        cudaFree(device_dbeta_bias);
    }

    if(device_label){
        cudaFree(device_label);
    }
}

void
CudaNeural::load_model(Model_t& model){

    double startTime = CycleTimer::currentSeconds();

    hidden_size = model.hidden_size;
    learning_rate = model.learning_rate;
    output_model = new Model_t(hidden_size, learning_rate);
    output_model = &model;
    cudaMalloc(&device_alpha, INPUT_SIZE * hidden_size * sizeof(double));
    cudaMalloc(&device_beta, hidden_size * OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&device_feature, INPUT_SIZE * sizeof(double));
    cudaMalloc(&device_z, hidden_size * sizeof(double));
    cudaMalloc(&device_derivative_value, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&device_loss, sizeof(double));
    cudaMalloc(&device_predict_y, sizeof(int));
    cudaMalloc(&device_alpha_bias, hidden_size * sizeof(double));
    cudaMalloc(&device_beta_bias, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&device_dalpha, INPUT_SIZE * hidden_size * sizeof(double));
    cudaMalloc(&device_dbeta, hidden_size * OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&device_dalpha_bias, hidden_size * sizeof(double));
    cudaMalloc(&device_dbeta_bias, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&device_label, OUTPUT_SIZE * sizeof(double));

    GlobalConstants params;
    params.input_size = INPUT_SIZE;
    params.output_size = OUTPUT_SIZE;
    params.hidden_size = hidden_size;
    params.learning_rate = output_model->learning_rate;

    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));

    double endTime = CycleTimer::currentSeconds();
    trans_t += endTime - startTime;
}


void
CudaNeural::sgd(){

    double startTime = CycleTimer::currentSeconds();

    int x = 1024, y = 1024/x;
    dim3 blockDim(x, 1);
    dim3 gridDimhidden((INPUT_SIZE * OUTPUT_SIZE + x - 1)/x, 1);

    double endTime = CycleTimer::currentSeconds();
    trans_t += endTime - startTime;

    double s_startTime = CycleTimer::currentSeconds();
    kernel_sgd<<<blockDim, gridDimhidden>>>(device_alpha, device_dalpha, device_beta, 
                device_dbeta, device_alpha_bias, device_dalpha_bias, device_beta_bias, device_dbeta_bias);
    double s_endTime = CycleTimer::currentSeconds();     

    cudaMemcpy(output_model->alpha, device_alpha, INPUT_SIZE * hidden_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_model->beta, device_beta, hidden_size * OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_model->alpha_bias, device_alpha_bias, hidden_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_model->beta_bias, device_beta_bias, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    endTime = CycleTimer::currentSeconds();

    cuda_sgd_t += s_endTime - s_startTime;
    trans_t += endTime - s_endTime;
}


void
CudaNeural::backward(){

    double startTime = CycleTimer::currentSeconds();

    int x = 1024, y = 1024/x;
    dim3 blockDim(x, y);
    dim3 gridDimhidden((INPUT_SIZE + x - 1)/x, (INPUT_SIZE + y - 1)/y);

    double endTime = CycleTimer::currentSeconds();
    trans_t += endTime - startTime;

    double b_startTime = CycleTimer::currentSeconds();

    kernel_back<<<blockDim, gridDimhidden>>>(device_alpha, device_dalpha, device_beta, device_dbeta, 
        device_dalpha_bias, device_dbeta_bias, device_feature, device_z, device_derivative_value);
    
    double b_endTime = CycleTimer::currentSeconds();        
    
    cudaMemcpy(output_model->dalpha, device_dalpha, INPUT_SIZE * hidden_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_model->dbeta, device_dbeta, hidden_size * OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_model->dalpha_bias, device_dalpha_bias, hidden_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_model->dbeta_bias, device_dbeta_bias, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    
    endTime = CycleTimer::currentSeconds();
    cuda_backward_t += b_endTime - b_startTime;
    trans_t += endTime - b_endTime;
}

int
CudaNeural::forward(double* data, double* label){

    double startTime = CycleTimer::currentSeconds();

    cudaMemcpy(device_feature, data, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_label, label, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    int x = 1024, y = 1024/x;
    dim3 blockDim(x, y);
    dim3 gridDimhidden((INPUT_SIZE + x - 1)/x, (INPUT_SIZE + y - 1)/y);

    double endTime = CycleTimer::currentSeconds();
    trans_t += endTime - startTime;

    double f_startTime = CycleTimer::currentSeconds();

    kernel_forward<<<blockDim, gridDimhidden>>>(device_feature, device_z, device_derivative_value, device_loss,
        device_predict_y, device_alpha, device_beta, device_alpha_bias, device_beta_bias, device_label);
    
    double f_endTime = CycleTimer::currentSeconds();
    
    int predict_y;
    cudaMemcpy(&predict_y, device_predict_y, sizeof(int), cudaMemcpyDeviceToHost);

    endTime = CycleTimer::currentSeconds();
    cuda_forward_t += f_endTime - f_startTime;
    trans_t += endTime - f_endTime;

    return predict_y;
}

void
CudaNeural::forward_backward(double* data, double* label){

    double startTime = CycleTimer::currentSeconds();

    cudaMemcpy(device_feature, data, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_label, label, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    
    
    int x = 1024, y = 1024/x;
    dim3 blockDim(x, y);
    dim3 gridDimhidden((INPUT_SIZE + x - 1)/x, (INPUT_SIZE + y - 1)/y);
    
    double endTime = CycleTimer::currentSeconds();
    trans_t += endTime - startTime;
    
    double f_startTime = CycleTimer::currentSeconds();
    
    kernel_forward<<<blockDim, gridDimhidden>>>(device_feature, device_z, device_derivative_value, device_loss,
        device_predict_y, device_alpha, device_beta, device_alpha_bias, device_beta_bias, device_label);

    double b_startTime = CycleTimer::currentSeconds();
    kernel_back<<<blockDim, gridDimhidden>>>(device_alpha, device_dalpha, device_beta, device_dbeta, 
        device_dalpha_bias, device_dbeta_bias, device_feature, device_z, device_derivative_value);
    
    double b_endTime = CycleTimer::currentSeconds();
            
    cudaMemcpy(&output_model->loss, device_loss, sizeof(double), cudaMemcpyDeviceToHost);

    endTime = CycleTimer::currentSeconds();

    cuda_forward_t += b_startTime - f_startTime;
    cuda_backward_t += b_endTime - b_startTime;
    trans_t += endTime - b_endTime;
}

void CudaNeural::forward_backward_sgd(double* data, double* label, int sample_size, int epoch){

    double startTime = CycleTimer::currentSeconds();
    cudaMalloc(&device_feature, sample_size*INPUT_SIZE * sizeof(double));
    cudaMalloc(&device_label, sample_size*OUTPUT_SIZE * sizeof(double));
    int i,j;
    cudaMemcpy(device_feature, data, sample_size*INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_label, label, sample_size*OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);


    int x = 128, y = 128/x;
    dim3 blockDim(x, y);
    dim3 gridDimhidden((INPUT_SIZE + x - 1)/x, (OUTPUT_SIZE+hidden_size + y - 1)/y);


    x = 1024;
    y = 1;
    dim3 blockDimsgd(x, y);
    dim3 gridDimsgd((INPUT_SIZE * hidden_size + x - 1)/x, 1);

    double endTime = CycleTimer::currentSeconds();

    trans_t += endTime - startTime;

    for(i = 0; i < epoch; ++i){
         for(j = 0; j < sample_size; ++j){
            
            double f_startTime = CycleTimer::currentSeconds();
            kernel_forward<<<blockDim, gridDimhidden>>>(device_feature+(j*INPUT_SIZE), device_z, device_derivative_value, device_loss,
                device_predict_y, device_alpha, device_beta, device_alpha_bias, device_beta_bias, device_label+(j*OUTPUT_SIZE));
            
            double b_startTime = CycleTimer::currentSeconds();
            kernel_back<<<blockDim, gridDimhidden>>>(device_alpha, device_dalpha, device_beta, device_dbeta, 
                device_dalpha_bias, device_dbeta_bias, device_feature+(j*INPUT_SIZE), device_z, device_derivative_value);

            double s_startTime = CycleTimer::currentSeconds();
            kernel_sgd<<<blockDimsgd, gridDimsgd>>>(device_alpha, device_dalpha, device_beta, 
                device_dbeta, device_alpha_bias, device_dalpha_bias, device_beta_bias, device_dbeta_bias);

            double s_endTime = CycleTimer::currentSeconds();

            cuda_forward_t += b_startTime - f_startTime;
            cuda_backward_t += s_startTime - b_startTime;
            cuda_sgd_t += s_endTime - s_startTime;
        }
    }

    
    cudaMemcpy(output_model->alpha, device_alpha, INPUT_SIZE * hidden_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_model->beta, device_beta, hidden_size * OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_model->alpha_bias, device_alpha_bias, hidden_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_model->beta_bias, device_dbeta_bias, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

}
void
CudaNeural::printTime(){
    printf("Cuda Setup:  %.4f sec\n", trans_t);
    printf("Cuda Forward:  %.4f sec\n", cuda_forward_t);
    printf("Cuda Backward:  %.4f sec\n", cuda_backward_t);
    printf("Cuda SGD:  %.4f sec\n", cuda_sgd_t);
}
