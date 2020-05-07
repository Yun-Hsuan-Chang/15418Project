#include "neural.h"
void sgd_wrapper(Model_t* model);

class CudaNeural : public NNModel{

private:
    Model_t* output_model;

    int hidden_size;
    double learning_rate;

    double* device_loss;

    double* device_alpha;
    double* device_beta;
    double* device_dalpha;
    double* device_dbeta;

    double* device_alpha_bias;
    double* device_beta_bias;
    double* device_dalpha_bias;
    double* device_dbeta_bias;

    double* device_feature;
    double* device_z;
    double* device_derivative_value;

    // for forward arguments and output
    double* device_label;
    int* device_predict_y;

public:

    CudaNeural();
    virtual ~CudaNeural();

    void load_model(Model_t& model);
    
    int forward(double* data, double* label);
    
    void backward();
    
    void sgd();

    void forward_backward(double* data, double* label);

    void forward_backward_sgd(double* data, double* label, int sample_size, int epoch);

    void printTime();

};