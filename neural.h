#include "model.h"

struct Model_t;

class NNModel{

public:
    virtual ~NNModel(){ };
    
    virtual void load_model(Model_t& model) = 0;
    
    virtual int forward(double* data, double* label) = 0;
    
    virtual void backward() = 0;
    
    virtual void sgd() = 0;

    virtual void forward_backward(double* data, double* label) = 0;

    virtual void forward_backward_sgd(double* data, double* label, int sample_size, int epoch) = 0;

    virtual void printTime() = 0;
};
