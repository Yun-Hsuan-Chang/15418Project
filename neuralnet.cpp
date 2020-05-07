#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "math.h"
#include "float.h"
#include "CycleTimer.h"

#ifndef CUDA
#define CUDA 1
#endif

#ifndef MY_MPI
#define MY_MPI 0
#endif

#if !CUDA
#include "model.h"
#endif

#if CUDA
#include "cudaNeural.h"
#endif

#if MY_MPI
#include "mpi.cpp"
#endif

double init_t = 0;
double forward_t = 0;
double backward_t = 0;
double sgd_t = 0;

#if CUDA
NNModel* cudaModel;
#endif

void init_gradient(Model_t* model){
    int hidden_size = model->hidden_size;

    memset(model->dalpha, 0, hidden_size * sizeof(double));
    memset(model->dbeta, 0, hidden_size * sizeof(double));
    
    memset(model->dalpha_bias, 0, hidden_size * sizeof(double));
    memset(model->dbeta_bias, 0, OUTPUT_SIZE * sizeof(double));
}

double** file_operation(char* file, int* sample_size){
    double** labels = (double**) calloc(MAX_INPUT_LINE, sizeof(double*));

    FILE* fp;
    size_t len = 0;
    size_t cnt;
    char* line = NULL;
    fp = fopen(file, "r");

    int line_cnt = 0;
    int i;
    while((cnt = getline(&line, &len, fp)) != -1){
        labels[line_cnt] = (double*) calloc(OUTPUT_SIZE+INPUT_SIZE, sizeof(double));
        for(i = 0; i < OUTPUT_SIZE; ++i)
            labels[line_cnt][i] = 0;

        char *ptr = strtok(line, ",");
        int flag = 0;
        while(ptr != NULL){
            if(flag == 0){
                int label = atoi(ptr); 
                labels[line_cnt][label] = 1;
            }
            else{
                double feature;
                sscanf(ptr, "%lf", &feature);
                labels[line_cnt][OUTPUT_SIZE+flag-1] = feature;
            }
            ptr = strtok(NULL, ",");
            flag++;
        }
        line_cnt++;
    }
    fclose(fp);
    if(line) free(line);
    *sample_size = line_cnt;
    return labels;
}

double* dot_product(double* a, double* b, int row, int input_col, int col){
    double* result = (double*) calloc(row*col, sizeof(double));
    int j, k;

    for(int i = 0; i < row; ++i){
        for(j = 0; j < col; ++j){
            for(k = 0; k < input_col; ++k){
                result[i*col+j] += a[i*input_col+k] * b[k*col +j];
            }
        }
    }
    return result;
}

int argmax(double* x, int size){

    double max = FLT_MIN;
    int idx = 0, i;

    for(i = 0; i < size; ++i){
        if(x[i] > max){
            max = x[i];
            idx = i;
        }
    }
    return idx;
}

int forward( Model_t* model, double* data, double* label){

    double startTime = CycleTimer::currentSeconds();
    init_gradient(model);

#if CUDA
    int predict_y = cudaModel->forward(data, label);
#else

    int hidden_size = model->hidden_size;

    model->feature = data;
    double* a = dot_product(data ,model->alpha, 1, INPUT_SIZE, hidden_size);

    int i;

    for(i = 0; i < hidden_size; ++i){
        a[i] += model->alpha_bias[i];
        model->z[i] = 1/(1+exp(-a[i]));
    }
    
    double* b = dot_product(model->z, model->beta, 1, hidden_size, OUTPUT_SIZE);
    double y2[OUTPUT_SIZE];
    double sum = 0;

    for(i = 0; i < OUTPUT_SIZE; ++i){
        b[i] += model->beta_bias[i];
        y2[i] = exp(b[i]);
        sum += exp(b[i]);
    }
    double loss_sum = 0;
    model->loss = 0;
    for(i = 0; i < OUTPUT_SIZE; ++i){
        y2[i] /= sum;
        model->derivative_value[i] = -label[i] + y2[i];
        loss_sum += label[i] * log(y2[i]);
        model->loss -= label[i] * log(y2[i]);
    }
    model->loss = -loss_sum;
    int predict_y = argmax(b, OUTPUT_SIZE);

    free(a);
    free(b);
#endif

    double endtime = CycleTimer::currentSeconds();
    forward_t += endtime - startTime;

    return predict_y;
}

double backward(Model_t* model){

    double startTime = CycleTimer::currentSeconds();

#if CUDA
    cudaModel->backward();
#else
    int hidden_size = model->hidden_size;

    model->dbeta_bias = model->derivative_value;
    model->dbeta = dot_product(model->z, model->derivative_value, hidden_size, 1, OUTPUT_SIZE);

    double* tmp = dot_product(model->beta, model->derivative_value, hidden_size, OUTPUT_SIZE, 1);
    double* dl = (double*) calloc(hidden_size, sizeof(double));

    int i;
    for(i = 0; i < hidden_size; ++i){
        dl[i] = tmp[i] * (model->z[i] * (1-model->z[i]));
    }
    model->dalpha = dot_product(model->feature, dl, INPUT_SIZE, 1, hidden_size);
    model->dalpha_bias = dl;
    
#endif

    double endtime = CycleTimer::currentSeconds();
    backward_t += endtime - startTime;
    return model->loss;
}

void sgd( Model_t* model){

    double startTime = CycleTimer::currentSeconds();

#if CUDA
    cudaModel->sgd();

#else
    double learning_rate = model->learning_rate;
    double hidden_size = model->hidden_size;

    int i;
    for(i = 0; i < INPUT_SIZE*hidden_size; ++i){
        model->alpha[i] -= learning_rate * model->dalpha[i];
    }

    for(i = 0; i < OUTPUT_SIZE*hidden_size; ++i){
            model->beta[i] -= learning_rate * model->dbeta[i];
    }

    for(i = 0; i < hidden_size; ++i){
        model->alpha_bias[i] -= learning_rate * model->dalpha_bias[i];
    }

    for(i = 0; i < OUTPUT_SIZE; ++i){
        model->beta_bias[i] -= learning_rate * model->dbeta_bias[i];
    }
#endif
    double endtime = CycleTimer::currentSeconds();
    sgd_t += endtime - startTime;
}

void output( Model_t* model, double** labels, double** data, int flag, char* outfile, int sample_size, FILE* f){
    double error = 0;
    FILE *fp = fopen(outfile ,"w");
    
    int i;
    for(i = 0; i < sample_size; ++i){
        int predict_y = forward(model, data[i], labels[i]);
        fprintf(fp,"%d\n", predict_y);
        if(argmax(labels[i], OUTPUT_SIZE) != predict_y){
            error++;
        }
    }
    error/=sample_size;

    if(flag == 1) fprintf(f, "error(train) %lf\n", error);
    else fprintf(f, "error(test) %lf\n", error);
    fclose(fp);
}

void write_metrics( Model_t* model, double** labels, double** data, int flag, FILE* fp, int sample_size, int epoch){
    double result_loss = 0, loss;
    int i;

    for(i = 0; i < sample_size; ++i){
    #if CUDA
        cudaModel->forward_backward(data[i], labels[i]);
        loss = model->loss;
    #else
        forward(model, data[i], labels[i]);
        loss = backward(model);
    #endif
        result_loss += loss;
    }
    result_loss /= sample_size;
    if(flag == 1) fprintf(fp, "epoch=%d crossentropy(train) %lf\n", epoch, result_loss);
    else fprintf(fp, "epoch=%d crossentropy(test) %lf\n", epoch, result_loss);
}

int main(int argc, char** argv){

    double startTime = CycleTimer::currentSeconds();

    int process_count = 1;
    int process_id = 0;
    int i, j, k;

#if MY_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
#endif

    bool mpi_master = process_id == 0;

    int epoch;
    int hidden_size;
    double learning_rate;
    
    int train_sample;
    double** train_label;
    double** train_feature;

    int test_sample;
    double** test_label;
    double** test_feature;

    epoch = atoi(argv[6]);
    hidden_size = atoi(argv[7]);
    learning_rate = 0.1;
    sscanf(argv[9], "%lf", &learning_rate);

    if(mpi_master){
        
        double** train_data = file_operation(argv[1], &train_sample);

        train_label = (double**) malloc(train_sample * sizeof(double*));
        train_feature = (double**) malloc(train_sample * sizeof(double*));

        for(i = 0; i < train_sample; ++i){
            train_label[i] = (double*) calloc(OUTPUT_SIZE, sizeof(double));
            train_feature[i] = (double*) calloc(INPUT_SIZE, sizeof(double));

            for(j = 0; j < OUTPUT_SIZE; ++j)
                train_label[i][j] = train_data[i][j];
            for(j = 0; j < INPUT_SIZE; ++j)
                train_feature[i][j] = train_data[i][OUTPUT_SIZE+j];
        }
        free(train_data);

        double** test_data = file_operation(argv[2], &test_sample);
        test_label = (double**) malloc(test_sample * sizeof(double*));
        test_feature = (double**) malloc(test_sample * sizeof(double*));

        for(i = 0; i < test_sample; ++i){
            test_label[i] = (double*) calloc(OUTPUT_SIZE, sizeof(double));
            test_feature[i] = (double*) calloc(INPUT_SIZE, sizeof(double));
            for(j = 0; j < OUTPUT_SIZE; ++j)
                test_label[i][j] = test_data[i][j];
            for(j = 0; j < INPUT_SIZE; ++j)
                test_feature[i][j] = test_data[i][OUTPUT_SIZE+j];
        }
        free(test_data);

    }

    Model_t* model = new Model_t(hidden_size, learning_rate);

#if MY_MPI

    int alpha_size = INPUT_SIZE * model->hidden_size;
    int beta_size = model->hidden_size * OUTPUT_SIZE;
    int alpha_bias_size = hidden_size;
    int beta_bias_size = OUTPUT_SIZE;

    if(mpi_master){
        // Initialize buffer
        model->allalpha = (double*) malloc(alpha_size * sizeof(double));
        model->allbeta = (double*) malloc(beta_size * sizeof(double));
        model->allalphabias = (double*) malloc(alpha_bias_size * sizeof(double));
        model->allbetabias = (double*) malloc(beta_bias_size * sizeof(double));

        send_data(train_feature, train_label, train_sample, test_feature, test_label, test_sample, process_count, model);
    }

    else{
        receive_data(model,  process_id);
    }

#else

    model->train_feature = train_feature;
    model->train_label = train_label;
    model->train_sample = train_sample;
    model->test_feature = test_feature;
    model->test_label = test_label;
    model->test_sample = test_sample;

#endif
    
    #if CUDA
        cudaModel = new CudaNeural();
        cudaModel->load_model(*model);
    #endif

    double init_end = CycleTimer::currentSeconds();
    init_t += init_end - startTime;

    FILE* metrics_out = fopen(argv[5], "w");


#if CUDA
    double* feature = (double*) calloc(model->train_sample * INPUT_SIZE, sizeof(double));
    double* label = (double*) calloc(model->train_sample * OUTPUT_SIZE, sizeof(double));

    for(j = 0; j < model->train_sample; ++j){
        for(k = 0; k < INPUT_SIZE; ++k){
            feature[j*INPUT_SIZE+k] = model->train_feature[j][k];
        }

        for(k = 0; k < OUTPUT_SIZE; ++k){
            label[j*OUTPUT_SIZE+k] = model->train_label[j][k];
        }
        // printf("\n");
    }
    cudaModel->forward_backward_sgd(feature, label,model->train_sample,epoch);
#else
    for(i = 0; i < epoch; ++i){
        for(j = 0; j < model->train_sample; ++j){
            forward(model, model->train_feature[j], model->train_label[j]);
            backward(model);
            sgd(model);
        }
        write_metrics(model, model->train_label, model->train_feature, 1, metrics_out, model->train_sample, i+1);
        write_metrics(model, model->test_label, model->test_feature, 0, metrics_out, model->test_sample, i+1);
    }
#endif


#if MY_MPI

    
    if(mpi_master){
        gather_results(model, process_count, process_id, alpha_size, beta_size, alpha_bias_size, beta_bias_size);
    }

    else{
        send_results(model, process_id, alpha_size, beta_size, alpha_bias_size, beta_bias_size);
    }

#endif

    if(mpi_master){
        output(model, train_label, train_feature, 1, argv[3], train_sample, metrics_out);
        output(model, test_label, test_feature, 0, argv[4], test_sample, metrics_out);
        fclose(metrics_out);

        double endtime = CycleTimer::currentSeconds();
        double totalTime = endtime - startTime;

        printf("Initialization:  %.4f sec\n", init_t);
    #if !CUDA
        printf("Forward:  %.4f sec\n", forward_t);
        printf("Backward:  %.4f sec\n", backward_t);
        printf("SGD:  %.4f sec\n", sgd_t);
    #else
        cudaModel->printTime();
    #endif
        printf("Overall:  %.4f sec\n", totalTime);
    }

#if MY_MPI
    MPI_Finalize();
#endif
}


// mpirun -n 8 ./neuralnet smallTrain.csv smallValidation.csv train2.txt test2.txt metrics2.txt 2 4 0 0.1

// mpirun -n 8 ./neuralnet largeTrain.csv largeValidation.csv train2.txt test2.txt metrics2.txt 100 9 0 0.1