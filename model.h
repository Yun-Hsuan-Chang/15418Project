#define OUTPUT_SIZE 10
#define INPUT_SIZE 128
#define MAX_INPUT_LINE 10000
#define MAX_PAR_LINE 2000

struct Model_t{

    Model_t(int hidden_size, double learning_rate){
        loss = 0;
        this->hidden_size = hidden_size;
        this->learning_rate = learning_rate;

        alpha = (double*) calloc(INPUT_SIZE*hidden_size, sizeof(double*));
        beta = (double*) calloc(hidden_size*OUTPUT_SIZE, sizeof(double*));
        alpha_bias = (double*) calloc(hidden_size, sizeof(double));
        beta_bias = (double*) calloc(OUTPUT_SIZE, sizeof(double));

        dalpha = (double*) calloc(INPUT_SIZE*hidden_size, sizeof(double*));
        dbeta = (double*) calloc(hidden_size*OUTPUT_SIZE, sizeof(double*));
        dalpha_bias = (double*) calloc(hidden_size, sizeof(double));
        dbeta_bias = (double*) calloc(OUTPUT_SIZE, sizeof(double));

        feature = (double*) calloc(INPUT_SIZE, sizeof(double));
        z = (double*) calloc(hidden_size, sizeof(double));
        derivative_value = (double*) calloc(OUTPUT_SIZE, sizeof(double));
    

    #if MY_MPI
        train_feature = (double**) calloc(MAX_PAR_LINE, sizeof(double*));

        train_label = (double**) calloc(MAX_PAR_LINE, sizeof(double*));
        test_feature = (double**) calloc(MAX_PAR_LINE, sizeof(double*));
        test_label = (double**) calloc(MAX_PAR_LINE, sizeof(double*));
 
        int i;
        for(i = 0; i < MAX_PAR_LINE; ++i){
            train_feature[i] = (double*) calloc(INPUT_SIZE, sizeof(double));
            train_label[i] = (double*) calloc(OUTPUT_SIZE, sizeof(double));
        
            test_feature[i] = (double*) calloc(INPUT_SIZE, sizeof(double));
            test_label[i] = (double*) calloc(OUTPUT_SIZE, sizeof(double));
        }
    #endif
    }

    double loss;

    double* alpha;
    double* beta;
    double* dalpha;
    double* dbeta;

    double* alpha_bias;
    double* beta_bias;

    double* dalpha_bias;
    double* dbeta_bias;
    
    double* feature;
    double* z;
    double* derivative_value;

    int hidden_size;
    double learning_rate;

    double* allalpha;
    double* allbeta;
    double* allalphabias;
    double* allbetabias;

    double** train_feature;
    double** train_label;
    int train_sample;

    double** test_feature;
    double** test_label;
    int test_sample;

    double* rtrain_feature;
    double* rtrain_label;
};


