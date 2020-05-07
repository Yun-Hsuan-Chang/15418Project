#include <mpi.h>

// Master send data to other processors
void send_data(double** train_feature, double** train_label, int train_sample, 
    double** test_feature, double** test_label, int test_sample, int process_count, Model_t* model){
    
    int i, j, k;
    MPI_Request Request;

    int train_part = train_sample / (process_count);
    int test_part = test_sample / (process_count);

    double* strain_feature = (double*) calloc(train_part * INPUT_SIZE, sizeof(double));
    double* strain_label = (double*) calloc(train_part * OUTPUT_SIZE, sizeof(double));

    double* stest_feature = (double*) calloc(test_part * INPUT_SIZE, sizeof(double));
    double* stest_label = (double*) calloc(test_part * OUTPUT_SIZE, sizeof(double));
   

    // master training data and label
    model->train_sample = train_part;
    model->test_sample = test_part;

    for(j = 0; j < train_part; ++j){
        for(k = 0; k < INPUT_SIZE; ++k){
            model->train_feature[j][k] = train_feature[j][k];
        }

        for(k = 0; k < OUTPUT_SIZE; ++k){
            model->train_label[j][k] = train_label[j][k];
        }
    }

    for(j = 0; j < test_part; ++j){
        for(k = 0; k < INPUT_SIZE; ++k){
            model->test_feature[j][k] = test_feature[j][k];
        }

        for(k = 0; k < OUTPUT_SIZE; ++k){
            model->test_label[j][k] = test_label[j][k];
        }
    }

    for(i = 1; i < process_count; ++i){
        MPI_Isend(&train_part, 1, MPI_INT, i, i, MPI_COMM_WORLD, &Request);

        for(j = 0; j < train_part; ++j){
            for(k = 0; k < INPUT_SIZE; ++k){
                strain_feature[j*INPUT_SIZE + k] = train_feature[(i) * train_part + j][k];
            }

            for(k = 0; k < OUTPUT_SIZE; ++k){
                strain_label[j*OUTPUT_SIZE + k] = train_label[(i) * train_part + j][k];
            }
        }
        MPI_Wait(&Request, MPI_STATUS_IGNORE);
        
        MPI_Isend(strain_feature, train_part*INPUT_SIZE, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &Request);
        MPI_Wait(&Request, MPI_STATUS_IGNORE);
        MPI_Isend(strain_label, train_part*OUTPUT_SIZE, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &Request);
        MPI_Wait(&Request, MPI_STATUS_IGNORE);

        MPI_Isend(&test_part, 1, MPI_INT, i, i, MPI_COMM_WORLD, &Request);

        for(j = 0; j < test_part; ++j){
            for(k = 0; k < INPUT_SIZE; ++k){
                stest_feature[j*INPUT_SIZE + k] = test_feature[i * test_part + j][k];
            }

            for(k = 0; k < OUTPUT_SIZE; ++k){
                stest_label[j*OUTPUT_SIZE + k] = test_label[i * test_part + j][k];
            }
        }
        MPI_Wait(&Request, MPI_STATUS_IGNORE);

        MPI_Isend(stest_feature, test_part*INPUT_SIZE, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &Request);
        MPI_Wait(&Request, MPI_STATUS_IGNORE);
        MPI_Isend(stest_label, test_part*OUTPUT_SIZE, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &Request);
        MPI_Wait(&Request, MPI_STATUS_IGNORE);
    }
    
}

void receive_data(Model_t* model, int process_id){

    MPI_Request request;
    int i, j;

    int train_sample;
    MPI_Recv(&train_sample, 1, MPI_INT, 0, process_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    model->train_sample = train_sample;
    double* rtrain_feature = (double*) calloc(train_sample * INPUT_SIZE, sizeof(double));
    double* rtrain_label = (double*) calloc(train_sample * OUTPUT_SIZE, sizeof(double)); 

    MPI_Recv(rtrain_feature, train_sample * INPUT_SIZE, MPI_DOUBLE, 0, process_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(rtrain_label, train_sample * OUTPUT_SIZE, MPI_DOUBLE, 0, process_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    int test_sample;
    MPI_Recv(&test_sample, 1, MPI_INT, 0, process_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    model->test_sample = test_sample;
    double* rtest_feature = (double*) calloc(test_sample * INPUT_SIZE, sizeof(double));
    double* rtest_label = (double*) calloc(test_sample * OUTPUT_SIZE, sizeof(double));
    
    MPI_Recv(rtest_feature, test_sample * INPUT_SIZE, MPI_DOUBLE, 0, process_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(rtest_label, test_sample * OUTPUT_SIZE, MPI_DOUBLE, 0, process_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    for(i = 0; i < train_sample; ++i){
        for(j = 0; j < INPUT_SIZE; ++j){
            model->train_feature[i][j] = rtrain_feature[i*INPUT_SIZE + j];
        }

        for(j = 0; j < OUTPUT_SIZE; ++j){
            model->train_label[i][j] = rtrain_label[i*OUTPUT_SIZE + j];
        }
    }

    for(i = 0; i < test_sample; ++i){
        for(j = 0; j < INPUT_SIZE; ++j){
            model->test_feature[i][j] = rtest_feature[i*INPUT_SIZE + j];
        }

        for(j = 0; j < OUTPUT_SIZE; ++j){
            model->test_label[i][j] = rtest_label[i*OUTPUT_SIZE + j];
        }
    }

    free(rtest_feature);
    free(rtest_label);
    free(rtrain_feature);
    free(rtrain_label);
}

void send_results(Model_t* model, int process_id, int alpha_size, int beta_size, int alpha_bias_size, int beta_bias_size){
    
	MPI_Request request;

    MPI_Isend(model->alpha, alpha_size, MPI_DOUBLE, 0 ,process_id, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    MPI_Isend(model->beta, beta_size, MPI_DOUBLE, 0 ,process_id, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    MPI_Isend(model->alpha_bias, alpha_bias_size, MPI_DOUBLE, 0 ,process_id, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    MPI_Isend(model->beta_bias, beta_bias_size, MPI_DOUBLE, 0 ,process_id, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
}

void gather_results(Model_t* model, int thread_count, int process_id, int alpha_size, int beta_size, int alpha_bias_size, int beta_bias_size){
	int i, j;
 
	for (i = 1; i < thread_count; i++){
		MPI_Recv(model->allalpha, alpha_size, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
		MPI_Recv(model->allbeta, beta_size, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(model->allalphabias, alpha_bias_size, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(model->allbetabias, beta_bias_size, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(j = 0; j < alpha_size; j++){
            model->alpha[j] += model->allalpha[j];
        }

        for (j = 0; j < beta_size; j++){
            model->beta[j] += model->allbeta[j];
        }

        for (j = 0; j < alpha_bias_size; j++){
            model->alpha_bias[j] += model->allalphabias[j];
        }
        for (j = 0; j < beta_bias_size; j++){
            model->beta_bias[j] += model->allbetabias[j];
        }       
	}

	for (j = 0; j < alpha_size; j++){
		model->alpha[j] /= (double)thread_count;
	}

	for (j = 0; j < beta_size; j++){
	   model->beta[j] /= (double)thread_count;
	} 

	for (j = 0; j < alpha_bias_size; j++){
		model->alpha_bias[j] /= (double)thread_count;
	}

	for (j = 0; j < beta_bias_size; j++){
		model->beta_bias[j]  /= (double)thread_count;
	}	

}