/*
In this experiment, we measure the memcpy time between receiving the packet 
and finish copying that to GPU-mapped memory.

In a normal setting that the client constantly sending data to GPU server, 
GPU is the bottleneck, thus network packets wait long for GPU to finish 
its job.

Thus, we simulate the situation similar to our experiment, in which lookups
are the bottleneck, thus GPU can always immediately consume what it receives.

Thus, in the client sender program, we limits the sending rate, so as to 
measure the latency between receiving the packet and cp that to GPU-mapped 
memory.
*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <unistd.h>
#include <time.h>
#include <pthread.h> 
#include <math.h>

#include <sys/types.h> 
#include <sys/ipc.h> 
#include <sys/shm.h>
#include <sys/socket.h> 
#include <netinet/in.h> 
#include <library_types.h>

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include "constant.h"

/* clock() is a low resolution one */
// clock_t network_time[TOTAL_BATCH_NUM]; // when finish receiving a batch from network 
// clock_t cuda_time[TOTAL_BATCH_NUM]; // when finish copying data from memory to CUDA 

typedef struct {
        time_t   tv_sec;        /* seconds */
        long     tv_nsec;       /* nanoseconds */
} timespec;

timespec network_time[TOTAL_BATCH_NUM]; // when finish receiving a batch from network 
timespec cuda_time[TOTAL_BATCH_NUM]; // when finish copying data from memory to CUDA 

timespec diff(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}


void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        // throw std::logic_error("cuda API failed");
    }
}

void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        // throw std::logic_error("cuBLAS API failed");
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            printf("CUBLAS_STATUS_NOT_INITIALIZED\n");
        }
        if (status == CUBLAS_STATUS_ALLOC_FAILED) {
            printf("CUBLAS_STATUS_ALLOC_FAILED\n");
        }
        if (status == CUBLAS_STATUS_INVALID_VALUE) {
            printf("CUBLAS_STATUS_INVALID_VALUE\n");
        }
        if (status == CUBLAS_STATUS_ARCH_MISMATCH) {
            printf("CUBLAS_STATUS_ARCH_MISMATCH\n");
        }
        if (status == CUBLAS_STATUS_MAPPING_ERROR) {
            printf("CUBLAS_STATUS_MAPPING_ERROR\n");
        }
        if (status == CUBLAS_STATUS_EXECUTION_FAILED) {
            printf("CUBLAS_STATUS_EXECUTION_FAILED\n");
        }
        if (status == CUBLAS_STATUS_INTERNAL_ERROR) {
            printf("CUBLAS_STATUS_INTERNAL_ERROR\n");
        }
        if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
            printf("CUBLAS_STATUS_NOT_SUPPORTED\n");
        }
        if (status == CUBLAS_STATUS_LICENSE_ERROR) {
            printf("CUBLAS_STATUS_LICENSE_ERROR\n");
        }
    }
}

void init_array(float* array, int length, float value) {
    for (int i = 0; i < length; i++) 
        array[i] = value;
}

struct Receiver_thread_info {
    int port;
    float* data;
    int* control;
};

struct CUDA_thread_info {
    float* data;
    int* control;
    cublasLtHandle_t* ltHandlePointer;
};

// A normal C function that is executed as a thread  
void *thread_receive_packets(void* vargp) 
{
    sleep(1); // wait CUDA init
    struct Receiver_thread_info* t_info = (struct Receiver_thread_info*) vargp;
    printf("Printing Port from Thread %d\n", t_info -> port); 

    int* writer_block_id = t_info -> control + 0;
    int* writer_iteration = t_info -> control + 1;
    int* reader_block_id = t_info -> control + 2;
    int* reader_iteration = t_info -> control + 3;

    ////////////////////  Network Init  ////////////////////

    int server_fd, new_socket, valread; 
    struct sockaddr_in address; 
    int opt = 1; 
    int addrlen = sizeof(address); 
    char *finish = "Finish receiving."; 
       
    // Creating socket file descriptor 
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) 
    { 
        perror("socket failed"); 
        exit(EXIT_FAILURE); 
    } 
   // Forcefully attaching socket to the port 8080 
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR , &opt, sizeof(opt)))
    { 
        perror("setsockopt"); 
        exit(EXIT_FAILURE); 
    } 

    address.sin_family = AF_INET; 
    address.sin_addr.s_addr = INADDR_ANY; 
    address.sin_port = htons( t_info -> port ); 
       
    // Forcefully attaching socket to the port 8080 
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 
    if (listen(server_fd, 3) < 0) 
    { 
        perror("listen"); 
        exit(EXIT_FAILURE); 
    } 
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address,  
                       (socklen_t*)&addrlen))<0) 
    { 
        perror("accept"); 
        exit(EXIT_FAILURE); 
    } 
    printf("Successfully built connection.\n");

    ////////////////////  Network Receive -> Shared Memory  ////////////////////

    clock_t start = clock();
    // use loop instead of memset for floating point init
    for (int iter = 0; iter < LOOP_NUM; iter++) {

        *writer_iteration = iter;
        // id in the queue
        for (int block_id = 0; block_id < BATCH_NUM_PER_LOOP; block_id++) {

#ifdef DEBUG
            printf("iteration: %d\tblock_id: %d\n", iter, block_id);
#endif
            // NOTE: if testing performance, comment the control logic below
            // 2 cases where writer need to wait reader to finish read
            // Case 1: If (reader hasn’t finished read this region last round, then writer halt) 
            // -> almost in front of reader for an entire round
            while ((iter > *reader_iteration) && (block_id + 1 >= *reader_block_id)) {
                // halt until the reader proceed
            }
            // Case 2: the writer is near the end of the iteration while the 
            // reader just start, in this case the distance still need to be kept
            while ((block_id == BATCH_NUM_PER_LOOP - 1) 
                && (iter >= *reader_iteration) // consider the case when read_iter=-1
                && (*reader_block_id < 1)) { // consider the case when read_iter=-1
                // halt until the reader proceed
            }  
            // WENQI: add this info, so that reader and writer are not very far away from each other
            //   this enables lower latency
            // while (*writer_block_id - 2 > *reader_block_id) {
                
            // }

            // write block
            int block_addr = block_id * BLOCK_ENTRY_NUM;
            int total_recv_bytes = 0;
            // replace with network transfer
            while(total_recv_bytes < BLOCK_SIZE) {
                int recv_bytes = read(new_socket, ((char*) (t_info -> data))  + block_addr * 4 + total_recv_bytes, 
                    BLOCK_SIZE - total_recv_bytes);
                if (recv_bytes == -1) {
                    printf("Receiving data UNSUCCESSFUL!\n");
                    return -1;
                }
                total_recv_bytes += recv_bytes;
            }
            if (total_recv_bytes != BLOCK_SIZE) {
                printf("Receiving error, receiving more bytes than a block!\n");
            }
            // network_time[iter * BATCH_NUM_PER_LOOP + block_id] = clock();
	        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &network_time[iter * BATCH_NUM_PER_LOOP + block_id]);

            *writer_block_id = block_id;
        }

    }
    clock_t end = clock() ;
    float elapsed_time = (end-start) / (float)CLOCKS_PER_SEC;
    printf("Consumed time: %f seconds, INCLUDING Waiting eader proceess\n", elapsed_time);
    printf("Throughput: %f GB / sec", (float)SHM_DATA_SIZE * LOOP_NUM / elapsed_time / 1024 / 1024 / 1024);

    return NULL; 
} 

void *thread_consume(void* vargp) {

    struct CUDA_thread_info* t_info = (struct CUDA_thread_info*) vargp;

    int* writer_block_id = t_info -> control + 0;
    int* writer_iteration = t_info -> control + 1;
    int* reader_block_id = t_info -> control + 2;
    int* reader_iteration = t_info -> control + 3;
    cublasLtHandle_t ltHandle = *(t_info -> ltHandlePointer);

    // ////////// Size definition ///////////
    // ele -> element, number of parameters
    int ele_input_feature = INPUT_FEATURE_LEN * BATCH_SIZE;
    int ele_r_layer1 = HIDDEN_SIZE1 * BATCH_SIZE; 
    int ele_r_layer2 = HIDDEN_SIZE2 * BATCH_SIZE; 
    int ele_r_layer3 = HIDDEN_SIZE3 * BATCH_SIZE;
    int ele_r_layer_out = OUTPUT_FEATURE_LEN * BATCH_SIZE;

    int ele_w_layer1 = HIDDEN_SIZE1 * INPUT_FEATURE_LEN;
    int ele_b_layer1 = HIDDEN_SIZE1 * BATCH_SIZE;
    int ele_w_layer2 = HIDDEN_SIZE2 * HIDDEN_SIZE1;
    int ele_b_layer2 = HIDDEN_SIZE2 * BATCH_SIZE;
    int ele_w_layer3 = HIDDEN_SIZE3 * HIDDEN_SIZE2;
    int ele_b_layer3 = HIDDEN_SIZE3 * BATCH_SIZE;
    int ele_w_layer_out = OUTPUT_FEATURE_LEN * HIDDEN_SIZE3;
    int ele_b_layer_out = OUTPUT_FEATURE_LEN * BATCH_SIZE;

    //////////Allocation: host ///////////
    float *input_feature, *r_layer1, 
        *r_layer2, *r_layer3, *r_layer_out; 
    float *w_layer1, *b_layer1, 
        *w_layer2, *b_layer2, 
        *w_layer3, *b_layer3,
        *w_layer_out, *b_layer_out;

    // input and intermediate results
    checkCudaStatus(cudaMallocHost((void**)&input_feature, ele_input_feature * sizeof(float)));   
    checkCudaStatus(cudaMallocHost((void**)&r_layer1, ele_r_layer1 * sizeof(float)));   
    checkCudaStatus(cudaMallocHost((void**)&r_layer2, ele_r_layer2 * sizeof(float)));   
    checkCudaStatus(cudaMallocHost((void**)&r_layer3, ele_r_layer3 * sizeof(float)));   
    checkCudaStatus(cudaMallocHost((void**)&r_layer_out, ele_r_layer_out * sizeof(float)));   

    // weights that should be copy to device at the beginning
    checkCudaStatus(cudaMallocHost((void**)&w_layer1, ele_w_layer1 * sizeof(float)));   
    checkCudaStatus(cudaMallocHost((void**)&b_layer1, ele_b_layer1 * sizeof(float)));   
    checkCudaStatus(cudaMallocHost((void**)&w_layer2, ele_w_layer2 * sizeof(float)));   
    checkCudaStatus(cudaMallocHost((void**)&b_layer2, ele_b_layer2 * sizeof(float)));   
    checkCudaStatus(cudaMallocHost((void**)&w_layer3, ele_w_layer3 * sizeof(float)));   
    checkCudaStatus(cudaMallocHost((void**)&b_layer3, ele_b_layer3 * sizeof(float)));   
    checkCudaStatus(cudaMallocHost((void**)&w_layer_out, ele_w_layer_out * sizeof(float)));   
    checkCudaStatus(cudaMallocHost((void**)&b_layer_out, ele_b_layer_out * sizeof(float)));   

    init_array(input_feature, ele_input_feature, 1.0f);
    init_array(w_layer1, ele_w_layer1, 1.0f);
    init_array(b_layer1, ele_b_layer1, 1.0f);
    init_array(w_layer2, ele_w_layer2, 1.0f);
    init_array(b_layer2, ele_b_layer2, 1.0f);
    init_array(w_layer3, ele_w_layer3, 1.0f);
    init_array(b_layer3, ele_b_layer3, 1.0f);
    init_array(w_layer_out, ele_w_layer_out, 1.0f);
    init_array(b_layer_out, ele_b_layer_out, 1.0f);

    //////////  Allocation: device ///////////
    float *d_input_feature, *d_r_layer1, 
        *d_r_layer2, *d_r_layer3, *d_r_layer_out;
    float *d_w_layer1, *d_b_layer1, 
        *d_w_layer2, *d_b_layer2, 
        *d_w_layer3, *d_b_layer3, 
        *d_w_layer_out, *d_b_layer_out;

    checkCudaStatus(cudaMalloc((void**)&d_input_feature, ele_input_feature * sizeof(float))); 
    checkCudaStatus(cudaMalloc((void**)&d_r_layer1, ele_r_layer1 * sizeof(float)));   
    checkCudaStatus(cudaMalloc((void**)&d_r_layer2, ele_r_layer2 * sizeof(float)));  
    checkCudaStatus(cudaMalloc((void**)&d_r_layer3, ele_r_layer3 * sizeof(float)));   
    checkCudaStatus(cudaMalloc((void**)&d_r_layer_out, ele_r_layer_out * sizeof(float)));  
    
    checkCudaStatus(cudaMalloc((void**)&d_w_layer1, ele_w_layer1 * sizeof(float)));   
    checkCudaStatus(cudaMalloc((void**)&d_b_layer1, ele_b_layer1 * sizeof(float)));   
    checkCudaStatus(cudaMalloc((void**)&d_w_layer2, ele_w_layer2 * sizeof(float)));   
    checkCudaStatus(cudaMalloc((void**)&d_b_layer2, ele_b_layer2 * sizeof(float)));   
    checkCudaStatus(cudaMalloc((void**)&d_w_layer3, ele_w_layer3 * sizeof(float)));   
    checkCudaStatus(cudaMalloc((void**)&d_b_layer3, ele_b_layer3 * sizeof(float)));   
    checkCudaStatus(cudaMalloc((void**)&d_w_layer_out, ele_w_layer_out * sizeof(float)));   
    checkCudaStatus(cudaMalloc((void**)&d_b_layer_out, ele_b_layer_out * sizeof(float)));   

    ////////// CublasLt Init //////////
    cudaStream_t stream;
    checkCudaStatus(cudaStreamCreate(&stream));

    ////////// LtSgemm Settings //////////

    // Layer 1
    size_t workspaceSize_layer1;
    void *workspace_layer1;
    // V100: shared memory size = 128KB
    // so I guess workspace size should be Global memory size
    workspaceSize_layer1 = 1024 * 1024 * 8;
    checkCudaStatus(cudaMalloc(&workspace_layer1, workspaceSize_layer1));

    cublasLtMatmulDesc_t operationDesc_layer1;
    cublasLtMatrixLayout_t weightsDesc_layer1, inputDesc_layer1, outputDesc_layer1;
    cublasLtMatmulPreference_t preference_layer1;
    cublasOperation_t transWeights_layer1;
    cublasOperation_t transInput_layer1;
    transWeights_layer1 = CUBLAS_OP_N;
    transInput_layer1 = CUBLAS_OP_N;

    int returnedResults_layer1;
    cublasLtMatmulHeuristicResult_t heuristicResult_layer1;


    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc_layer1, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc_layer1, CUBLASLT_MATMUL_DESC_TRANSA, &transWeights_layer1, sizeof(transWeights_layer1)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc_layer1, CUBLASLT_MATMUL_DESC_TRANSB, &transInput_layer1, sizeof(transInput_layer1)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&weightsDesc_layer1, CUDA_R_32F, HIDDEN_SIZE1, INPUT_FEATURE_LEN, HIDDEN_SIZE1));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&inputDesc_layer1, CUDA_R_32F, INPUT_FEATURE_LEN, BATCH_SIZE, INPUT_FEATURE_LEN));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&outputDesc_layer1, CUDA_R_32F, HIDDEN_SIZE1, BATCH_SIZE, HIDDEN_SIZE1));

    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference_layer1));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference_layer1, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize_layer1, sizeof(workspaceSize_layer1)));

    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc_layer1, weightsDesc_layer1, inputDesc_layer1, outputDesc_layer1, outputDesc_layer1, 
        preference_layer1, 1, &heuristicResult_layer1, &returnedResults_layer1));

    if (returnedResults_layer1 == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    
 
    // layer 2
    size_t workspaceSize_layer2;
    void *workspace_layer2;
    // V100: shared memory size = 128KB
    // so I guess workspace size should be Global memory size
    workspaceSize_layer2 = 1024 * 1024 * 8;
    checkCudaStatus(cudaMalloc(&workspace_layer2, workspaceSize_layer2));

    cublasLtMatmulDesc_t operationDesc_layer2;
    cublasLtMatrixLayout_t weightsDesc_layer2, inputDesc_layer2, outputDesc_layer2;
    cublasLtMatmulPreference_t preference_layer2;
    cublasOperation_t transWeights_layer2;
    cublasOperation_t transInput_layer2;
    transWeights_layer2 = CUBLAS_OP_N;
    transInput_layer2 = CUBLAS_OP_N;

    int returnedResults_layer2;
    cublasLtMatmulHeuristicResult_t heuristicResult_layer2;

    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc_layer2, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc_layer2, CUBLASLT_MATMUL_DESC_TRANSA, &transWeights_layer2, sizeof(transWeights_layer2)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc_layer2, CUBLASLT_MATMUL_DESC_TRANSB, &transInput_layer2, sizeof(transInput_layer2)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&weightsDesc_layer2, CUDA_R_32F, HIDDEN_SIZE2, HIDDEN_SIZE1, HIDDEN_SIZE2));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&inputDesc_layer2, CUDA_R_32F, HIDDEN_SIZE1, BATCH_SIZE, HIDDEN_SIZE1));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&outputDesc_layer2, CUDA_R_32F, HIDDEN_SIZE2, BATCH_SIZE, HIDDEN_SIZE2));

    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference_layer2));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference_layer2, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize_layer2, sizeof(workspaceSize_layer2)));

    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc_layer2, weightsDesc_layer2, inputDesc_layer2, outputDesc_layer2, outputDesc_layer2, 
        preference_layer2, 1, &heuristicResult_layer2, &returnedResults_layer2));

    if (returnedResults_layer2 == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    // layer 3
    size_t workspaceSize_layer3;
    void *workspace_layer3;
    // V100: shared memory size = 128KB
    // so I guess workspace size should be Global memory size
    workspaceSize_layer3 = 1024 * 1024 * 8;
    checkCudaStatus(cudaMalloc(&workspace_layer3, workspaceSize_layer3));

    cublasLtMatmulDesc_t operationDesc_layer3;
    cublasLtMatrixLayout_t weightsDesc_layer3, inputDesc_layer3, outputDesc_layer3;
    cublasLtMatmulPreference_t preference_layer3;
    cublasOperation_t transWeights_layer3;
    cublasOperation_t transInput_layer3;
    transWeights_layer3 = CUBLAS_OP_N;
    transInput_layer3 = CUBLAS_OP_N;

    int returnedResults_layer3;
    cublasLtMatmulHeuristicResult_t heuristicResult_layer3;

    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc_layer3, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc_layer3, CUBLASLT_MATMUL_DESC_TRANSA, &transWeights_layer3, sizeof(transWeights_layer3)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc_layer3, CUBLASLT_MATMUL_DESC_TRANSB, &transInput_layer3, sizeof(transInput_layer3)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&weightsDesc_layer3, CUDA_R_32F, HIDDEN_SIZE3, HIDDEN_SIZE2, HIDDEN_SIZE3));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&inputDesc_layer3, CUDA_R_32F, HIDDEN_SIZE2, BATCH_SIZE, HIDDEN_SIZE2));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&outputDesc_layer3, CUDA_R_32F, HIDDEN_SIZE3, BATCH_SIZE, HIDDEN_SIZE3));

    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference_layer3));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference_layer3, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize_layer3, sizeof(workspaceSize_layer3)));

    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc_layer3, weightsDesc_layer3, inputDesc_layer3, outputDesc_layer3, outputDesc_layer3, 
        preference_layer3, 1, &heuristicResult_layer3, &returnedResults_layer3));

    if (returnedResults_layer3 == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    
    // layer out
    size_t workspaceSize_layer_out;
    void *workspace_layer_out;
    // V100: shared memory size = 128KB
    // so I guess workspace size should be Global memory size
    workspaceSize_layer_out = 1024 * 1024 * 8;
    checkCudaStatus(cudaMalloc(&workspace_layer_out, workspaceSize_layer_out));

    cublasLtMatmulDesc_t operationDesc_layer_out;
    cublasLtMatrixLayout_t weightsDesc_layer_out, inputDesc_layer_out, outputDesc_layer_out;
    cublasLtMatmulPreference_t preference_layer_out;
    cublasOperation_t transWeights_layer_out;
    cublasOperation_t transInput_layer_out;
    transWeights_layer_out = CUBLAS_OP_N;
    transInput_layer_out = CUBLAS_OP_N;

    int returnedResults_layer_out;
    cublasLtMatmulHeuristicResult_t heuristicResult_layer_out;

    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc_layer_out, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc_layer_out, CUBLASLT_MATMUL_DESC_TRANSA, &transWeights_layer_out, sizeof(transWeights_layer_out)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc_layer_out, CUBLASLT_MATMUL_DESC_TRANSB, &transInput_layer_out, sizeof(transInput_layer_out)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&weightsDesc_layer_out, CUDA_R_32F, OUTPUT_FEATURE_LEN, HIDDEN_SIZE3, OUTPUT_FEATURE_LEN));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&inputDesc_layer_out, CUDA_R_32F, HIDDEN_SIZE3, BATCH_SIZE, HIDDEN_SIZE3));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&outputDesc_layer_out, CUDA_R_32F, OUTPUT_FEATURE_LEN, BATCH_SIZE, OUTPUT_FEATURE_LEN));

    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference_layer_out));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference_layer_out, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize_layer_out, sizeof(workspaceSize_layer_out)));

    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc_layer_out, weightsDesc_layer_out, inputDesc_layer_out, outputDesc_layer_out, outputDesc_layer_out, 
        preference_layer_out, 1, &heuristicResult_layer_out, &returnedResults_layer_out));

    if (returnedResults_layer_out == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    
    ////////// Before computation: weights H2D //////////

    checkCudaStatus(cudaMemcpyAsync(d_w_layer1, w_layer1, 
        ele_w_layer1 * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(d_w_layer2, w_layer2, 
        ele_w_layer2 * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(d_w_layer3, w_layer3, 
        ele_w_layer3 * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaMemcpyAsync(d_w_layer_out, w_layer_out, 
        ele_w_layer_out * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkCudaStatus(cudaStreamSynchronize(stream));

    const float alpha = 1.0f, beta = 0.0f;

    for (int iter = 0; iter < LOOP_NUM; iter++) {

        *reader_iteration = iter;
        // id in the queue
        for (int block_id = 0; block_id < BATCH_NUM_PER_LOOP; block_id++) {

    #ifdef DEBUG
            printf("iteration: %d\tblock_id: %d\n", iter, block_id);
            printf("control signal: writer_iteration %d\twriter_block_id%d\n",
                *writer_iteration, *writer_block_id);
    #endif
            // NOTE: if testing performance, comment the control logic below
            // 2 cases where reader need to wait writer to finish write
            // Case 1: In the same w/r iteration, writer is just in front of reader
            //   for 1 blcok
            while ((iter >= *writer_iteration) && (block_id + 1 >= *writer_block_id)) {
                // halt until the reader proceed
                // if last iteration, go through
                if ((iter == LOOP_NUM - 1) 
                    && (*writer_iteration == LOOP_NUM - 1)
                    && (*writer_block_id == BATCH_NUM_PER_LOOP - 1)) {
                    break;
                }
            }
            // Case 2: the reader is near the end of the iteration while the 
            //   writer just start writing the next iteration, in this case 
            //   the distance still need to be kept
            while ((block_id == BATCH_NUM_PER_LOOP - 1) 
                && (iter == *writer_iteration - 1) // consider the case when read_iter=-1
                && (*writer_block_id < 1)) { // consider the case when read_iter=-1
                // halt until the reader proceed
                // if last iteration, go through
                if ((iter == LOOP_NUM - 1) 
                    && (*writer_iteration == LOOP_NUM - 1)
                    && (*writer_block_id == BATCH_NUM_PER_LOOP - 1)) { 
                    break;
                }
            }  

            // write block
            int block_addr = block_id * BLOCK_ENTRY_NUM;
            memcpy(input_feature, t_info -> data + block_addr, BLOCK_SIZE);
            *reader_block_id = block_id;


            /////////////// input H2D /////////////// 
            checkCudaStatus(cudaMemcpyAsync(d_input_feature, input_feature, 
                ele_input_feature * sizeof(float), cudaMemcpyHostToDevice, stream));
	        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cuda_time[iter * BATCH_NUM_PER_LOOP + block_id]);
            // cuda_time[iter * BATCH_NUM_PER_LOOP + block_id] = clock();

            ///////////// Computation /////////////// 
            // if I change the third matrix to bias, it failed
            // Note: This function currently ONLY supports the case where C == D and Cdesc == Ddesc.
            // checkCublasStatus(cublasLtMatmul(
            //     ltHandle, operationDesc_layer1, &alpha, d_w_layer1, weightsDesc_layer1,
            //     d_input_feature, inputDesc_layer1, &beta, 
            //     d_r_layer1, outputDesc_layer1, d_r_layer1, outputDesc_layer1,
            //     &heuristicResult_layer1.algo,
            //     workspace_layer1, workspaceSize_layer1, stream));
            // checkCublasStatus(cublasLtMatmul(
            //     ltHandle, operationDesc_layer2, &alpha, d_w_layer2, weightsDesc_layer2,
            //     d_r_layer1, inputDesc_layer2, &beta, 
            //     d_r_layer2, outputDesc_layer2, d_r_layer2, outputDesc_layer2,
            //     &heuristicResult_layer2.algo,
            //     workspace_layer2, workspaceSize_layer2, stream));
            // checkCublasStatus(cublasLtMatmul(
            //     ltHandle, operationDesc_layer3, &alpha, d_w_layer3, weightsDesc_layer3,
            //     d_r_layer2, inputDesc_layer3, &beta, 
            //     d_r_layer3, outputDesc_layer3, d_r_layer3, outputDesc_layer3,
            //     &heuristicResult_layer3.algo,
            //     workspace_layer3, workspaceSize_layer3, stream));
            // checkCublasStatus(cublasLtMatmul(
            //     ltHandle, operationDesc_layer_out, &alpha, d_w_layer_out, weightsDesc_layer_out,
            //     d_r_layer3, inputDesc_layer_out, &beta, 
            //     d_r_layer_out, outputDesc_layer_out, d_r_layer_out, outputDesc_layer_out,
            //     &heuristicResult_layer_out.algo,
            //     workspace_layer_out, workspaceSize_layer_out, stream));

            // /////////////// output D2H /////////////// 
            // checkCudaStatus(cudaMemcpyAsync(r_layer_out, d_r_layer_out, 
            //     ele_r_layer_out * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
    }
    // print result of the first stream
    const int first_n_print = 5;
    for (int j = 0; j < first_n_print; j++) {
        printf("%f ", r_layer_out[j]);
    }
}

int main(int argc, char *argv[]) {

    ////////// Device capability check //////////
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        // https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__DEVICE_g5aa4f47938af8276f08074d09b7d520c.html
        struct cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("concurrentKernels: %d\n", deviceProp.concurrentKernels);
        printf("= 1: Concurrent Kernel Execution\n");
        printf("asyncEngineCount: %d\n", deviceProp.asyncEngineCount);
        printf("> 0: Overlap of Data Transfer and Kernel Execution\n");
        printf("= 2: Concurrent Data Transfers\n");
        printf("Device %d has compute capability %d.%d.\n",
            device, deviceProp.major, deviceProp.minor);
    }

    ////////////////////  Shared Memory Init  ////////////////////
 
    float* data_0 = malloc(SHM_DATA_SIZE); 
    float* data_1 = malloc(SHM_DATA_SIZE); 
    float* data_2 = malloc(SHM_DATA_SIZE); 
    float* data_3 = malloc(SHM_DATA_SIZE); 

    int* control_0 = malloc(4 * sizeof(int));
    int* control_1 = malloc(4 * sizeof(int));
    int* control_2 = malloc(4 * sizeof(int));
    int* control_3 = malloc(4 * sizeof(int));

    // initialize data block, allocate consecutive memory
    memset(data_0, 0, SHM_DATA_SIZE);
    memset(data_1, 0, SHM_DATA_SIZE);
    memset(data_2, 0, SHM_DATA_SIZE);
    memset(data_3, 0, SHM_DATA_SIZE);

    // initialize control signal
    control_0[0] = -1; // first value: block id of writer (client)
    control_0[1] = -1; // second value: iteration id of writer
    control_0[2] = -1; // third value: block id of reader (CUDA)
    control_0[3] = -1; // fourth value: iteration id of reader
    control_1[0] = -1; // first value: block id of writer (client)
    control_1[1] = -1; // second value: iteration id of writer
    control_1[2] = -1; // third value: block id of reader (CUDA)
    control_1[3] = -1; // fourth value: iteration id of reader
    control_2[0] = -1; // first value: block id of writer (client)
    control_2[1] = -1; // second value: iteration id of writer
    control_2[2] = -1; // third value: block id of reader (CUDA)
    control_2[3] = -1; // fourth value: iteration id of reader
    control_3[0] = -1; // first value: block id of writer (client)
    control_3[1] = -1; // second value: iteration id of writer
    control_3[2] = -1; // third value: block id of reader (CUDA)
    control_3[3] = -1; // fourth value: iteration id of reader
    
    pthread_t thread_id_0_store; 
    pthread_t thread_id_1_store; 
    pthread_t thread_id_2_store; 
    pthread_t thread_id_3_store; 

    pthread_t thread_id_0_consume; 
    pthread_t thread_id_1_consume; 
    pthread_t thread_id_2_consume; 
    pthread_t thread_id_3_consume; 
    printf("Before Thread\n"); 

    struct Receiver_thread_info t_receiver_info_0;
    struct Receiver_thread_info t_receiver_info_1;
    struct Receiver_thread_info t_receiver_info_2;
    struct Receiver_thread_info t_receiver_info_3;

    t_receiver_info_0.port = PORT + 0;
    t_receiver_info_1.port = PORT + 1;
    t_receiver_info_2.port = PORT + 2;
    t_receiver_info_3.port = PORT + 3;

    t_receiver_info_0.data = data_0;
    t_receiver_info_1.data = data_1;
    t_receiver_info_2.data = data_2;
    t_receiver_info_3.data = data_3;
    
    t_receiver_info_0.control = control_0;
    t_receiver_info_1.control = control_1;
    t_receiver_info_2.control = control_2;
    t_receiver_info_3.control = control_3;

    struct CUDA_thread_info t_cuda_info_0;
    struct CUDA_thread_info t_cuda_info_1;
    struct CUDA_thread_info t_cuda_info_2;
    struct CUDA_thread_info t_cuda_info_3;

    t_cuda_info_0.data = data_0;
    t_cuda_info_1.data = data_1;
    t_cuda_info_2.data = data_2;
    t_cuda_info_3.data = data_3;
    
    t_cuda_info_0.control = control_0;
    t_cuda_info_1.control = control_1;
    t_cuda_info_2.control = control_2;
    t_cuda_info_3.control = control_3;

    cublasLtHandle_t ltHandle;
    checkCublasStatus(cublasLtCreate(&ltHandle));
    t_cuda_info_0.ltHandlePointer = &ltHandle;
    t_cuda_info_1.ltHandlePointer = &ltHandle;
    t_cuda_info_2.ltHandlePointer = &ltHandle;
    t_cuda_info_3.ltHandlePointer = &ltHandle;

    pthread_create(&thread_id_0_consume, NULL, thread_consume, (void*) &t_cuda_info_0); 
    // pthread_create(&thread_id_1_consume, NULL, thread_consume, (void*) &t_cuda_info_1); 
    // pthread_create(&thread_id_2_consume, NULL, thread_consume, (void*) &t_cuda_info_2); 
    // pthread_create(&thread_id_3_consume, NULL, thread_consume, (void*) &t_cuda_info_3); 

    sleep(1); // wait CUDA init

    pthread_create(&thread_id_0_store, NULL, thread_receive_packets, (void*) &t_receiver_info_0);
    // pthread_create(&thread_id_1_store, NULL, thread_receive_packets, (void*) &t_receiver_info_1);  
    // pthread_create(&thread_id_2_store, NULL, thread_receive_packets, (void*) &t_receiver_info_2); 
    // pthread_create(&thread_id_3_store, NULL, thread_receive_packets, (void*) &t_receiver_info_3); 


    // pthread_create(&thread_id, NULL, thread_receive_packets, NULL); 
    pthread_join(thread_id_0_store, NULL); 
    // pthread_join(thread_id_1_store, NULL); 
    // pthread_join(thread_id_2_store, NULL); 
    // pthread_join(thread_id_3_store, NULL); 

    pthread_join(thread_id_0_consume, NULL); 
    // pthread_join(thread_id_1_consume, NULL); 
    // pthread_join(thread_id_2_consume, NULL); 
    // pthread_join(thread_id_3_consume, NULL); 

    // memcpy time from (after receiving the batch) to (copy the batch to CUDA)
    timespec memcpy_timespec[TOTAL_BATCH_NUM];
    long memcpy_time_ns[TOTAL_BATCH_NUM];
    float total_memcpy_ns_fl = 0;
    float average_memcpy_time = 0;
    for (int i = 0; i < TOTAL_BATCH_NUM; i++) {
        memcpy_timespec[i] = diff(network_time[i], cuda_time[i]);
        memcpy_time_ns[i] = memcpy_timespec[i].tv_sec * 1000 * 1000 * 1000 + memcpy_timespec[i].tv_nsec;
        total_memcpy_ns_fl += (float) memcpy_time_ns[i];
        printf("i = %d memcpy time = %ld ns\n", i, memcpy_time_ns[i]);
    }
    average_memcpy_time = (float) total_memcpy_ns_fl / TOTAL_BATCH_NUM;
    printf("\nAverage memcpt time per batch: %f sec = %f ms = %f us = %f ns\n",
        average_memcpy_time / ( 1000 * 1000 * 1000), average_memcpy_time / (1000 * 1000), 
        average_memcpy_time / 1000, average_memcpy_time);

    printf("After Thread\n"); 
    
    return 0;
}