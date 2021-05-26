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
};

// A normal C function that is executed as a thread  
void *thread_receive_packets(void* vargp) 
{
    sleep(2); // wait CUDA init
    struct Receiver_thread_info* t_info = (struct Receiver_thread_info*) vargp;
    printf("Printing Port from Thread %d\n", t_info -> port); 

    int* writer_block_id = t_info -> control + 0;
    int* writer_iteration = t_info -> control + 1;
    int* reader_block_id = t_info -> control + 2;
    int* reader_iteration = t_info -> control + 3;

    ////////////////////  Network Init  ////////////////////

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


            // network_time[iter * BATCH_NUM_PER_LOOP + block_id] = clock();
            int block_addr = block_id * BLOCK_ENTRY_NUM;
            memset(t_info -> data + block_addr, 0, BLOCK_SIZE);
	        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &network_time[iter * BATCH_NUM_PER_LOOP + block_id]);

            *writer_block_id = block_id;
            // usleep(2000);
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
    float *input_feature = (float*) aligned_alloc(BLOCK_SIZE, BLOCK_SIZE);
    // float *input_feature = (float*) malloc(BLOCK_SIZE);

    int* writer_block_id = t_info -> control + 0;
    int* writer_iteration = t_info -> control + 1;
    int* reader_block_id = t_info -> control + 2;
    int* reader_iteration = t_info -> control + 3;

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
	        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cuda_time[iter * BATCH_NUM_PER_LOOP + block_id]);
            // int block_addr = block_id * BLOCK_ENTRY_NUM;
            // memcpy(input_feature, t_info -> data + block_addr, BLOCK_SIZE);
            *reader_block_id = block_id;


            /////////////// input H2D /////////////// 
            // checkCudaStatus(cudaMemcpyAsync(d_input_feature, input_feature, 
            //     ele_input_feature * sizeof(float), cudaMemcpyHostToDevice, stream));
            // cuda_time[iter * BATCH_NUM_PER_LOOP + block_id] = clock();

            /////////////// Computation /////////////// 
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
    // const int first_n_print = 5;
    // for (int j = 0; j < first_n_print; j++) {
    //     printf("%f ", r_layer_out[j]);
    // }
}

int main(int argc, char *argv[]) {

    ////////////////////  Shared Memory Init  ////////////////////
 
    float* data_0 = aligned_alloc(SHM_DATA_SIZE, SHM_DATA_SIZE); 
    // float* data_0 = malloc(SHM_DATA_SIZE); 
    float* data_1 = malloc(SHM_DATA_SIZE); 
    float* data_2 = malloc(SHM_DATA_SIZE); 
    float* data_3 = malloc(SHM_DATA_SIZE); 

    int* control_0 = aligned_alloc(4 * sizeof(int), 4 * sizeof(int));
    // int* control_0 = malloc(4 * sizeof(int));
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