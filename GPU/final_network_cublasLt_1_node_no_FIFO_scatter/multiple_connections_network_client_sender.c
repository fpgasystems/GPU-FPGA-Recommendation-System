// Client side C/C++ program to demonstrate Socket programming 
#include <stdio.h> 
#include <stdlib.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <unistd.h> 
#include <string.h> 
#include <unistd.h>
#include <time.h>
#include <pthread.h> 
// #include <mutex.h>

#include "constant.h"

int global_batch_count = 0;
pthread_mutex_t mtx;//=PTHREAD_MUTEX_INITIALIZER;

struct Thread_info {
    int port;
};

// A normal C function that is executed as a thread  
void *thread_send_packets(void* vargp) 
{ 
    struct Thread_info* t_info = (struct Thread_info*) vargp;
    printf("Printing Port from Thread %d\n", t_info -> port); 
    

    int sock = 0, valread; 
    struct sockaddr_in serv_addr; 

    float* array_buf = malloc(sizeof(float) * BLOCK_ENTRY_NUM);
    for (int i = 0; i < BLOCK_ENTRY_NUM; i++) {
        array_buf[i] = 1;
    }

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    { 
        printf("\n Socket creation error \n"); 
        return -1; 
    } 
   
    serv_addr.sin_family = AF_INET; 
    serv_addr.sin_port = htons(t_info -> port); 
       
    // Convert IPv4 and IPv6 addresses from text to binary form 
    // if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0)  
    // if(inet_pton(AF_INET, "10.128.0.11", &serv_addr.sin_addr)<=0)  
    if(inet_pton(AF_INET, "10.1.212.25", &serv_addr.sin_addr)<=0)  // d5005
    { 
        printf("\nInvalid address/ Address not supported \n"); 
        return -1; 
    } 
   
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr))<0) 
    { 
        printf("\nConnection Failed \n"); 
        return -1; 
    } 

    printf("Start sending data.\n");
    ////////////////   Data transfer   ////////////////
    int i = 0;
    float total_sent_bytes = 0.0;

    clock_t start = clock();


    while (1) {

        pthread_mutex_lock(&mtx);
        if (global_batch_count == (TOTAL_BATCH_NUM * 2)) {
            pthread_mutex_unlock(&mtx);  
            break;
        }
        else {
            int local_batch_count = global_batch_count;
            global_batch_count++;
            pthread_mutex_unlock(&mtx);  

// #define DEBUG
#ifdef DEBUG
    printf("batch_count: %d\n", local_batch_count);
#endif    
            // g_i_mutex is automatically released when lock
            // goes out of scope


            int total_sent_bytes = 0;

            while (total_sent_bytes < BLOCK_SIZE) {
                int sent_bytes = send(sock, array_buf + total_sent_bytes, BLOCK_SIZE - total_sent_bytes, 0);
                total_sent_bytes += sent_bytes;
                if (sent_bytes == -1) {
                    printf("Sending data UNSUCCESSFUL!\n");
                    return -1;
                }
            }

            if (total_sent_bytes != BLOCK_SIZE) {
                printf("Sending error, sending more bytes than a block\n");
            }
        }
    }

    clock_t end = clock();

    // Should wait until the server said all the data was sent correctly,
    // otherwise the sender may send packets yet the server did not receive.
    char msg[32];
    int recv_bytes = read(sock, msg, 32);
    printf("received from server: %s\n", msg);

    // float total_size = (float)LOOP_NUM * BATCH_NUM_PER_LOOP * BLOCK_SIZE;
    // printf("Data sent. Packet number:%d\tPacket size:%d bytes\tTotal data:%fGB\n",
    //     LOOP_NUM * BATCH_NUM_PER_LOOP, BLOCK_SIZE, total_size / (1024 * 1024 * 1024));   
    // float elapsed_time = (end-start) / (float)CLOCKS_PER_SEC;
    // printf("\nConsumed time: %f seconds\n", elapsed_time);
    // printf("Transfer Throughput: %f GB / sec\n", total_size / elapsed_time / 1024 / 1024 / 1024); 

    return NULL; 
} 

int main(int argc, char const *argv[]) 
{ 

    pthread_t thread_id[THREAD_NUM]; 
    printf("Before Thread\n"); 

    struct Thread_info t_info[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++) {
        t_info[i].port = PORT + i; 
    }

    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_create(&thread_id[i], NULL, thread_send_packets, (void*) &t_info[i]); 
    }

    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_join(thread_id[i], NULL); 
    }
    printf("After Thread\n"); 

    return 0; 
} 
