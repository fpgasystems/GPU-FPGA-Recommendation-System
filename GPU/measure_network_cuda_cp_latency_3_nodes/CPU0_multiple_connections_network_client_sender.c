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

#include "constant.h"


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

    float array_buf[BLOCK_ENTRY_NUM_CPU_SENDER];
    for (int i = 0; i < BLOCK_ENTRY_NUM_CPU_SENDER; i++) {
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

    for (int i = 0; i < LOOP_NUM * BATCH_NUM_PER_LOOP; i++) {

        int total_sent_bytes = 0;

        while (total_sent_bytes < BLOCK_SIZE_CPU_SENDER) {
            int sent_bytes = send(sock, array_buf + total_sent_bytes, BLOCK_SIZE_CPU_SENDER - total_sent_bytes, 0);
            total_sent_bytes += sent_bytes;
            if (sent_bytes == -1) {
                printf("Sending data UNSUCCESSFUL!\n");
                return -1;
            }
        }

        if (total_sent_bytes != BLOCK_SIZE_CPU_SENDER) {
            printf("Sending error, sending more bytes than a block\n");
        }
    }

    clock_t end = clock();

    // Should wait until the server said all the data was sent correctly,
    // otherwise the sender may send packets yet the server did not receive.
    char msg[32];
    int recv_bytes = read(sock, msg, 32);
    printf("received from server: %s\n", msg);

    float total_size = (float)LOOP_NUM * BATCH_NUM_PER_LOOP * BLOCK_SIZE_CPU_SENDER;
    printf("Data sent. Packet number:%d\tPacket size:%d bytes\tTotal data:%fGB\n",
        LOOP_NUM * BATCH_NUM_PER_LOOP, BLOCK_SIZE_CPU_SENDER, total_size / (1024 * 1024 * 1024));   
    float elapsed_time = (end-start) / (float)CLOCKS_PER_SEC;
    printf("\nConsumed time: %f seconds\n", elapsed_time);
    printf("Transfer Throughput: %f GB / sec\n", total_size / elapsed_time / 1024 / 1024 / 1024); 

    return NULL; 
} 

int main(int argc, char const *argv[]) 
{ 

    pthread_t thread_id_0; 
    pthread_t thread_id_1; 
    pthread_t thread_id_2; 
    pthread_t thread_id_3; 
    printf("Before Thread\n"); 

    struct Thread_info t_info_0;
    struct Thread_info t_info_1;
    struct Thread_info t_info_2;
    struct Thread_info t_info_3;
    t_info_0.port = PORT_CPU_SENDER_0 + 0;
    t_info_1.port = PORT_CPU_SENDER_0 + 1;
    t_info_2.port = PORT_CPU_SENDER_0 + 2;
    t_info_3.port = PORT_CPU_SENDER_0 + 3;

    pthread_create(&thread_id_0, NULL, thread_send_packets, (void*) &t_info_0); 
    // pthread_create(&thread_id_1, NULL, thread_send_packets, (void*) &t_info_1); 
    // pthread_create(&thread_id_2, NULL, thread_send_packets, (void*) &t_info_2); 
    // pthread_create(&thread_id_3, NULL, thread_send_packets, (void*) &t_info_3); 
    // pthread_create(&thread_id, NULL, thread_send_packets, NULL); 
    pthread_join(thread_id_0, NULL); 
    // pthread_join(thread_id_1, NULL); 
    // pthread_join(thread_id_2, NULL); 
    // pthread_join(thread_id_3, NULL); 
    printf("After Thread\n"); 

    return 0; 
} 
