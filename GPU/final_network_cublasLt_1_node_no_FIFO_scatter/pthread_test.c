#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h>  //Header file for sleep(). man 3 sleep for details. 
#include <pthread.h> 

struct Thread_info {
    int port;
    float* buffer;
};

// A normal C function that is executed as a thread  
void *myThreadFun(void* vargp) 
{ 
    sleep(1); 
    struct Thread_info* t_info = (struct Thread_info*) vargp;
    printf("Printing GeeksQuiz from Thread %d\n", t_info -> port); 
    for (int i = 0; i < 32; i++) {
        printf("%f\t", t_info -> buffer[i]);
    }
    return NULL; 
} 

int main() 
{ 
    pthread_t thread_id; 
    printf("Before Thread\n"); 

    int port = 8080;
    float* buffer = malloc(128);
    for (int i = 0; i < 32; i++) {
        buffer[i] = i;
    }

    struct Thread_info t_info_0;
    t_info_0.port = port;
    t_info_0.buffer = buffer;

    pthread_create(&thread_id, NULL, myThreadFun, (void*) &t_info_0); 
    // pthread_create(&thread_id, NULL, myThreadFun, NULL); 
    pthread_join(thread_id, NULL); 
    printf("After Thread\n"); 
    exit(0); 
}