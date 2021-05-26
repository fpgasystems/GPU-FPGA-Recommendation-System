# Programs

First start Terminal 1: run cuda_server first (./run_cuda_server.sh)

Then start Terminal 2: run client to send data (./run_client_sender.sh) -> Remember to Adjust the Server IP Address

Correct Results:

Input Feature Size = 512 -> 68719476736

Input Feature Size = 1024 -> 137438953472

## cuda_server.c

rm cuda_server

nvcc -l cublasLt -lpthread cuda_server.c -o cuda_server

nvprof -f --export-profile timeline.prof --concurrent-kernels on ./cuda_server

## multiple_connections_network_client_sender.c

### NOTE! Sender must send more data than receiver side, because on both sender and receiver threads have different progress, e.g., receiver connection 1 is waiting for the last batch, yet sender is trying to send the data through connection 2.

As a result, we use more sender data (2 * required)

This program simulates FPGA that opens 4 connections and sending data to the CUDA server.

Start CUDA server first, then client.

gcc multiple_connections_network_client_sender.c -lpthread -o multiple_connections_network_client_sender

./multiple_connections_network_client_sender


# Other programs (for building the final version)

## pthread_test.c

https://www.geeksforgeeks.org/multithreading-c-2/

Pass port info and memory address space to the thread as a structure, and execute that thread.

gcc pthread_test.c -lpthread

./a.out

## single_connection_network_server_receiver.c

Start server first, then client.

gcc single_connection_network_server_receiver.c -lpthread -o single_connection_network_server_receiver

./single_connection_network_server_receiver

## single_connection_network_client_sender.c

Start server first, then client.

gcc single_connection_network_client_sender.c -lpthread -o single_connection_network_client_sender

./single_connection_network_client_sender


## multiple_connections_network_server_receiver.c

Start server first, then client.

4 TCP connections.

gcc multiple_connections_network_server_receiver.c -lpthread -o multiple_connections_network_server_receiver

./multiple_connections_network_server_receiver

## multiple_connections_network_client_sender.c

Start server first, then client.

gcc multiple_connections_network_client_sender.c -lpthread -o multiple_connections_network_client_sender

./multiple_connections_network_client_sender