# Run this script before client
rm simple_2_thread_latency
gcc -lpthread simple_2_thread_latency.c -o simple_2_thread_latency
./simple_2_thread_latency
