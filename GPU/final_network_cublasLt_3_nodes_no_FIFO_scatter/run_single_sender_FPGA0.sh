# run this script after server
rm FPGA0_multiple_connections_network_client_sender 
gcc FPGA0_multiple_connections_network_client_sender.c -lpthread -o FPGA0_multiple_connections_network_client_sender
./FPGA0_multiple_connections_network_client_sender &