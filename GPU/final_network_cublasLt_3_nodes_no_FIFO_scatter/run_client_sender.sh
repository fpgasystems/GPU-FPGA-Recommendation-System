# run this script after server
rm FPGA0_multiple_connections_network_client_sender FPGA1_multiple_connections_network_client_sender CPU0_multiple_connections_network_client_sender
gcc FPGA0_multiple_connections_network_client_sender.c -lpthread -o FPGA0_multiple_connections_network_client_sender
gcc FPGA1_multiple_connections_network_client_sender.c -lpthread -o FPGA1_multiple_connections_network_client_sender
gcc CPU0_multiple_connections_network_client_sender.c -lpthread -o CPU0_multiple_connections_network_client_sender
./CPU0_multiple_connections_network_client_sender &
sleep 10
./FPGA0_multiple_connections_network_client_sender &
sleep 10
./FPGA1_multiple_connections_network_client_sender &
