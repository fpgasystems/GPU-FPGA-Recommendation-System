
#include "ap_axi_sdata.h"
#include <ap_fixed.h>
#include "ap_int.h" 
#include "in_casting_bench.hpp"
#include "hls_stream.h"



#define DWIDTH 32

typedef ap_axiu<DWIDTH, 0, 0, 0> pkt;

#define DWIDTH512 512
#define DWIDTH256 256
#define DWIDTH128 128
#define DWIDTH64 64
#define DWIDTH32 32
#define DWIDTH16 16
#define DWIDTH8 8

typedef ap_axiu<DWIDTH512, 0, 0, 0> pkt512;
typedef ap_axiu<DWIDTH256, 0, 0, 0> pkt256;
typedef ap_axiu<DWIDTH128, 0, 0, 0> pkt128;
typedef ap_axiu<DWIDTH64, 0, 0, 0> pkt64;
typedef ap_axiu<DWIDTH32, 0, 0, 0> pkt32;
typedef ap_axiu<DWIDTH16, 0, 0, 0> pkt16;
typedef ap_axiu<DWIDTH8, 0, 0, 0> pkt8;


void openConnections(int useConn, int baseIpAddress, int basePort, hls::stream<pkt64>& m_axis_tcp_open_connection, hls::stream<pkt32>& s_axis_tcp_open_status, ap_uint<16>* sessionID)
{
#pragma HLS dataflow

     int numOpenedCon = 0;
     pkt64 openConnection_pkt;
     for (int i = 0; i < useConn; ++i)
     {
     #pragma HLS PIPELINE II=1
          openConnection_pkt.data(31,0) = baseIpAddress;
          openConnection_pkt.data(47,32) = basePort;
          m_axis_tcp_open_connection.write(openConnection_pkt);
     }

     openStatus status;
     for (int i = 0; i < useConn; ++i)
     {
     #pragma HLS PIPELINE II=1
          pkt32 open_status_pkt = s_axis_tcp_open_status.read();
          status.sessionID = open_status_pkt.data(15,0);
          status.success = open_status_pkt.data(16,16);

          if (status.success)
          {
               sessionID[numOpenedCon] = status.sessionID;
               numOpenedCon++;
               std::cout << "Connection successfully opened." << std::endl;
          }
     }
}

void sendData(hls::stream<pkt32>& m_axis_tcp_tx_meta, 
               hls::stream<pkt512>& m_axis_tcp_tx_data, 
               hls::stream<pkt64>& s_axis_tcp_tx_status,
               ap_uint<16>* sessionID,
               int useConn,
               int expectedTxPkgCnt, 
               int pkgWordCount
                )
{
     bool first_round = true;
     int sentPkgCnt = 0;

     do{
          pkt32 tx_meta_pkt;
          appTxRsp resp;

          if (first_round)
          {
               
               for (int i = 0; i < useConn; ++i)
               {
               #pragma HLS PIPELINE II=1
                    tx_meta_pkt.data(15,0) = sessionID[i];
                    tx_meta_pkt.data(31,16) = pkgWordCount*(512/8);
                    m_axis_tcp_tx_meta.write(tx_meta_pkt);
               }
               first_round = false;
          }
          else
          {
               if (!s_axis_tcp_tx_status.empty())
               {
                    pkt64 txStatus_pkt = s_axis_tcp_tx_status.read();
                    resp.sessionID = txStatus_pkt.data(15,0);
                    resp.length = txStatus_pkt.data(31,16);
                    resp.remaining_space = txStatus_pkt.data(61,32);
                    resp.error = txStatus_pkt.data(63,62);

                    if (resp.error == 0)
                    {
                         if (sentPkgCnt < expectedTxPkgCnt-1)
                         {
                              tx_meta_pkt.data(15,0) = resp.sessionID;
                              tx_meta_pkt.data(31,16) = pkgWordCount*(512/8);
                              m_axis_tcp_tx_meta.write(tx_meta_pkt);
                         }
                         
                         for (int j = 0; j < pkgWordCount; ++j)
                         {
                         #pragma HLS PIPELINE II=1
                              pkt512 currWord;
                              for (int i = 0; i < (512/64); i++) 
                              {
                                   #pragma HLS UNROLL
                                   currWord.data(i*64+63, i*64) = 0xdeadbeefdeadbeef;
                                   currWord.keep(i*8+7, i*8) = 0xff;
                              }
                              currWord.last = (j == pkgWordCount-1);
                              m_axis_tcp_tx_data.write(currWord);
                         }
                         sentPkgCnt++;

                    }
                    else
                    {
                         //Check if connection  was torn down
                         if (resp.error == 1)
                         {
                              std::cout << "Connection was torn down. " << resp.sessionID << std::endl;
                         }
                         else
                         {
                              tx_meta_pkt.data(15,0) = resp.sessionID;
                              tx_meta_pkt.data(31,16) = pkgWordCount*(512/8);
                              m_axis_tcp_tx_meta.write(tx_meta_pkt);
                         }
                    }
               }
          }
          
     }
     while(sentPkgCnt<expectedTxPkgCnt);
}

extern "C" {
void hls_test_krnl(
               // Internal Stream
               hls::stream<pkt512>& s_axis_udp_rx, 
               hls::stream<pkt512>& m_axis_udp_tx, 
               hls::stream<pkt256>& s_axis_udp_rx_meta, 
               hls::stream<pkt256>& m_axis_udp_tx_meta, 
               
               hls::stream<pkt16>& m_axis_tcp_listen_port, 
               hls::stream<pkt8>& s_axis_tcp_port_status, 
               hls::stream<pkt64>& m_axis_tcp_open_connection, 
               hls::stream<pkt32>& s_axis_tcp_open_status, 
               hls::stream<pkt16>& m_axis_tcp_close_connection, 
               hls::stream<pkt128>& s_axis_tcp_notification, 
               hls::stream<pkt32>& m_axis_tcp_read_pkg, 
               hls::stream<pkt16>& s_axis_tcp_rx_meta, 
               hls::stream<pkt512>& s_axis_tcp_rx_data, 
               hls::stream<pkt32>& m_axis_tcp_tx_meta, 
               hls::stream<pkt512>& m_axis_tcp_tx_data, 
               hls::stream<pkt64>& s_axis_tcp_tx_status,
               int useConn, 
               int useIpAddr, 
               int pkgWordCount, 
               int basePort, 
               int usePort, 
               int expectedTxPkgCnt, 
               int baseIpAddress, 
               int expectedRespInKBTotal     
                      ) {

#pragma HLS INTERFACE axis port = s_axis_udp_rx
#pragma HLS INTERFACE axis port = m_axis_udp_tx
#pragma HLS INTERFACE axis port = s_axis_udp_rx_meta
#pragma HLS INTERFACE axis port = m_axis_udp_tx_meta
#pragma HLS INTERFACE axis port = m_axis_tcp_listen_port
#pragma HLS INTERFACE axis port = s_axis_tcp_port_status
#pragma HLS INTERFACE axis port = m_axis_tcp_open_connection
#pragma HLS INTERFACE axis port = s_axis_tcp_open_status
#pragma HLS INTERFACE axis port = m_axis_tcp_close_connection
#pragma HLS INTERFACE axis port = s_axis_tcp_notification
#pragma HLS INTERFACE axis port = m_axis_tcp_read_pkg
#pragma HLS INTERFACE axis port = s_axis_tcp_rx_meta
#pragma HLS INTERFACE axis port = s_axis_tcp_rx_data
#pragma HLS INTERFACE axis port = m_axis_tcp_tx_meta
#pragma HLS INTERFACE axis port = m_axis_tcp_tx_data
#pragma HLS INTERFACE axis port = s_axis_tcp_tx_status
#pragma HLS INTERFACE s_axilite port=useConn bundle = control
#pragma HLS INTERFACE s_axilite port=useIpAddr bundle = control
#pragma HLS INTERFACE s_axilite port=pkgWordCount bundle = control
#pragma HLS INTERFACE s_axilite port=basePort bundle = control
#pragma HLS INTERFACE s_axilite port=usePort bundle = control
#pragma HLS INTERFACE s_axilite port=expectedTxPkgCnt bundle = control
#pragma HLS INTERFACE s_axilite port=baseIpAddress bundle=control
#pragma HLS INTERFACE s_axilite port=expectedRespInKBTotal bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#pragma HLS dataflow

hls::stream<ap_uint<16> > listenPort;
hls::stream<bool> listenPortStatus;
hls::stream<appNotification> notifications;
hls::stream<appReadRequest> readRequest;
hls::stream<ap_uint<16> > rxMetaData;
hls::stream<net_axis<DATA_WIDTH> > rxData;
hls::stream<ap_uint<16> > closeConnection;
hls::stream<ipTuple> openConnection;
hls::stream<openStatus> openConStatus;
hls::stream<appTxMeta> txMetaData;
hls::stream<net_axis<DATA_WIDTH> > txData;
hls::stream<appTxRsp> txStatus;

#pragma HLS stream variable=openConStatus depth=128
#pragma HLS stream variable=txStatus depth=128
#pragma HLS stream variable=openConnection depth=128
#pragma HLS stream variable=txMetaData depth=128
#pragma HLS stream variable=txData depth=128


          ap_uint<16> sessionID [128];
          
          openConnections( useConn,  baseIpAddress,  basePort, m_axis_tcp_open_connection, s_axis_tcp_open_status, sessionID);

          sendData( m_axis_tcp_tx_meta, m_axis_tcp_tx_data, s_axis_tcp_tx_status,sessionID,useConn, expectedTxPkgCnt, pkgWordCount);
          





          if (!s_axis_udp_rx.empty())
          {
               pkt512 udp_rx = s_axis_udp_rx.read();
               m_axis_udp_tx.write(udp_rx);
          }

          if (!s_axis_udp_rx_meta.empty())
          {
               pkt256 udp_rx_meta = s_axis_udp_rx_meta.read();
               m_axis_udp_tx_meta.write(udp_rx_meta);
          }

          pkt16 listenPort_pkt;
          if (!listenPort.empty())
          {
               ap_uint<16> listenPort_data = listenPort.read();
               listenPort_pkt.data = listenPort_data;
               m_axis_tcp_listen_port.write(listenPort_pkt);
          }

          if (!s_axis_tcp_port_status.empty())
          {
               pkt8 port_status_pkt = s_axis_tcp_port_status.read();
               bool port_status_data = (port_status_pkt.data == 1) ? true: false;
               listenPortStatus.write(port_status_data);
          }

          // pkt64 openConnection_pkt;
          // if (!openConnection.empty())
          // {
          //      ipTuple openConnection_data = openConnection.read();
          //      openConnection_pkt.data(31,0) = openConnection_data.ip_address;
          //      openConnection_pkt.data(47,32) = openConnection_data.ip_port;
          //      m_axis_tcp_open_connection.write(openConnection_pkt);
          // }

          // openStatus open_status_data;
          // if (!s_axis_tcp_open_status.empty())
          // {
          //      pkt32 open_status_pkt = s_axis_tcp_open_status.read();
          //      open_status_data.sessionID = open_status_pkt.data(15,0);
          //      open_status_data.success = open_status_pkt.data(16,16);
          //      openConStatus.write(open_status_data);
          // }

          pkt16 close_connection_pkt;
          if (!closeConnection.empty())
          {
               close_connection_pkt.data = closeConnection.read();
               m_axis_tcp_close_connection.write(close_connection_pkt);
          }

          appNotification tcp_notification_data;
          if (!s_axis_tcp_notification.empty())
          {
               pkt128 tcp_notification_pkt = s_axis_tcp_notification.read();
               tcp_notification_data.sessionID = tcp_notification_pkt.data(15,0);
               tcp_notification_data.length = tcp_notification_pkt.data(31,16);
               tcp_notification_data.ipAddress = tcp_notification_pkt.data(63,32);
               tcp_notification_data.dstPort = tcp_notification_pkt.data(79,64);
               tcp_notification_data.closed = tcp_notification_pkt.data(80,80);
               notifications.write(tcp_notification_data);

          }

          pkt32 readRequest_pkt;
          if (!readRequest.empty())
          {
               appReadRequest readRequest_data = readRequest.read();
               readRequest_pkt.data(15,0) = readRequest_data.sessionID;
               readRequest_pkt.data(31,16) = readRequest_data.length;
               m_axis_tcp_read_pkg.write(readRequest_pkt);
          }

          if (!s_axis_tcp_rx_meta.empty())
          {
               pkt16 tcp_rx_meta_pkt = s_axis_tcp_rx_meta.read();
               ap_uint<16> tcp_rx_meta_data = tcp_rx_meta_pkt.data;
               rxMetaData.write(tcp_rx_meta_data);
          }

          net_axis<512> tcp_rx_data;
          if (!s_axis_tcp_rx_data.empty())
          {
               pkt512 tcp_rx_pkt = s_axis_tcp_rx_data.read();
               tcp_rx_data.data = tcp_rx_pkt.data;
               tcp_rx_data.keep = tcp_rx_pkt.keep;
               tcp_rx_data.last = tcp_rx_pkt.last;
               rxData.write(tcp_rx_data);
          }

          // pkt32 tx_meta_pkt;
          // if (!txMetaData.empty())
          // {
          //      appTxMeta tx_meta_data = txMetaData.read();
          //      tx_meta_pkt.data(15,0) = tx_meta_data.sessionID;
          //      tx_meta_pkt.data(31,16) = tx_meta_data.length;
          //      m_axis_tcp_tx_meta.write(tx_meta_pkt);
          // }

          // pkt512 txData_pkt;
          // if (!txData.empty())
          // {
          //      net_axis<512> tx_data = txData.read();
          //      txData_pkt.data = tx_data.data;
          //      txData_pkt.keep = tx_data.keep;
          //      txData_pkt.last = tx_data.last;
          //      m_axis_tcp_tx_data.write(txData_pkt);
          // }

          // appTxRsp tx_status_data;
          // if (!s_axis_tcp_tx_status.empty())
          // {
          //      pkt64 txStatus_pkt = s_axis_tcp_tx_status.read();
          //      tx_status_data.sessionID = txStatus_pkt.data(15,0);
          //      tx_status_data.length = txStatus_pkt.data(31,16);
          //      tx_status_data.remaining_space = txStatus_pkt.data(61,32);
          //      tx_status_data.error = txStatus_pkt.data(63,62);
          //      txStatus.write(tx_status_data);
          // }



     }
}