#include "embedding_98_krnl.hpp"

#include "ap_axi_sdata.h"
#include <ap_fixed.h>
#include "ap_int.h" 
#include "hls_stream.h"

#include "axi_utils.hpp"
#include "toe.hpp"



extern "C" {
void openConnections(int useConn, int baseIpAddress, int basePort, hls::stream<pkt64>& m_axis_tcp_open_connection, hls::stream<pkt32>& s_axis_tcp_open_status, ap_uint<16>* sessionID)
{
#pragma HLS dataflow

     int numOpenedCon = 0;
     pkt64 openConnection_pkt;
     for (int i = 0; i < useConn; ++i)
     {
     #pragma HLS PIPELINE II=1
          openConnection_pkt.data(31,0) = baseIpAddress;
          openConnection_pkt.data(47,32) = basePort+i;
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
               hls::stream<ap_uint<512> >& s_data_in,
               ap_uint<16>* sessionID,
               int useConn,
               ap_uint<64> expectedTxByteCnt, 
               int pkgWordCount
                )
{
     bool first_round = true;
     ap_uint<64> sentByteCnt = 0;
     ap_uint<64> currentPkgWordCnt = 0;
     int currentPkgSessionIndex = 0;

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
                         sentByteCnt = sentByteCnt + resp.length;

                         currentPkgSessionIndex++;
                         if (currentPkgSessionIndex == useConn)
                         {
                              currentPkgSessionIndex = 0;
                         }

                         if (sentByteCnt < expectedTxByteCnt)
                         {
                              tx_meta_pkt.data(15,0) = resp.sessionID; //sessionID[currentPkgSessionIndex];
                              if (sentByteCnt + pkgWordCount*64 < expectedTxByteCnt )
                              {
                                  tx_meta_pkt.data(31,16) = pkgWordCount*(512/8);
                                  currentPkgWordCnt = pkgWordCount;
                              }
                              else
                              {
                                  tx_meta_pkt.data(31,16) = expectedTxByteCnt - sentByteCnt;
                                  currentPkgWordCnt = (expectedTxByteCnt - sentByteCnt)>>6;
                              }
                              
                              m_axis_tcp_tx_meta.write(tx_meta_pkt);
                         }
                         
                         for (int j = 0; j < currentPkgWordCnt; ++j)
                         {
                         #pragma HLS PIPELINE II=1
                              ap_uint<512> s_data = s_data_in.read();
                              pkt512 currWord;
                              for (int i = 0; i < (512/64); i++) 
                              {
                                   #pragma HLS UNROLL
                                   currWord.data(i*64+63, i*64) = s_data(i*64+63, i*64);
                                   currWord.keep(i*8+7, i*8) = 0xff;
                              }
                              currWord.last = (j == currentPkgWordCnt-1);
                              m_axis_tcp_tx_data.write(currWord);
                         }
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
                              tx_meta_pkt.data(31,16) = resp.length;
                              m_axis_tcp_tx_meta.write(tx_meta_pkt);
                         }
                    }
               }
          }
          
     }
     while(sentByteCnt<expectedTxByteCnt);
}




void embedding_98_krnl(  
    const axi_t* table_HBM0, const axi_t* table_HBM1, 
    const axi_t* table_HBM2, const axi_t* table_HBM3, 
    const axi_t* table_HBM4, const axi_t* table_HBM5, 
    const axi_t* table_HBM6, const axi_t* table_HBM7, 
    const axi_t* table_HBM8, const axi_t* table_HBM9, 
    const axi_t* table_HBM10, const axi_t* table_HBM11, 
    const axi_t* table_HBM12, const axi_t* table_HBM13, 
    const axi_t* table_HBM14, const axi_t* table_HBM15, 
    const axi_t* table_HBM16, const axi_t* table_HBM17, 
    const axi_t* table_HBM18, const axi_t* table_HBM19, 
    const axi_t* table_HBM20, const axi_t* table_HBM21, 
    const axi_t* table_HBM22, const axi_t* table_HBM23, 
    const axi_t* table_HBM24, const axi_t* table_HBM25, 
    const axi_t* table_HBM26, const axi_t* table_HBM27, 
    const axi_t* table_DDR0, const axi_t* table_DDR1,

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
    int pkgWordCount, 
    int basePort, 
    int baseIpAddress,
    int batch_num   )
{
    
#pragma HLS DATAFLOW

#pragma HLS INTERFACE m_axi port=table_HBM0  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=table_HBM1  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=table_HBM2  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=table_HBM3  offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=table_HBM4  offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=table_HBM5  offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=table_HBM6  offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=table_HBM7  offset=slave bundle=gmem7
#pragma HLS INTERFACE m_axi port=table_HBM8  offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=table_HBM9  offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=table_HBM10  offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=table_HBM11  offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=table_HBM12  offset=slave bundle=gmem12
#pragma HLS INTERFACE m_axi port=table_HBM13  offset=slave bundle=gmem13
#pragma HLS INTERFACE m_axi port=table_HBM14  offset=slave bundle=gmem14
#pragma HLS INTERFACE m_axi port=table_HBM15  offset=slave bundle=gmem15
#pragma HLS INTERFACE m_axi port=table_HBM16  offset=slave bundle=gmem16
#pragma HLS INTERFACE m_axi port=table_HBM17  offset=slave bundle=gmem17
#pragma HLS INTERFACE m_axi port=table_HBM18  offset=slave bundle=gmem18
#pragma HLS INTERFACE m_axi port=table_HBM19  offset=slave bundle=gmem19
#pragma HLS INTERFACE m_axi port=table_HBM20  offset=slave bundle=gmem20
#pragma HLS INTERFACE m_axi port=table_HBM21  offset=slave bundle=gmem21
#pragma HLS INTERFACE m_axi port=table_HBM22  offset=slave bundle=gmem22
#pragma HLS INTERFACE m_axi port=table_HBM23  offset=slave bundle=gmem23
#pragma HLS INTERFACE m_axi port=table_HBM24  offset=slave bundle=gmem24
#pragma HLS INTERFACE m_axi port=table_HBM25  offset=slave bundle=gmem25
#pragma HLS INTERFACE m_axi port=table_HBM26  offset=slave bundle=gmem26
#pragma HLS INTERFACE m_axi port=table_HBM27  offset=slave bundle=gmem27

#pragma HLS INTERFACE m_axi port=table_DDR0  offset=slave bundle=gmem32
#pragma HLS INTERFACE m_axi port=table_DDR1  offset=slave bundle=gmem33

#pragma HLS INTERFACE s_axilite port=table_HBM0  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM1  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM2  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM3  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM4  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM5  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM6  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM7  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM8  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM9  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM10  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM11  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM12  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM13  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM14  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM15  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM16  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM17  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM18  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM19  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM20  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM21  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM22  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM23  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM24  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM25  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM26  bundle=control
#pragma HLS INTERFACE s_axilite port=table_HBM27  bundle=control

#pragma HLS INTERFACE s_axilite port=table_DDR0  bundle=control
#pragma HLS INTERFACE s_axilite port=table_DDR1  bundle=control

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
#pragma HLS INTERFACE s_axilite port=pkgWordCount bundle = control
#pragma HLS INTERFACE s_axilite port=basePort bundle = control
#pragma HLS INTERFACE s_axilite port=baseIpAddress bundle=control
#pragma HLS INTERFACE s_axilite port=batch_num bundle=control
  
#pragma HLS INTERFACE s_axilite port=return bundle=control
    
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



ap_uint<16> sessionID [16];
          
    openConnections( useConn,  baseIpAddress,  basePort, m_axis_tcp_open_connection, s_axis_tcp_open_status, sessionID);



    hls::stream<hbm_t> s_embedding_buffer_HBM0;
    hls::stream<hbm_t> s_embedding_buffer_HBM1;
    hls::stream<hbm_t> s_embedding_buffer_HBM2;
    hls::stream<hbm_t> s_embedding_buffer_HBM3;
    hls::stream<hbm_t> s_embedding_buffer_HBM4;
    hls::stream<hbm_t> s_embedding_buffer_HBM5;
    hls::stream<hbm_t> s_embedding_buffer_HBM6;
    hls::stream<hbm_t> s_embedding_buffer_HBM7;
    hls::stream<hbm_t> s_embedding_buffer_HBM8;
    hls::stream<hbm_t> s_embedding_buffer_HBM9;
    hls::stream<hbm_t> s_embedding_buffer_HBM10;
    hls::stream<hbm_t> s_embedding_buffer_HBM11;
    hls::stream<hbm_t> s_embedding_buffer_HBM12;
    hls::stream<hbm_t> s_embedding_buffer_HBM13;
    hls::stream<hbm_t> s_embedding_buffer_HBM14;
    hls::stream<hbm_t> s_embedding_buffer_HBM15;
    hls::stream<hbm_t> s_embedding_buffer_HBM16;
    hls::stream<hbm_t> s_embedding_buffer_HBM17;
    hls::stream<hbm_t> s_embedding_buffer_HBM18;
    hls::stream<hbm_t> s_embedding_buffer_HBM19;
    hls::stream<hbm_t> s_embedding_buffer_HBM20;
    hls::stream<hbm_t> s_embedding_buffer_HBM21;
    hls::stream<hbm_t> s_embedding_buffer_HBM22;
    hls::stream<hbm_t> s_embedding_buffer_HBM23;
    hls::stream<hbm_t> s_embedding_buffer_HBM24;
    hls::stream<hbm_t> s_embedding_buffer_HBM25;
    hls::stream<hbm_t> s_embedding_buffer_HBM26;
    hls::stream<hbm_t> s_embedding_buffer_HBM27;
#pragma HLS stream variable=s_embedding_buffer_HBM0 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM1 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM2 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM3 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM4 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM5 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM6 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM7 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM8 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM9 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM10 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM11 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM12 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM13 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM14 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM15 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM16 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM17 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM18 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM19 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM20 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM21 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM22 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM23 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM24 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM25 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM26 depth=8
#pragma HLS stream variable=s_embedding_buffer_HBM27 depth=8

#pragma HLS resource variable=s_embedding_buffer_HBM0 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM1 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM2 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM3 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM4 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM5 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM6 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM7 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM8 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM9 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM10 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM11 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM12 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM13 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM14 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM15 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM16 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM17 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM18 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM19 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM20 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM21 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM22 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM23 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM24 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM25 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM26 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_HBM27 core=FIFO_SRL


    hls::stream<plram_t> s_embedding_buffer_PLRAM0;
    hls::stream<plram_t> s_embedding_buffer_PLRAM1;
    hls::stream<plram_t> s_embedding_buffer_PLRAM2;
    hls::stream<plram_t> s_embedding_buffer_PLRAM3;
    hls::stream<plram_t> s_embedding_buffer_PLRAM4;
    hls::stream<plram_t> s_embedding_buffer_PLRAM5;
    hls::stream<plram_t> s_embedding_buffer_PLRAM6;
    hls::stream<plram_t> s_embedding_buffer_PLRAM7;
    hls::stream<plram_t> s_embedding_buffer_PLRAM8;
    hls::stream<plram_t> s_embedding_buffer_PLRAM9;
    hls::stream<plram_t> s_embedding_buffer_PLRAM10;
    hls::stream<plram_t> s_embedding_buffer_PLRAM11;
    hls::stream<plram_t> s_embedding_buffer_PLRAM12;
    hls::stream<plram_t> s_embedding_buffer_PLRAM13;
    hls::stream<plram_t> s_embedding_buffer_PLRAM14;
    hls::stream<plram_t> s_embedding_buffer_PLRAM15;
    hls::stream<plram_t> s_embedding_buffer_PLRAM16;
    hls::stream<plram_t> s_embedding_buffer_PLRAM17;
    hls::stream<plram_t> s_embedding_buffer_PLRAM18;
#pragma HLS stream variable=s_embedding_buffer_PLRAM0 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM1 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM2 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM3 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM4 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM5 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM6 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM7 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM8 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM9 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM10 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM11 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM12 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM13 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM14 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM15 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM16 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM17 depth=8
#pragma HLS stream variable=s_embedding_buffer_PLRAM18 depth=8

#pragma HLS resource variable=s_embedding_buffer_PLRAM0 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM1 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM2 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM3 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM4 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM5 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM6 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM7 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM8 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM9 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM10 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM11 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM12 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM13 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM14 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM15 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM16 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM17 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_PLRAM18 core=FIFO_SRL

    hls::stream<ddr_t> s_embedding_buffer_DDR0;
    hls::stream<ddr_t> s_embedding_buffer_DDR1;
#pragma HLS stream variable=s_embedding_buffer_DDR0 depth=8
#pragma HLS stream variable=s_embedding_buffer_DDR1 depth=8

#pragma HLS resource variable=s_embedding_buffer_DDR0 core=FIFO_SRL
#pragma HLS resource variable=s_embedding_buffer_DDR1 core=FIFO_SRL

    hls::stream<int> s_idx_buffer_HBM0;
    hls::stream<int> s_idx_buffer_HBM1;
    hls::stream<int> s_idx_buffer_HBM2;
    hls::stream<int> s_idx_buffer_HBM3;
    hls::stream<int> s_idx_buffer_HBM4;
    hls::stream<int> s_idx_buffer_HBM5;
    hls::stream<int> s_idx_buffer_HBM6;
    hls::stream<int> s_idx_buffer_HBM7;
    hls::stream<int> s_idx_buffer_HBM8;
    hls::stream<int> s_idx_buffer_HBM9;
    hls::stream<int> s_idx_buffer_HBM10;
    hls::stream<int> s_idx_buffer_HBM11;
    hls::stream<int> s_idx_buffer_HBM12;
    hls::stream<int> s_idx_buffer_HBM13;
    hls::stream<int> s_idx_buffer_HBM14;
    hls::stream<int> s_idx_buffer_HBM15;
    hls::stream<int> s_idx_buffer_HBM16;
    hls::stream<int> s_idx_buffer_HBM17;
    hls::stream<int> s_idx_buffer_HBM18;
    hls::stream<int> s_idx_buffer_HBM19;
    hls::stream<int> s_idx_buffer_HBM20;
    hls::stream<int> s_idx_buffer_HBM21;
    hls::stream<int> s_idx_buffer_HBM22;
    hls::stream<int> s_idx_buffer_HBM23;
    hls::stream<int> s_idx_buffer_HBM24;
    hls::stream<int> s_idx_buffer_HBM25;
    hls::stream<int> s_idx_buffer_HBM26;
    hls::stream<int> s_idx_buffer_HBM27;
#pragma HLS stream variable=s_idx_buffer_HBM0 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM1 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM2 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM3 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM4 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM5 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM6 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM7 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM8 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM9 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM10 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM11 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM12 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM13 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM14 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM15 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM16 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM17 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM18 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM19 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM20 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM21 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM22 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM23 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM24 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM25 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM26 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_HBM27 depth=fifo_batch_size

    hls::stream<int> s_idx_buffer_PLRAM0;
    hls::stream<int> s_idx_buffer_PLRAM1;
    hls::stream<int> s_idx_buffer_PLRAM2;
    hls::stream<int> s_idx_buffer_PLRAM3;
    hls::stream<int> s_idx_buffer_PLRAM4;
    hls::stream<int> s_idx_buffer_PLRAM5;
    hls::stream<int> s_idx_buffer_PLRAM6;
    hls::stream<int> s_idx_buffer_PLRAM7;
    hls::stream<int> s_idx_buffer_PLRAM8;
    hls::stream<int> s_idx_buffer_PLRAM9;
    hls::stream<int> s_idx_buffer_PLRAM10;
    hls::stream<int> s_idx_buffer_PLRAM11;
    hls::stream<int> s_idx_buffer_PLRAM12;
    hls::stream<int> s_idx_buffer_PLRAM13;
    hls::stream<int> s_idx_buffer_PLRAM14;
    hls::stream<int> s_idx_buffer_PLRAM15;
    hls::stream<int> s_idx_buffer_PLRAM16;
    hls::stream<int> s_idx_buffer_PLRAM17;
    hls::stream<int> s_idx_buffer_PLRAM18;
#pragma HLS stream variable=s_idx_buffer_PLRAM0 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM0 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM1 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM2 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM3 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM4 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM5 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM6 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM7 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM8 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM9 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM10 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM11 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM12 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM13 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM14 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM15 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM16 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM17 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_PLRAM18 depth=fifo_batch_size

    hls::stream<int> s_idx_buffer_DDR0;
    hls::stream<int> s_idx_buffer_DDR1;
#pragma HLS stream variable=s_idx_buffer_DDR0 depth=fifo_batch_size
#pragma HLS stream variable=s_idx_buffer_DDR1 depth=fifo_batch_size

    hls::stream<network_t> s_network;
#pragma HLS stream variable=s_network depth=128
    
    plram_t table_PLRAM0[PLRAM_BANK0_SIZE]; 
    plram_t table_PLRAM1[PLRAM_BANK1_SIZE]; 
    plram_t table_PLRAM2[PLRAM_BANK2_SIZE]; 
    plram_t table_PLRAM3[PLRAM_BANK3_SIZE]; 
    plram_t table_PLRAM4[PLRAM_BANK4_SIZE]; 
    plram_t table_PLRAM5[PLRAM_BANK5_SIZE]; 
    plram_t table_PLRAM6[PLRAM_BANK6_SIZE]; 
    plram_t table_PLRAM7[PLRAM_BANK7_SIZE]; 
    plram_t table_PLRAM8[PLRAM_BANK8_SIZE]; 
    plram_t table_PLRAM9[PLRAM_BANK9_SIZE]; 
    plram_t table_PLRAM10[PLRAM_BANK10_SIZE]; 
    plram_t table_PLRAM11[PLRAM_BANK11_SIZE]; 
    plram_t table_PLRAM12[PLRAM_BANK12_SIZE]; 
    plram_t table_PLRAM13[PLRAM_BANK13_SIZE]; 
    plram_t table_PLRAM14[PLRAM_BANK14_SIZE]; 
    plram_t table_PLRAM15[PLRAM_BANK15_SIZE]; 
    plram_t table_PLRAM16[PLRAM_BANK16_SIZE]; 
    plram_t table_PLRAM17[PLRAM_BANK17_SIZE]; 
    plram_t table_PLRAM18[PLRAM_BANK18_SIZE]; 
#pragma HLS resource variable=table_PLRAM0 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM1 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM2 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM3 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM4 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM5 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM6 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM7 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM8 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM9 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM10 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM11 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM12 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM13 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM14 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM15 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM16 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM17 core=RAM_2P_URAM
#pragma HLS resource variable=table_PLRAM18 core=RAM_2P_URAM

    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_0, AXI_PADDED_SIZE_PLRAM_0, TABLE_SIZE_PLRAM_0,
         ADDR_AXI_PLRAM_19, AXI_PADDED_SIZE_PLRAM_19, TABLE_SIZE_PLRAM_19>(table_PLRAM0);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_1, AXI_PADDED_SIZE_PLRAM_1, TABLE_SIZE_PLRAM_1,
         ADDR_AXI_PLRAM_20, AXI_PADDED_SIZE_PLRAM_20, TABLE_SIZE_PLRAM_20>(table_PLRAM1);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_2, AXI_PADDED_SIZE_PLRAM_2, TABLE_SIZE_PLRAM_2,
         ADDR_AXI_PLRAM_21, AXI_PADDED_SIZE_PLRAM_21, TABLE_SIZE_PLRAM_21>(table_PLRAM2);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_3, AXI_PADDED_SIZE_PLRAM_3, TABLE_SIZE_PLRAM_3,
         ADDR_AXI_PLRAM_22, AXI_PADDED_SIZE_PLRAM_22, TABLE_SIZE_PLRAM_22>(table_PLRAM3);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_4, AXI_PADDED_SIZE_PLRAM_4, TABLE_SIZE_PLRAM_4,
         ADDR_AXI_PLRAM_23, AXI_PADDED_SIZE_PLRAM_23, TABLE_SIZE_PLRAM_23>(table_PLRAM4);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_5, AXI_PADDED_SIZE_PLRAM_5, TABLE_SIZE_PLRAM_5,
         ADDR_AXI_PLRAM_24, AXI_PADDED_SIZE_PLRAM_24, TABLE_SIZE_PLRAM_24>(table_PLRAM5);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_6, AXI_PADDED_SIZE_PLRAM_6, TABLE_SIZE_PLRAM_6,
         ADDR_AXI_PLRAM_25, AXI_PADDED_SIZE_PLRAM_25, TABLE_SIZE_PLRAM_25>(table_PLRAM6);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_7, AXI_PADDED_SIZE_PLRAM_7, TABLE_SIZE_PLRAM_7,
         ADDR_AXI_PLRAM_26, AXI_PADDED_SIZE_PLRAM_26, TABLE_SIZE_PLRAM_26>(table_PLRAM7);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_8, AXI_PADDED_SIZE_PLRAM_8, TABLE_SIZE_PLRAM_8,
         ADDR_AXI_PLRAM_27, AXI_PADDED_SIZE_PLRAM_27, TABLE_SIZE_PLRAM_27>(table_PLRAM8);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_9, AXI_PADDED_SIZE_PLRAM_9, TABLE_SIZE_PLRAM_9,
         ADDR_AXI_PLRAM_28, AXI_PADDED_SIZE_PLRAM_28, TABLE_SIZE_PLRAM_28>(table_PLRAM9);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_10, AXI_PADDED_SIZE_PLRAM_10, TABLE_SIZE_PLRAM_10,
         ADDR_AXI_PLRAM_29, AXI_PADDED_SIZE_PLRAM_29, TABLE_SIZE_PLRAM_29>(table_PLRAM10);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_11, AXI_PADDED_SIZE_PLRAM_11, TABLE_SIZE_PLRAM_11,
         ADDR_AXI_PLRAM_30, AXI_PADDED_SIZE_PLRAM_30, TABLE_SIZE_PLRAM_30>(table_PLRAM11);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_12, AXI_PADDED_SIZE_PLRAM_12, TABLE_SIZE_PLRAM_12,
         ADDR_AXI_PLRAM_31, AXI_PADDED_SIZE_PLRAM_31, TABLE_SIZE_PLRAM_31>(table_PLRAM12);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_13, AXI_PADDED_SIZE_PLRAM_13, TABLE_SIZE_PLRAM_13,
         ADDR_AXI_PLRAM_32, AXI_PADDED_SIZE_PLRAM_32, TABLE_SIZE_PLRAM_32>(table_PLRAM13);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_14, AXI_PADDED_SIZE_PLRAM_14, TABLE_SIZE_PLRAM_14,
         ADDR_AXI_PLRAM_33, AXI_PADDED_SIZE_PLRAM_33, TABLE_SIZE_PLRAM_33>(table_PLRAM14);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_15, AXI_PADDED_SIZE_PLRAM_15, TABLE_SIZE_PLRAM_15,
         ADDR_AXI_PLRAM_34, AXI_PADDED_SIZE_PLRAM_34, TABLE_SIZE_PLRAM_34>(table_PLRAM15);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_16, AXI_PADDED_SIZE_PLRAM_16, TABLE_SIZE_PLRAM_16,
         ADDR_AXI_PLRAM_35, AXI_PADDED_SIZE_PLRAM_35, TABLE_SIZE_PLRAM_35>(table_PLRAM16);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_17, AXI_PADDED_SIZE_PLRAM_17, TABLE_SIZE_PLRAM_17,
         ADDR_AXI_PLRAM_36, AXI_PADDED_SIZE_PLRAM_36, TABLE_SIZE_PLRAM_36>(table_PLRAM17);
    init_plram_t_2_tables
        <ADDR_AXI_PLRAM_18, AXI_PADDED_SIZE_PLRAM_18, TABLE_SIZE_PLRAM_18,
         ADDR_AXI_PLRAM_37, AXI_PADDED_SIZE_PLRAM_37, TABLE_SIZE_PLRAM_37>(table_PLRAM18);

    load_access_idx(s_idx_buffer_HBM0, batch_num);
    load_access_idx(s_idx_buffer_HBM1, batch_num);
    load_access_idx(s_idx_buffer_HBM2, batch_num);
    load_access_idx(s_idx_buffer_HBM3, batch_num);
    load_access_idx(s_idx_buffer_HBM4, batch_num);
    load_access_idx(s_idx_buffer_HBM5, batch_num);
    load_access_idx(s_idx_buffer_HBM6, batch_num);
    load_access_idx(s_idx_buffer_HBM7, batch_num);
    load_access_idx(s_idx_buffer_HBM8, batch_num);
    load_access_idx(s_idx_buffer_HBM9, batch_num);
    load_access_idx(s_idx_buffer_HBM10, batch_num);
    load_access_idx(s_idx_buffer_HBM11, batch_num);
    load_access_idx(s_idx_buffer_HBM12, batch_num);
    load_access_idx(s_idx_buffer_HBM13, batch_num);
    load_access_idx(s_idx_buffer_HBM14, batch_num);
    load_access_idx(s_idx_buffer_HBM15, batch_num);
    load_access_idx(s_idx_buffer_HBM16, batch_num);
    load_access_idx(s_idx_buffer_HBM17, batch_num);
    load_access_idx(s_idx_buffer_HBM18, batch_num);
    load_access_idx(s_idx_buffer_HBM19, batch_num);
    load_access_idx(s_idx_buffer_HBM20, batch_num);
    load_access_idx(s_idx_buffer_HBM21, batch_num);
    load_access_idx(s_idx_buffer_HBM22, batch_num);
    load_access_idx(s_idx_buffer_HBM23, batch_num);
    load_access_idx(s_idx_buffer_HBM24, batch_num);
    load_access_idx(s_idx_buffer_HBM25, batch_num);
    load_access_idx(s_idx_buffer_HBM26, batch_num);
    load_access_idx(s_idx_buffer_HBM27, batch_num);

    load_access_idx(s_idx_buffer_DDR0, batch_num);
    load_access_idx(s_idx_buffer_DDR1, batch_num);

    load_access_idx(s_idx_buffer_PLRAM0, batch_num);
    load_access_idx(s_idx_buffer_PLRAM1, batch_num);
    load_access_idx(s_idx_buffer_PLRAM2, batch_num);
    load_access_idx(s_idx_buffer_PLRAM3, batch_num);
    load_access_idx(s_idx_buffer_PLRAM4, batch_num);
    load_access_idx(s_idx_buffer_PLRAM5, batch_num);
    load_access_idx(s_idx_buffer_PLRAM6, batch_num);
    load_access_idx(s_idx_buffer_PLRAM7, batch_num);
    load_access_idx(s_idx_buffer_PLRAM8, batch_num);
    load_access_idx(s_idx_buffer_PLRAM9, batch_num);
    load_access_idx(s_idx_buffer_PLRAM10, batch_num);
    load_access_idx(s_idx_buffer_PLRAM11, batch_num);
    load_access_idx(s_idx_buffer_PLRAM12, batch_num);
    load_access_idx(s_idx_buffer_PLRAM13, batch_num);
    load_access_idx(s_idx_buffer_PLRAM14, batch_num);
    load_access_idx(s_idx_buffer_PLRAM15, batch_num);
    load_access_idx(s_idx_buffer_PLRAM16, batch_num);
    load_access_idx(s_idx_buffer_PLRAM17, batch_num);
    load_access_idx(s_idx_buffer_PLRAM18, batch_num);

//////////////////////////////     Load Vectors & Concatenate     ////////////////////////////// 

    load_single_embedding_2_tables<ADDR_AXI_HBM_0, AXI_PADDED_SIZE_HBM_0, ADDR_AXI_HBM_28, AXI_PADDED_SIZE_HBM_28>(
        s_idx_buffer_HBM0, table_HBM0, s_embedding_buffer_HBM0, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_1, AXI_PADDED_SIZE_HBM_1, ADDR_AXI_HBM_29, AXI_PADDED_SIZE_HBM_29>(
        s_idx_buffer_HBM1, table_HBM1, s_embedding_buffer_HBM1, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_2, AXI_PADDED_SIZE_HBM_2, ADDR_AXI_HBM_30, AXI_PADDED_SIZE_HBM_30>(
        s_idx_buffer_HBM2, table_HBM2, s_embedding_buffer_HBM2, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_3, AXI_PADDED_SIZE_HBM_3, ADDR_AXI_HBM_31, AXI_PADDED_SIZE_HBM_31>(
        s_idx_buffer_HBM3, table_HBM3, s_embedding_buffer_HBM3, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_4, AXI_PADDED_SIZE_HBM_4, ADDR_AXI_HBM_32, AXI_PADDED_SIZE_HBM_32>(
        s_idx_buffer_HBM4, table_HBM4, s_embedding_buffer_HBM4, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_5, AXI_PADDED_SIZE_HBM_5, ADDR_AXI_HBM_33, AXI_PADDED_SIZE_HBM_33>(
        s_idx_buffer_HBM5, table_HBM5, s_embedding_buffer_HBM5, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_6, AXI_PADDED_SIZE_HBM_6, ADDR_AXI_HBM_34, AXI_PADDED_SIZE_HBM_34>(
        s_idx_buffer_HBM6, table_HBM6, s_embedding_buffer_HBM6, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_7, AXI_PADDED_SIZE_HBM_7, ADDR_AXI_HBM_35, AXI_PADDED_SIZE_HBM_35>(
        s_idx_buffer_HBM7, table_HBM7, s_embedding_buffer_HBM7, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_8, AXI_PADDED_SIZE_HBM_8, ADDR_AXI_HBM_36, AXI_PADDED_SIZE_HBM_36>(
        s_idx_buffer_HBM8, table_HBM8, s_embedding_buffer_HBM8, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_9, AXI_PADDED_SIZE_HBM_9, ADDR_AXI_HBM_37, AXI_PADDED_SIZE_HBM_37>(
        s_idx_buffer_HBM9, table_HBM9, s_embedding_buffer_HBM9, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_10, AXI_PADDED_SIZE_HBM_10, ADDR_AXI_HBM_38, AXI_PADDED_SIZE_HBM_38>(
        s_idx_buffer_HBM10, table_HBM10, s_embedding_buffer_HBM10, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_11, AXI_PADDED_SIZE_HBM_11, ADDR_AXI_HBM_39, AXI_PADDED_SIZE_HBM_39>(
        s_idx_buffer_HBM11, table_HBM11, s_embedding_buffer_HBM11, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_12, AXI_PADDED_SIZE_HBM_12, ADDR_AXI_HBM_40, AXI_PADDED_SIZE_HBM_40>(
        s_idx_buffer_HBM12, table_HBM12, s_embedding_buffer_HBM12, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_13, AXI_PADDED_SIZE_HBM_13, ADDR_AXI_HBM_41, AXI_PADDED_SIZE_HBM_41>(
        s_idx_buffer_HBM13, table_HBM13, s_embedding_buffer_HBM13, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_14, AXI_PADDED_SIZE_HBM_14, ADDR_AXI_HBM_42, AXI_PADDED_SIZE_HBM_42>(
        s_idx_buffer_HBM14, table_HBM14, s_embedding_buffer_HBM14, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_15, AXI_PADDED_SIZE_HBM_15, ADDR_AXI_HBM_43, AXI_PADDED_SIZE_HBM_43>(
        s_idx_buffer_HBM15, table_HBM15, s_embedding_buffer_HBM15, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_16, AXI_PADDED_SIZE_HBM_16, ADDR_AXI_HBM_44, AXI_PADDED_SIZE_HBM_44>(
        s_idx_buffer_HBM16, table_HBM16, s_embedding_buffer_HBM16, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_17, AXI_PADDED_SIZE_HBM_17, ADDR_AXI_HBM_45, AXI_PADDED_SIZE_HBM_45>(
        s_idx_buffer_HBM17, table_HBM17, s_embedding_buffer_HBM17, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_18, AXI_PADDED_SIZE_HBM_18, ADDR_AXI_HBM_46, AXI_PADDED_SIZE_HBM_46>(
        s_idx_buffer_HBM18, table_HBM18, s_embedding_buffer_HBM18, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_19, AXI_PADDED_SIZE_HBM_19, ADDR_AXI_HBM_47, AXI_PADDED_SIZE_HBM_47>(
        s_idx_buffer_HBM19, table_HBM19, s_embedding_buffer_HBM19, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_20, AXI_PADDED_SIZE_HBM_20, ADDR_AXI_HBM_48, AXI_PADDED_SIZE_HBM_48>(
        s_idx_buffer_HBM20, table_HBM20, s_embedding_buffer_HBM20, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_21, AXI_PADDED_SIZE_HBM_21, ADDR_AXI_HBM_49, AXI_PADDED_SIZE_HBM_49>(
        s_idx_buffer_HBM21, table_HBM21, s_embedding_buffer_HBM21, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_22, AXI_PADDED_SIZE_HBM_22, ADDR_AXI_HBM_50, AXI_PADDED_SIZE_HBM_50>(
        s_idx_buffer_HBM22, table_HBM22, s_embedding_buffer_HBM22, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_23, AXI_PADDED_SIZE_HBM_23, ADDR_AXI_HBM_51, AXI_PADDED_SIZE_HBM_51>(
        s_idx_buffer_HBM23, table_HBM23, s_embedding_buffer_HBM23, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_24, AXI_PADDED_SIZE_HBM_24, ADDR_AXI_HBM_52, AXI_PADDED_SIZE_HBM_52>(
        s_idx_buffer_HBM24, table_HBM24, s_embedding_buffer_HBM24, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_25, AXI_PADDED_SIZE_HBM_25, ADDR_AXI_HBM_53, AXI_PADDED_SIZE_HBM_53>(
        s_idx_buffer_HBM25, table_HBM25, s_embedding_buffer_HBM25, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_26, AXI_PADDED_SIZE_HBM_26, ADDR_AXI_HBM_54, AXI_PADDED_SIZE_HBM_54>(
        s_idx_buffer_HBM26, table_HBM26, s_embedding_buffer_HBM26, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_HBM_27, AXI_PADDED_SIZE_HBM_27, ADDR_AXI_HBM_55, AXI_PADDED_SIZE_HBM_55>(
        s_idx_buffer_HBM27, table_HBM27, s_embedding_buffer_HBM27, batch_num);

    load_single_embedding_2_tables<ADDR_AXI_DDR_0, AXI_PADDED_SIZE_DDR_0, ADDR_AXI_DDR_2, AXI_PADDED_SIZE_DDR_2>(
        s_idx_buffer_DDR0, table_DDR0, s_embedding_buffer_DDR0, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_DDR_1, AXI_PADDED_SIZE_DDR_1, ADDR_AXI_DDR_3, AXI_PADDED_SIZE_DDR_3>(
        s_idx_buffer_DDR1, table_DDR1, s_embedding_buffer_DDR1, batch_num);

    load_single_embedding_2_tables<ADDR_AXI_PLRAM_0, AXI_PADDED_SIZE_PLRAM_0, ADDR_AXI_PLRAM_19, AXI_PADDED_SIZE_PLRAM_19>(
        s_idx_buffer_PLRAM0, table_PLRAM0, s_embedding_buffer_PLRAM0, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_1, AXI_PADDED_SIZE_PLRAM_1, ADDR_AXI_PLRAM_20, AXI_PADDED_SIZE_PLRAM_20>(
        s_idx_buffer_PLRAM1, table_PLRAM1, s_embedding_buffer_PLRAM1, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_2, AXI_PADDED_SIZE_PLRAM_2, ADDR_AXI_PLRAM_21, AXI_PADDED_SIZE_PLRAM_21>(
        s_idx_buffer_PLRAM2, table_PLRAM2, s_embedding_buffer_PLRAM2, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_3, AXI_PADDED_SIZE_PLRAM_3, ADDR_AXI_PLRAM_22, AXI_PADDED_SIZE_PLRAM_22>(
        s_idx_buffer_PLRAM3, table_PLRAM3, s_embedding_buffer_PLRAM3, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_4, AXI_PADDED_SIZE_PLRAM_4, ADDR_AXI_PLRAM_23, AXI_PADDED_SIZE_PLRAM_23>(
        s_idx_buffer_PLRAM4, table_PLRAM4, s_embedding_buffer_PLRAM4, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_5, AXI_PADDED_SIZE_PLRAM_5, ADDR_AXI_PLRAM_24, AXI_PADDED_SIZE_PLRAM_24>(
        s_idx_buffer_PLRAM5, table_PLRAM5, s_embedding_buffer_PLRAM5, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_6, AXI_PADDED_SIZE_PLRAM_6, ADDR_AXI_PLRAM_25, AXI_PADDED_SIZE_PLRAM_25>(
        s_idx_buffer_PLRAM6, table_PLRAM6, s_embedding_buffer_PLRAM6, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_7, AXI_PADDED_SIZE_PLRAM_7, ADDR_AXI_PLRAM_26, AXI_PADDED_SIZE_PLRAM_26>(
        s_idx_buffer_PLRAM7, table_PLRAM7, s_embedding_buffer_PLRAM7, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_8, AXI_PADDED_SIZE_PLRAM_8, ADDR_AXI_PLRAM_27, AXI_PADDED_SIZE_PLRAM_27>(
        s_idx_buffer_PLRAM8, table_PLRAM8, s_embedding_buffer_PLRAM8, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_9, AXI_PADDED_SIZE_PLRAM_9, ADDR_AXI_PLRAM_28, AXI_PADDED_SIZE_PLRAM_28>(
        s_idx_buffer_PLRAM9, table_PLRAM9, s_embedding_buffer_PLRAM9, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_10, AXI_PADDED_SIZE_PLRAM_10, ADDR_AXI_PLRAM_29, AXI_PADDED_SIZE_PLRAM_29>(
        s_idx_buffer_PLRAM10, table_PLRAM10, s_embedding_buffer_PLRAM10, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_11, AXI_PADDED_SIZE_PLRAM_11, ADDR_AXI_PLRAM_30, AXI_PADDED_SIZE_PLRAM_30>(
        s_idx_buffer_PLRAM11, table_PLRAM11, s_embedding_buffer_PLRAM11, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_12, AXI_PADDED_SIZE_PLRAM_12, ADDR_AXI_PLRAM_31, AXI_PADDED_SIZE_PLRAM_31>(
        s_idx_buffer_PLRAM12, table_PLRAM12, s_embedding_buffer_PLRAM12, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_13, AXI_PADDED_SIZE_PLRAM_13, ADDR_AXI_PLRAM_32, AXI_PADDED_SIZE_PLRAM_32>(
        s_idx_buffer_PLRAM13, table_PLRAM13, s_embedding_buffer_PLRAM13, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_14, AXI_PADDED_SIZE_PLRAM_14, ADDR_AXI_PLRAM_33, AXI_PADDED_SIZE_PLRAM_33>(
        s_idx_buffer_PLRAM14, table_PLRAM14, s_embedding_buffer_PLRAM14, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_15, AXI_PADDED_SIZE_PLRAM_15, ADDR_AXI_PLRAM_34, AXI_PADDED_SIZE_PLRAM_34>(
        s_idx_buffer_PLRAM15, table_PLRAM15, s_embedding_buffer_PLRAM15, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_16, AXI_PADDED_SIZE_PLRAM_16, ADDR_AXI_PLRAM_35, AXI_PADDED_SIZE_PLRAM_35>(
        s_idx_buffer_PLRAM16, table_PLRAM16, s_embedding_buffer_PLRAM16, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_17, AXI_PADDED_SIZE_PLRAM_17, ADDR_AXI_PLRAM_36, AXI_PADDED_SIZE_PLRAM_36>(
        s_idx_buffer_PLRAM17, table_PLRAM17, s_embedding_buffer_PLRAM17, batch_num);
    load_single_embedding_2_tables<ADDR_AXI_PLRAM_18, AXI_PADDED_SIZE_PLRAM_18, ADDR_AXI_PLRAM_37, AXI_PADDED_SIZE_PLRAM_37>(
        s_idx_buffer_PLRAM18, table_PLRAM18, s_embedding_buffer_PLRAM18, batch_num);

    gather_embeddings(
        s_embedding_buffer_HBM0, s_embedding_buffer_HBM1, 
        s_embedding_buffer_HBM2, s_embedding_buffer_HBM3, 
        s_embedding_buffer_HBM4, s_embedding_buffer_HBM5, 
        s_embedding_buffer_HBM6, s_embedding_buffer_HBM7, 
        s_embedding_buffer_HBM8, s_embedding_buffer_HBM9, 
        s_embedding_buffer_HBM10, s_embedding_buffer_HBM11, 
        s_embedding_buffer_HBM12, s_embedding_buffer_HBM13, 
        s_embedding_buffer_HBM14, s_embedding_buffer_HBM15, 
        s_embedding_buffer_HBM16, s_embedding_buffer_HBM17, 
        s_embedding_buffer_HBM18, s_embedding_buffer_HBM19, 
        s_embedding_buffer_HBM20, s_embedding_buffer_HBM21, 
        s_embedding_buffer_HBM22, s_embedding_buffer_HBM23, 
        s_embedding_buffer_HBM24, s_embedding_buffer_HBM25, 
        s_embedding_buffer_HBM26, s_embedding_buffer_HBM27, 
        
        s_embedding_buffer_DDR0, s_embedding_buffer_DDR1,

        s_embedding_buffer_PLRAM0, s_embedding_buffer_PLRAM1, 
        s_embedding_buffer_PLRAM2, s_embedding_buffer_PLRAM3,
        s_embedding_buffer_PLRAM4, s_embedding_buffer_PLRAM5,
        s_embedding_buffer_PLRAM6, s_embedding_buffer_PLRAM7,
        s_embedding_buffer_PLRAM8, s_embedding_buffer_PLRAM9,
        s_embedding_buffer_PLRAM10, s_embedding_buffer_PLRAM11,
        s_embedding_buffer_PLRAM12, s_embedding_buffer_PLRAM13,
        s_embedding_buffer_PLRAM14, s_embedding_buffer_PLRAM15,
        s_embedding_buffer_PLRAM16, s_embedding_buffer_PLRAM17,
        s_embedding_buffer_PLRAM18,

        s_network,
        
        batch_num);

    ap_uint<64> expectedTxByteCnt = batch_num * BATCH_SIZE * INPUT_SIZE_AXI_512 * 64;
    
    sendData(m_axis_tcp_tx_meta, 
             m_axis_tcp_tx_data, 
             s_axis_tcp_tx_status,
             s_network,
             sessionID,
             useConn,
             expectedTxByteCnt, 
             pkgWordCount
             );
    

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


    // consume_W(s_feature_in);



//////////////////////////////     Write results     ////////////////////////////// 
}

}

template<
    const int start_addr_0, const int axi_padded_size_0, const int table_entry_num_0,
    const int start_addr_1, const int axi_padded_size_1, const int table_entry_num_1>
void init_plram_t_2_tables(
    plram_t* table_PLRAM) {

    // even row: 1 
    // odd row: -1
    // NOTE! must use ap_uint<128> here
    // float one_arr[4] = {1, 1, 1, 1};
    // float zero_arr[4] = {0, 0, 0, 0};
    int one = 1065353216; // float 1 -> same bit ->  int 1065353216
    int zero = 0; // float 0 -> int 0
    for (int i = 0 ; i < table_entry_num_0 / 2; i++) {
        for (int j = 0; j < axi_padded_size_0; j++) {
            // memcpy(&table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j], one_arr, 16);
            table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j].range(31 , 0) = one;
            table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j].range(63 , 32) = one;
            table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j].range(95 , 64) = one;
            table_PLRAM[start_addr_0 + (2 * i) * axi_padded_size_0 + j].range(127 , 96) = one;
            }
        for (int j = 0; j < axi_padded_size_0; j++) {
            // memcpy(&table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j], zero_arr, 16);
            table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j].range(31, 0) = zero;
            table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j].range(63, 32) = zero;
            table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j].range(95, 64) = zero;
            table_PLRAM[start_addr_0 + (2 * i + 1) * axi_padded_size_0 + j].range(127, 96) = zero;
        }
    }
    for (int i = 0 ; i < table_entry_num_1 / 2; i++) {
        for (int j = 0; j < axi_padded_size_1; j++) {
            // memcpy(&table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j], one_arr, 16);
            table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j].range(31 , 0) = one;
            table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j].range(63 , 32) = one;
            table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j].range(95 , 64) = one;
            table_PLRAM[start_addr_1 + (2 * i) * axi_padded_size_1 + j].range(127 , 96) = one;
            }
        for (int j = 0; j < axi_padded_size_1; j++) {
            // memcpy(&table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j], zero_arr, 16);
            table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j].range(31, 0) = zero;
            table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j].range(63, 32) = zero;
            table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j].range(95, 64) = zero;
            table_PLRAM[start_addr_1 + (2 * i + 1) * axi_padded_size_1 + j].range(127, 96) = zero;
        }
    }
}

void load_access_idx(
    hls::stream<int>& s_idx_buffer_single_channel, int batch_num) { 

    int idx_random[] = {3, 99, 38, 72, 29, 57, 1, 72, 36, 76, 35, 50, 37, 57, 
        13, 66, 26, 70, 41, 93, 48, 82, 44, 78, 25, 52, 3, 92, 36, 56, 46, 88};

    for (int i = 0; i < batch_num; i++) {
        for (int j = 0; j < BATCH_SIZE; j++) {
            #pragma HLS pipeline II=1

            int idx = idx_random[j]; 
            s_idx_buffer_single_channel.write(idx);
        }
    }
}

template<
    const long start_addr_0, const long AXI_padded_size_0, 
    const long start_addr_1, const long AXI_padded_size_1>
void load_single_embedding_2_tables(
    hls::stream<int>& s_idx_buffer, const hbm_t* table_RAM, 
    hls::stream<hbm_t>& s_embedding_buffer,
    int batch_num) {
#pragma HLS INLINE off

    // 8 < data size <= 16, load 2 times
    for (int i = 0; i < batch_num * BATCH_SIZE; i++) {

        long idx = s_idx_buffer.read();

        long base_addr_0 = start_addr_0 + idx * AXI_padded_size_0;
        for (int j = 0; j < AXI_padded_size_0; j++) {
            #pragma HLS pipeline II=1
            s_embedding_buffer.write(table_RAM[base_addr_0 + j]);
        }
        long base_addr_1 = start_addr_1 + idx * AXI_padded_size_1;
        for (int j = 0; j < AXI_padded_size_1; j++) {
            #pragma HLS pipeline II=1
            s_embedding_buffer.write(table_RAM[base_addr_1 + j]);
        }
    }
}

void group_4_embeddings_padded(
    hls::stream<axi_t>& s_embedding_0, hls::stream<axi_t>& s_embedding_1,
    hls::stream<axi_t>& s_embedding_2, hls::stream<axi_t>& s_embedding_3,
    hls::stream<network_t>& s_gathered_embedding_0,
    hls::stream<network_t>& s_gathered_embedding_1,
    hls::stream<network_t>& s_gathered_embedding_2,
    hls::stream<network_t>& s_gathered_embedding_3,
    int batch_num
) {
    // Concatenate 4 embedding streams
    //   padding: plram 16~18 (3x3) + hbm 27 (1x6) + 1 = 16 x axi 128
    //   out: 4 stream of axi 512

    for (int i = 0; i < batch_num * BATCH_SIZE; i++) {
        #pragma HLS pipeline II=6

        axi_t tmp0 = s_embedding_0.read();
        axi_t tmp1 = s_embedding_0.read();
        axi_t tmp2 = s_embedding_0.read();
        
        axi_t tmp3 = s_embedding_1.read();
        axi_t tmp4 = s_embedding_1.read();
        axi_t tmp5 = s_embedding_1.read();

        axi_t tmp6 = s_embedding_2.read();
        axi_t tmp7 = s_embedding_2.read();
        axi_t tmp8 = s_embedding_2.read();
        
        axi_t tmp9 = s_embedding_3.read();
        axi_t tmp10 = s_embedding_3.read();
        axi_t tmp11 = s_embedding_3.read();
        axi_t tmp12 = s_embedding_3.read();
        axi_t tmp13 = s_embedding_3.read();
        axi_t tmp14 = s_embedding_3.read();

        axi_t tmp_padded = tmp0;
        
        network_t out0, out1, out2, out3;
        out0.range(127, 0) = tmp0;
        out0.range(255, 128) = tmp1;
        out0.range(383, 256) = tmp2;
        out0.range(511, 384) = tmp3;

        out1.range(127, 0) = tmp4;
        out1.range(255, 128) = tmp5;
        out1.range(383, 256) = tmp6;
        out1.range(511, 384) = tmp7;

        out2.range(127, 0) = tmp8;
        out2.range(255, 128) = tmp9;
        out2.range(383, 256) = tmp10;
        out2.range(511, 384) = tmp11;

        out3.range(127, 0) = tmp12;
        out3.range(255, 128) = tmp13;
        out3.range(383, 256) = tmp14;
        out3.range(511, 384) = tmp_padded;

        s_gathered_embedding_0.write(out0);
        s_gathered_embedding_1.write(out1);
        s_gathered_embedding_2.write(out2);
        s_gathered_embedding_3.write(out3);
    }
}

void group_4_embeddings_len3(
    hls::stream<axi_t>& s_embedding_0, hls::stream<axi_t>& s_embedding_1,
    hls::stream<axi_t>& s_embedding_2, hls::stream<axi_t>& s_embedding_3,
    hls::stream<network_t>& s_gathered_embedding_0,
    hls::stream<network_t>& s_gathered_embedding_1,
    hls::stream<network_t>& s_gathered_embedding_2,
    int batch_num
) {
    // Concatenate 4 embedding streams
    //   each stream: 3 x axi 128

    for (int i = 0; i < batch_num * BATCH_SIZE; i++) {
        #pragma HLS pipeline II=3

        axi_t tmp0 = s_embedding_0.read();
        axi_t tmp1 = s_embedding_0.read();
        axi_t tmp2 = s_embedding_0.read();
        
        axi_t tmp3 = s_embedding_1.read();
        axi_t tmp4 = s_embedding_1.read();
        axi_t tmp5 = s_embedding_1.read();

        axi_t tmp6 = s_embedding_2.read();
        axi_t tmp7 = s_embedding_2.read();
        axi_t tmp8 = s_embedding_2.read();
        
        axi_t tmp9 = s_embedding_3.read();
        axi_t tmp10 = s_embedding_3.read();
        axi_t tmp11 = s_embedding_3.read();
        
        network_t out0, out1, out2;
        out0.range(127, 0) = tmp0;
        out0.range(255, 128) = tmp1;
        out0.range(383, 256) = tmp2;
        out0.range(511, 384) = tmp3;

        out1.range(127, 0) = tmp4;
        out1.range(255, 128) = tmp5;
        out1.range(383, 256) = tmp6;
        out1.range(511, 384) = tmp7;

        out2.range(127, 0) = tmp8;
        out2.range(255, 128) = tmp9;
        out2.range(383, 256) = tmp10;
        out2.range(511, 384) = tmp11;

        s_gathered_embedding_0.write(out0);
        s_gathered_embedding_1.write(out1);
        s_gathered_embedding_2.write(out2);
    }
}

void group_2_embeddings_len_2(
    hls::stream<axi_t>& s_embedding_0, hls::stream<axi_t>& s_embedding_1,
    hls::stream<network_t>& s_gathered_embedding,
    int batch_num
) {
    // Concatenate 2 embedding streams
    //   each stream: 2 x axi 128

    for (int i = 0; i < batch_num * BATCH_SIZE; i++) {
        #pragma HLS pipeline II=2

        axi_t tmp0 = s_embedding_0.read();
        axi_t tmp1 = s_embedding_0.read();
        axi_t tmp2 = s_embedding_1.read();
        axi_t tmp3 = s_embedding_1.read();
        
        network_t out;
        out.range(127, 0) = tmp0;
        out.range(255, 128) = tmp1;
        out.range(383, 256) = tmp2;
        out.range(511, 384) = tmp3;

        s_gathered_embedding.write(out);
    }
}

void group_2_embeddings_len_6(
    hls::stream<axi_t>& s_embedding_0, hls::stream<axi_t>& s_embedding_1,
    hls::stream<network_t>& s_gathered_embedding_0,
    hls::stream<network_t>& s_gathered_embedding_1,
    hls::stream<network_t>& s_gathered_embedding_2,
    int batch_num
) {
    // Concatenate 2 embedding streams
    //   each stream: 6 x axi 128

    for (int i = 0; i < batch_num * BATCH_SIZE; i++) {
        #pragma HLS pipeline II=6

        axi_t tmp0 = s_embedding_0.read();
        axi_t tmp1 = s_embedding_0.read();
        axi_t tmp2 = s_embedding_0.read();
        axi_t tmp3 = s_embedding_0.read();
        axi_t tmp4 = s_embedding_0.read();
        axi_t tmp5 = s_embedding_0.read();

        axi_t tmp6 = s_embedding_1.read();
        axi_t tmp7 = s_embedding_1.read();
        axi_t tmp8 = s_embedding_1.read();
        axi_t tmp9 = s_embedding_1.read();
        axi_t tmp10 = s_embedding_1.read();
        axi_t tmp11 = s_embedding_1.read();
        
        network_t out0, out1, out2;

        out0.range(127, 0) = tmp0;
        out0.range(255, 128) = tmp1;
        out0.range(383, 256) = tmp2;
        out0.range(511, 384) = tmp3;

        out1.range(127, 0) = tmp4;
        out1.range(255, 128) = tmp5;
        out1.range(383, 256) = tmp6;
        out1.range(511, 384) = tmp7;

        out2.range(127, 0) = tmp8;
        out2.range(255, 128) = tmp9;
        out2.range(383, 256) = tmp10;
        out2.range(511, 384) = tmp11;

        s_gathered_embedding_0.write(out0);
        s_gathered_embedding_1.write(out1);
        s_gathered_embedding_2.write(out2);
    }
}

void group_1_embeddings_len4(
    hls::stream<axi_t>& s_embedding_0,
    hls::stream<network_t>& s_gathered_embedding,
    int batch_num
) {
    // Concatenate 1 embedding sub-vectors
    //     each stream: 4 x axi 128

    for (int i = 0; i < batch_num * BATCH_SIZE; i++) {
        #pragma HLS pipeline II=4

        axi_t tmp0 = s_embedding_0.read();
        axi_t tmp1 = s_embedding_0.read();
        axi_t tmp2 = s_embedding_0.read();
        axi_t tmp3 = s_embedding_0.read();
        
        network_t out;
        out.range(127, 0) = tmp0;
        out.range(255, 128) = tmp1;
        out.range(383, 256) = tmp2;
        out.range(511, 384) = tmp3;

        s_gathered_embedding.write(out);
    }
}

void group_1_embeddings_len12(
    hls::stream<axi_t>& s_embedding_0,
    hls::stream<network_t>& s_gathered_embedding_0,
    hls::stream<network_t>& s_gathered_embedding_1,
    hls::stream<network_t>& s_gathered_embedding_2,
    int batch_num
) {
    // Concatenate 1 embedding sub-vectors
    //     each stream: 12 x axi 128

    for (int i = 0; i < batch_num * BATCH_SIZE; i++) {
        #pragma HLS pipeline II=8

        axi_t tmp0 = s_embedding_0.read();
        axi_t tmp1 = s_embedding_0.read();
        axi_t tmp2 = s_embedding_0.read();
        axi_t tmp3 = s_embedding_0.read();
        
        network_t out0;
        out0.range(127, 0) = tmp0;
        out0.range(255, 128) = tmp1;
        out0.range(383, 256) = tmp2;
        out0.range(511, 384) = tmp3;

        s_gathered_embedding_0.write(out0);

        axi_t tmp4 = s_embedding_0.read();
        axi_t tmp5 = s_embedding_0.read();
        axi_t tmp6 = s_embedding_0.read();
        axi_t tmp7 = s_embedding_0.read();
        
        network_t out1;
        out1.range(127, 0) = tmp4;
        out1.range(255, 128) = tmp5;
        out1.range(383, 256) = tmp6;
        out1.range(511, 384) = tmp7;

        s_gathered_embedding_1.write(out1);

        axi_t tmp8 = s_embedding_0.read();
        axi_t tmp9 = s_embedding_0.read();
        axi_t tmp10 = s_embedding_0.read();
        axi_t tmp11 = s_embedding_0.read();
        
        network_t out2;
        out2.range(127, 0) = tmp8;
        out2.range(255, 128) = tmp9;
        out2.range(383, 256) = tmp10;
        out2.range(511, 384) = tmp11;

        s_gathered_embedding_2.write(out2);
    }
}

void gather_55_embedding_streams(
    hls::stream<network_t> (&s_embedding_gather_level_A)[55],
    hls::stream<network_t>& s_network, 
    int batch_num) {

    for (int i = 0; i < batch_num * BATCH_SIZE; i++) {

        for (int j = 0; j < 55; j++) {
            #pragma HLS pipeline II=1
            //printf("gather_55_embedding_streams, item_id = %d, stream_id = %d\n", i, j);
            s_network.write(s_embedding_gather_level_A[j].read());
        }
    }
}

void gather_embeddings(
    hls::stream<hbm_t>& s_embedding_buffer_HBM0, hls::stream<hbm_t>& s_embedding_buffer_HBM1, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM2, hls::stream<hbm_t>& s_embedding_buffer_HBM3, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM4, hls::stream<hbm_t>& s_embedding_buffer_HBM5, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM6, hls::stream<hbm_t>& s_embedding_buffer_HBM7, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM8, hls::stream<hbm_t>& s_embedding_buffer_HBM9, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM10, hls::stream<hbm_t>& s_embedding_buffer_HBM11, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM12, hls::stream<hbm_t>& s_embedding_buffer_HBM13, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM14, hls::stream<hbm_t>& s_embedding_buffer_HBM15, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM16, hls::stream<hbm_t>& s_embedding_buffer_HBM17, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM18, hls::stream<hbm_t>& s_embedding_buffer_HBM19, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM20, hls::stream<hbm_t>& s_embedding_buffer_HBM21, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM22, hls::stream<hbm_t>& s_embedding_buffer_HBM23, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM24, hls::stream<hbm_t>& s_embedding_buffer_HBM25, 
    hls::stream<hbm_t>& s_embedding_buffer_HBM26, hls::stream<hbm_t>& s_embedding_buffer_HBM27, 
    
    hls::stream<ddr_t>& s_embedding_buffer_DDR0, hls::stream<ddr_t>& s_embedding_buffer_DDR1,

    hls::stream<plram_t>& s_embedding_buffer_PLRAM0, hls::stream<plram_t>& s_embedding_buffer_PLRAM1, 
    hls::stream<plram_t>& s_embedding_buffer_PLRAM2, hls::stream<plram_t>& s_embedding_buffer_PLRAM3,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM4, hls::stream<plram_t>& s_embedding_buffer_PLRAM5,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM6, hls::stream<plram_t>& s_embedding_buffer_PLRAM7,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM8, hls::stream<plram_t>& s_embedding_buffer_PLRAM9,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM10, hls::stream<plram_t>& s_embedding_buffer_PLRAM11,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM12, hls::stream<plram_t>& s_embedding_buffer_PLRAM13,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM14, hls::stream<plram_t>& s_embedding_buffer_PLRAM15,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM16, hls::stream<plram_t>& s_embedding_buffer_PLRAM17,
    hls::stream<plram_t>& s_embedding_buffer_PLRAM18,
    
    hls::stream<network_t>& s_network,
    int batch_num
) {
#pragma HLS inline

    /// NOTE THAT THIS FUNCTION IS ONLY APPLICABLE TO THE CURRENT TABLE SIZES ///
    /// THIS IS NOT A GENERALIZED GATHER FUNCTION, IT IS DATA DEPENDENT ///

// #define AXI_PADDED_SIZE_PLRAM_0 1
// #define AXI_PADDED_SIZE_PLRAM_1 1
// #define AXI_PADDED_SIZE_PLRAM_2 1
// #define AXI_PADDED_SIZE_PLRAM_3 1
// #define AXI_PADDED_SIZE_PLRAM_4 1
// #define AXI_PADDED_SIZE_PLRAM_5 1
// #define AXI_PADDED_SIZE_PLRAM_6 1
// #define AXI_PADDED_SIZE_PLRAM_7 1
// #define AXI_PADDED_SIZE_PLRAM_8 1
// #define AXI_PADDED_SIZE_PLRAM_9 1
// #define AXI_PADDED_SIZE_PLRAM_10 1
// #define AXI_PADDED_SIZE_PLRAM_11 1
// #define AXI_PADDED_SIZE_PLRAM_12 1
// #define AXI_PADDED_SIZE_PLRAM_13 1
// #define AXI_PADDED_SIZE_PLRAM_14 1
// #define AXI_PADDED_SIZE_PLRAM_15 1
// #define AXI_PADDED_SIZE_PLRAM_16 1
// #define AXI_PADDED_SIZE_PLRAM_17 1
// #define AXI_PADDED_SIZE_PLRAM_18 1

// #define AXI_PADDED_SIZE_PLRAM_19 1
// #define AXI_PADDED_SIZE_PLRAM_20 1
// #define AXI_PADDED_SIZE_PLRAM_21 1
// #define AXI_PADDED_SIZE_PLRAM_22 1
// #define AXI_PADDED_SIZE_PLRAM_23 1
// #define AXI_PADDED_SIZE_PLRAM_24 1
// #define AXI_PADDED_SIZE_PLRAM_25 1
// #define AXI_PADDED_SIZE_PLRAM_26 1 -> 26 - 19 = 7, plram  0~7: 2 axi<128>

// #define AXI_PADDED_SIZE_PLRAM_27 2
// #define AXI_PADDED_SIZE_PLRAM_28 2
// #define AXI_PADDED_SIZE_PLRAM_29 2
// #define AXI_PADDED_SIZE_PLRAM_30 2
// #define AXI_PADDED_SIZE_PLRAM_31 2
// #define AXI_PADDED_SIZE_PLRAM_32 2
// #define AXI_PADDED_SIZE_PLRAM_33 2
// #define AXI_PADDED_SIZE_PLRAM_34 2
// #define AXI_PADDED_SIZE_PLRAM_35 2
// #define AXI_PADDED_SIZE_PLRAM_36 2
// #define AXI_PADDED_SIZE_PLRAM_37 2 -> plram 8 ~ 18, 3 x axi <128>

// #define AXI_PADDED_SIZE_HBM_0 2
// #define AXI_PADDED_SIZE_HBM_1 2
// #define AXI_PADDED_SIZE_HBM_2 2
// #define AXI_PADDED_SIZE_HBM_3 2
// #define AXI_PADDED_SIZE_HBM_4 2
// #define AXI_PADDED_SIZE_HBM_5 2
// #define AXI_PADDED_SIZE_HBM_6 2
// #define AXI_PADDED_SIZE_HBM_7 2
// #define AXI_PADDED_SIZE_HBM_8 2
// #define AXI_PADDED_SIZE_HBM_9 2
// #define AXI_PADDED_SIZE_HBM_10 2
// #define AXI_PADDED_SIZE_HBM_11 2
// #define AXI_PADDED_SIZE_HBM_12 2
// #define AXI_PADDED_SIZE_HBM_13 2
// #define AXI_PADDED_SIZE_HBM_14 2
// #define AXI_PADDED_SIZE_HBM_15 2
// #define AXI_PADDED_SIZE_HBM_16 2
// #define AXI_PADDED_SIZE_HBM_17 2
// #define AXI_PADDED_SIZE_HBM_18 2
// #define AXI_PADDED_SIZE_HBM_19 2
// #define AXI_PADDED_SIZE_HBM_20 2
// #define AXI_PADDED_SIZE_HBM_21 2
// #define AXI_PADDED_SIZE_HBM_22 2
// #define AXI_PADDED_SIZE_HBM_23 2
// #define AXI_PADDED_SIZE_HBM_24 2
// #define AXI_PADDED_SIZE_HBM_25 2
// #define AXI_PADDED_SIZE_HBM_26 2
// #define AXI_PADDED_SIZE_HBM_27 2

// #define AXI_PADDED_SIZE_HBM_28 2
// #define AXI_PADDED_SIZE_HBM_29 2
// #define AXI_PADDED_SIZE_HBM_30 2
// #define AXI_PADDED_SIZE_HBM_31 2
// #define AXI_PADDED_SIZE_HBM_32 2
// #define AXI_PADDED_SIZE_HBM_33 2
// #define AXI_PADDED_SIZE_HBM_34 2
// #define AXI_PADDED_SIZE_HBM_35 2
// #define AXI_PADDED_SIZE_HBM_36 2
// #define AXI_PADDED_SIZE_HBM_37 2
// #define AXI_PADDED_SIZE_HBM_38 2 -> 38-28=10, hbm 0~10 -> 4 x axi 128

// #define AXI_PADDED_SIZE_HBM_39 4
// #define AXI_PADDED_SIZE_HBM_40 4
// #define AXI_PADDED_SIZE_HBM_41 4
// #define AXI_PADDED_SIZE_HBM_42 4
// #define AXI_PADDED_SIZE_HBM_43 4
// #define AXI_PADDED_SIZE_HBM_44 4
// #define AXI_PADDED_SIZE_HBM_45 4
// #define AXI_PADDED_SIZE_HBM_46 4
// #define AXI_PADDED_SIZE_HBM_47 4
// #define AXI_PADDED_SIZE_HBM_48 4
// #define AXI_PADDED_SIZE_HBM_49 4
// #define AXI_PADDED_SIZE_HBM_50 4
// #define AXI_PADDED_SIZE_HBM_51 4
// #define AXI_PADDED_SIZE_HBM_52 4
// #define AXI_PADDED_SIZE_HBM_53 4
// #define AXI_PADDED_SIZE_HBM_54 4
// #define AXI_PADDED_SIZE_HBM_55 4 -> hbm 11~27 ->  6 x axi 128

// #define AXI_PADDED_SIZE_DDR_0 4
// #define AXI_PADDED_SIZE_DDR_1 4
// #define AXI_PADDED_SIZE_DDR_2 8
// #define AXI_PADDED_SIZE_DDR_3 8 -> dram 0~1 -> 12 x axi 128

    // 8 x 2 + 11 x 3 + 11 x 4 + 17 x 6 + 2 x 12 + 1 (pad)
    // plram 0~7 -> 8 x 2
    // plram 8~18 -> 11 x 3 = 8 x 3 + 3 x 3 
    // hbm 0~10 -> 11 x 4
    // hbm 11~27 -> 17 x 6 = 16 x 6 + 1 x 6
    // ddr 0~1 -> 2 x 12
    // padding: plram 16~18 (3x3) + hbm 27 (1x6) + 1 = 16 x axi 128

    // 880 (padded to 876) * 4 * 8 / 512 = 55
    hls::stream<network_t> s_embedding_gather_level_A[55];
#pragma HLS stream variable=s_embedding_gather_level_A depth=2
#pragma HLS array_partition variable=s_embedding_gather_level_A dim=1 complete
#pragma HLS resource variable=s_embedding_gather_level_A core=FIFO_SRL

    // plram 0~7 -> 8 x 2
    group_2_embeddings_len_2(
        s_embedding_buffer_PLRAM0, s_embedding_buffer_PLRAM1,
        s_embedding_gather_level_A[0], batch_num);
    group_2_embeddings_len_2(
        s_embedding_buffer_PLRAM2, s_embedding_buffer_PLRAM3,
        s_embedding_gather_level_A[1], batch_num);
    group_2_embeddings_len_2(
        s_embedding_buffer_PLRAM4, s_embedding_buffer_PLRAM5,
        s_embedding_gather_level_A[2], batch_num);
    group_2_embeddings_len_2(
        s_embedding_buffer_PLRAM6, s_embedding_buffer_PLRAM7,
        s_embedding_gather_level_A[3], batch_num);

    // plram 8~15 -> 8 x 3
    group_4_embeddings_len3(
        s_embedding_buffer_PLRAM8, s_embedding_buffer_PLRAM9,
        s_embedding_buffer_PLRAM10, s_embedding_buffer_PLRAM11,
        s_embedding_gather_level_A[4],
        s_embedding_gather_level_A[5],
        s_embedding_gather_level_A[6], batch_num); 
    group_4_embeddings_len3(
        s_embedding_buffer_PLRAM12, s_embedding_buffer_PLRAM13,
        s_embedding_buffer_PLRAM14, s_embedding_buffer_PLRAM15,
        s_embedding_gather_level_A[7],
        s_embedding_gather_level_A[8],
        s_embedding_gather_level_A[9], batch_num); 

    // plram 16~18: 3 x 3 + hbm 27 (1x6)
    group_4_embeddings_padded(
        s_embedding_buffer_PLRAM16, s_embedding_buffer_PLRAM17,
        s_embedding_buffer_PLRAM18, s_embedding_buffer_HBM27,
        s_embedding_gather_level_A[10],
        s_embedding_gather_level_A[11],
        s_embedding_gather_level_A[12],
        s_embedding_gather_level_A[13], 
        batch_num);

    // hbm 0~10 -> 11 x 4
    group_1_embeddings_len4(
        s_embedding_buffer_HBM0, s_embedding_gather_level_A[14], batch_num);
    group_1_embeddings_len4(
        s_embedding_buffer_HBM1, s_embedding_gather_level_A[15], batch_num);
    group_1_embeddings_len4(
        s_embedding_buffer_HBM2, s_embedding_gather_level_A[16], batch_num);
    group_1_embeddings_len4(
        s_embedding_buffer_HBM3, s_embedding_gather_level_A[17], batch_num);
    group_1_embeddings_len4(
        s_embedding_buffer_HBM4, s_embedding_gather_level_A[18], batch_num);
    group_1_embeddings_len4(
        s_embedding_buffer_HBM5, s_embedding_gather_level_A[19], batch_num);
    group_1_embeddings_len4(
        s_embedding_buffer_HBM6, s_embedding_gather_level_A[20], batch_num);
    group_1_embeddings_len4(
        s_embedding_buffer_HBM7, s_embedding_gather_level_A[21], batch_num);
    group_1_embeddings_len4(
        s_embedding_buffer_HBM8, s_embedding_gather_level_A[22], batch_num);
    group_1_embeddings_len4(
        s_embedding_buffer_HBM9, s_embedding_gather_level_A[23], batch_num);
    group_1_embeddings_len4(
        s_embedding_buffer_HBM10, s_embedding_gather_level_A[24], batch_num);

    // hbm 11~26 -> 16 x 6
    group_2_embeddings_len_6(
        s_embedding_buffer_HBM11, s_embedding_buffer_HBM12,
        s_embedding_gather_level_A[25],
        s_embedding_gather_level_A[26],
        s_embedding_gather_level_A[27], batch_num);
    group_2_embeddings_len_6(
        s_embedding_buffer_HBM13, s_embedding_buffer_HBM14,
        s_embedding_gather_level_A[28],
        s_embedding_gather_level_A[29],
        s_embedding_gather_level_A[30], batch_num);
    group_2_embeddings_len_6(
        s_embedding_buffer_HBM15, s_embedding_buffer_HBM16,
        s_embedding_gather_level_A[31],
        s_embedding_gather_level_A[32],
        s_embedding_gather_level_A[33], batch_num);
    group_2_embeddings_len_6(
        s_embedding_buffer_HBM17, s_embedding_buffer_HBM18,
        s_embedding_gather_level_A[34],
        s_embedding_gather_level_A[35],
        s_embedding_gather_level_A[36], batch_num);
    group_2_embeddings_len_6(
        s_embedding_buffer_HBM19, s_embedding_buffer_HBM20,
        s_embedding_gather_level_A[37],
        s_embedding_gather_level_A[38],
        s_embedding_gather_level_A[39], batch_num);
    group_2_embeddings_len_6(
        s_embedding_buffer_HBM21, s_embedding_buffer_HBM22,
        s_embedding_gather_level_A[40],
        s_embedding_gather_level_A[41],
        s_embedding_gather_level_A[42], batch_num);
    group_2_embeddings_len_6(
        s_embedding_buffer_HBM23, s_embedding_buffer_HBM24,
        s_embedding_gather_level_A[43],
        s_embedding_gather_level_A[44],
        s_embedding_gather_level_A[45], batch_num);
    group_2_embeddings_len_6(
        s_embedding_buffer_HBM25, s_embedding_buffer_HBM26,
        s_embedding_gather_level_A[46],
        s_embedding_gather_level_A[47],
        s_embedding_gather_level_A[48], batch_num);

    // ddr 0~1 -> 2 x 12
    group_1_embeddings_len12(
        s_embedding_buffer_DDR0,
        s_embedding_gather_level_A[49],
        s_embedding_gather_level_A[50],
        s_embedding_gather_level_A[51], batch_num); 
    group_1_embeddings_len12(
        s_embedding_buffer_DDR1,
        s_embedding_gather_level_A[52],
        s_embedding_gather_level_A[53],
        s_embedding_gather_level_A[54], batch_num); 

    // gather all 22 streams into 1
    gather_55_embedding_streams(s_embedding_gather_level_A, s_network, batch_num);
}

void consume_and_write(
    // 22 = 352 (input feature len) * 4 * 8 / 512 bit
    hls::stream<network_t>& s_network, network_t* out_RAM, int batch_num
) {
    // consume all gathered embedding vectors, only write the last batch
    network_t out_local[BATCH_SIZE * INPUT_SIZE_AXI_512];

    for (int i = 0 ; i < batch_num; i++){
        for (int j = 0; j < BATCH_SIZE; j++) {
            for (int k = 0; k < INPUT_SIZE_AXI_512; k++) {
                #pragma HLS pipeline II=1
                //printf("consume_and_write: i = %d j = %d k = %d\n", i, j, k);
                out_local[j * INPUT_SIZE_AXI_512 + k] = s_network.read();
            }
        }
    }
    for (int k = 0; k < BATCH_SIZE * INPUT_SIZE_AXI_512; k++) {
        #pragma HLS pipeline II=1
        out_RAM[k] = out_local[k];
    }
}
