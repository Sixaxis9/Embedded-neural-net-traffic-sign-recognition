/**
  ******************************************************************************
  * @file    tflite.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Tue May 26 21:56:56 2020
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2018 STMicroelectronics.
  * All rights reserved.
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */


#include "tflite.h"

#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "layers.h"

#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#define AI_TOOLS_VERSION_MAJOR 5
#define AI_TOOLS_VERSION_MINOR 0
#define AI_TOOLS_VERSION_MICRO 0


#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#define AI_TOOLS_API_VERSION_MAJOR 1
#define AI_TOOLS_API_VERSION_MINOR 3
#define AI_TOOLS_API_VERSION_MICRO 0

#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_tflite
 
#undef AI_TFLITE_MODEL_SIGNATURE
#define AI_TFLITE_MODEL_SIGNATURE     "f3c2876f7422ac65016e9e91c0f2af0e"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     "(rev-5.0.0)"
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Tue May 26 21:56:56 2020"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_TFLITE_N_BATCHES
#define AI_TFLITE_N_BATCHES         (1)

/**  Forward network declaration section  *************************************/
AI_STATIC ai_network AI_NET_OBJ_INSTANCE;


/**  Forward network array declarations  **************************************/
AI_STATIC ai_array conv2d_8_scratch1_array;   /* Array #0 */
AI_STATIC ai_array conv2d_8_scratch0_array;   /* Array #1 */
AI_STATIC ai_array conv2d_4_scratch1_array;   /* Array #2 */
AI_STATIC ai_array conv2d_4_scratch0_array;   /* Array #3 */
AI_STATIC ai_array conv2d_0_scratch1_array;   /* Array #4 */
AI_STATIC ai_array conv2d_0_scratch0_array;   /* Array #5 */
AI_STATIC ai_array dense_13_bias_array;   /* Array #6 */
AI_STATIC ai_array dense_13_weights_array;   /* Array #7 */
AI_STATIC ai_array dense_12_bias_array;   /* Array #8 */
AI_STATIC ai_array dense_12_weights_array;   /* Array #9 */
AI_STATIC ai_array input_11_array;   /* Array #10 */
AI_STATIC ai_array input_10_array;   /* Array #11 */
AI_STATIC ai_array conv2d_8_bias_array;   /* Array #12 */
AI_STATIC ai_array conv2d_8_weights_array;   /* Array #13 */
AI_STATIC ai_array input_7_array;   /* Array #14 */
AI_STATIC ai_array input_6_array;   /* Array #15 */
AI_STATIC ai_array conv2d_4_bias_array;   /* Array #16 */
AI_STATIC ai_array conv2d_4_weights_array;   /* Array #17 */
AI_STATIC ai_array input_3_array;   /* Array #18 */
AI_STATIC ai_array input_2_array;   /* Array #19 */
AI_STATIC ai_array conv2d_0_bias_array;   /* Array #20 */
AI_STATIC ai_array conv2d_0_weights_array;   /* Array #21 */
AI_STATIC ai_array input_0_output_array;   /* Array #22 */
AI_STATIC ai_array conv2d_0_output_array;   /* Array #23 */
AI_STATIC ai_array pool_1_fmt_output_array;   /* Array #24 */
AI_STATIC ai_array eltwise_2_output_array;   /* Array #25 */
AI_STATIC ai_array eltwise_2_fmt_output_array;   /* Array #26 */
AI_STATIC ai_array eltwise_3_output_array;   /* Array #27 */
AI_STATIC ai_array conv2d_4_output_array;   /* Array #28 */
AI_STATIC ai_array pool_5_fmt_output_array;   /* Array #29 */
AI_STATIC ai_array eltwise_6_output_array;   /* Array #30 */
AI_STATIC ai_array eltwise_6_fmt_output_array;   /* Array #31 */
AI_STATIC ai_array eltwise_7_output_array;   /* Array #32 */
AI_STATIC ai_array conv2d_8_output_array;   /* Array #33 */
AI_STATIC ai_array pool_9_fmt_output_array;   /* Array #34 */
AI_STATIC ai_array eltwise_10_output_array;   /* Array #35 */
AI_STATIC ai_array eltwise_10_fmt_output_array;   /* Array #36 */
AI_STATIC ai_array eltwise_11_output_array;   /* Array #37 */
AI_STATIC ai_array dense_12_output_array;   /* Array #38 */
AI_STATIC ai_array dense_13_output_array;   /* Array #39 */
AI_STATIC ai_array dense_13_fmt_output_array;   /* Array #40 */
AI_STATIC ai_array nl_14_output_array;   /* Array #41 */
AI_STATIC ai_array nl_14_fmt_output_array;   /* Array #42 */


/**  Forward network tensor declarations  *************************************/
AI_STATIC ai_tensor conv2d_8_scratch1;   /* Tensor #0 */
AI_STATIC ai_tensor conv2d_8_scratch0;   /* Tensor #1 */
AI_STATIC ai_tensor conv2d_4_scratch1;   /* Tensor #2 */
AI_STATIC ai_tensor conv2d_4_scratch0;   /* Tensor #3 */
AI_STATIC ai_tensor conv2d_0_scratch1;   /* Tensor #4 */
AI_STATIC ai_tensor conv2d_0_scratch0;   /* Tensor #5 */
AI_STATIC ai_tensor dense_13_bias;   /* Tensor #6 */
AI_STATIC ai_tensor dense_13_weights;   /* Tensor #7 */
AI_STATIC ai_tensor dense_12_bias;   /* Tensor #8 */
AI_STATIC ai_tensor dense_12_weights;   /* Tensor #9 */
AI_STATIC ai_tensor input_11;   /* Tensor #10 */
AI_STATIC ai_tensor input_10;   /* Tensor #11 */
AI_STATIC ai_tensor conv2d_8_bias;   /* Tensor #12 */
AI_STATIC ai_tensor conv2d_8_weights;   /* Tensor #13 */
AI_STATIC ai_tensor input_7;   /* Tensor #14 */
AI_STATIC ai_tensor input_6;   /* Tensor #15 */
AI_STATIC ai_tensor conv2d_4_bias;   /* Tensor #16 */
AI_STATIC ai_tensor conv2d_4_weights;   /* Tensor #17 */
AI_STATIC ai_tensor input_3;   /* Tensor #18 */
AI_STATIC ai_tensor input_2;   /* Tensor #19 */
AI_STATIC ai_tensor conv2d_0_bias;   /* Tensor #20 */
AI_STATIC ai_tensor conv2d_0_weights;   /* Tensor #21 */
AI_STATIC ai_tensor input_0_output;   /* Tensor #22 */
AI_STATIC ai_tensor conv2d_0_output;   /* Tensor #23 */
AI_STATIC ai_tensor pool_1_fmt_output;   /* Tensor #24 */
AI_STATIC ai_tensor eltwise_2_output;   /* Tensor #25 */
AI_STATIC ai_tensor eltwise_2_fmt_output;   /* Tensor #26 */
AI_STATIC ai_tensor eltwise_3_output;   /* Tensor #27 */
AI_STATIC ai_tensor conv2d_4_output;   /* Tensor #28 */
AI_STATIC ai_tensor pool_5_fmt_output;   /* Tensor #29 */
AI_STATIC ai_tensor eltwise_6_output;   /* Tensor #30 */
AI_STATIC ai_tensor eltwise_6_fmt_output;   /* Tensor #31 */
AI_STATIC ai_tensor eltwise_7_output;   /* Tensor #32 */
AI_STATIC ai_tensor conv2d_8_output;   /* Tensor #33 */
AI_STATIC ai_tensor pool_9_fmt_output;   /* Tensor #34 */
AI_STATIC ai_tensor eltwise_10_output;   /* Tensor #35 */
AI_STATIC ai_tensor eltwise_10_fmt_output;   /* Tensor #36 */
AI_STATIC ai_tensor eltwise_11_output;   /* Tensor #37 */
AI_STATIC ai_tensor dense_12_output;   /* Tensor #38 */
AI_STATIC ai_tensor dense_13_output;   /* Tensor #39 */
AI_STATIC ai_tensor dense_13_fmt_output;   /* Tensor #40 */
AI_STATIC ai_tensor nl_14_output;   /* Tensor #41 */
AI_STATIC ai_tensor nl_14_fmt_output;   /* Tensor #42 */


/**  Forward network tensor chain declarations  *******************************/
AI_STATIC_CONST ai_tensor_chain conv2d_0_chain;   /* Chain #0 */
AI_STATIC_CONST ai_tensor_chain pool_1_fmt_chain;   /* Chain #1 */
AI_STATIC_CONST ai_tensor_chain eltwise_2_chain;   /* Chain #2 */
AI_STATIC_CONST ai_tensor_chain eltwise_2_fmt_chain;   /* Chain #3 */
AI_STATIC_CONST ai_tensor_chain eltwise_3_chain;   /* Chain #4 */
AI_STATIC_CONST ai_tensor_chain conv2d_4_chain;   /* Chain #5 */
AI_STATIC_CONST ai_tensor_chain pool_5_fmt_chain;   /* Chain #6 */
AI_STATIC_CONST ai_tensor_chain eltwise_6_chain;   /* Chain #7 */
AI_STATIC_CONST ai_tensor_chain eltwise_6_fmt_chain;   /* Chain #8 */
AI_STATIC_CONST ai_tensor_chain eltwise_7_chain;   /* Chain #9 */
AI_STATIC_CONST ai_tensor_chain conv2d_8_chain;   /* Chain #10 */
AI_STATIC_CONST ai_tensor_chain pool_9_fmt_chain;   /* Chain #11 */
AI_STATIC_CONST ai_tensor_chain eltwise_10_chain;   /* Chain #12 */
AI_STATIC_CONST ai_tensor_chain eltwise_10_fmt_chain;   /* Chain #13 */
AI_STATIC_CONST ai_tensor_chain eltwise_11_chain;   /* Chain #14 */
AI_STATIC_CONST ai_tensor_chain dense_12_chain;   /* Chain #15 */
AI_STATIC_CONST ai_tensor_chain dense_13_chain;   /* Chain #16 */
AI_STATIC_CONST ai_tensor_chain dense_13_fmt_chain;   /* Chain #17 */
AI_STATIC_CONST ai_tensor_chain nl_14_chain;   /* Chain #18 */
AI_STATIC_CONST ai_tensor_chain nl_14_fmt_chain;   /* Chain #19 */


/**  Forward network layer declarations  **************************************/
AI_STATIC ai_layer_conv2d_nl_pool conv2d_0_layer; /* Layer #0 */
AI_STATIC ai_layer_nl pool_1_fmt_layer; /* Layer #1 */
AI_STATIC ai_layer_eltwise eltwise_2_layer; /* Layer #2 */
AI_STATIC ai_layer_nl eltwise_2_fmt_layer; /* Layer #3 */
AI_STATIC ai_layer_eltwise eltwise_3_layer; /* Layer #4 */
AI_STATIC ai_layer_conv2d_nl_pool conv2d_4_layer; /* Layer #5 */
AI_STATIC ai_layer_nl pool_5_fmt_layer; /* Layer #6 */
AI_STATIC ai_layer_eltwise eltwise_6_layer; /* Layer #7 */
AI_STATIC ai_layer_nl eltwise_6_fmt_layer; /* Layer #8 */
AI_STATIC ai_layer_eltwise eltwise_7_layer; /* Layer #9 */
AI_STATIC ai_layer_conv2d_nl_pool conv2d_8_layer; /* Layer #10 */
AI_STATIC ai_layer_nl pool_9_fmt_layer; /* Layer #11 */
AI_STATIC ai_layer_eltwise eltwise_10_layer; /* Layer #12 */
AI_STATIC ai_layer_nl eltwise_10_fmt_layer; /* Layer #13 */
AI_STATIC ai_layer_eltwise eltwise_11_layer; /* Layer #14 */
AI_STATIC ai_layer_dense dense_12_layer; /* Layer #15 */
AI_STATIC ai_layer_dense dense_13_layer; /* Layer #16 */
AI_STATIC ai_layer_nl dense_13_fmt_layer; /* Layer #17 */
AI_STATIC ai_layer_nl nl_14_layer; /* Layer #18 */
AI_STATIC ai_layer_nl nl_14_fmt_layer; /* Layer #19 */


/**  Array declarations section  **********************************************/
AI_ARRAY_OBJ_DECLARE(
    conv2d_8_scratch1_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 400,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_8_scratch0_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 4600,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_4_scratch1_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 1000,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_4_scratch0_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 1340,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_0_scratch1_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 520,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_0_scratch0_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 336,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_13_bias_array, AI_ARRAY_FORMAT_S32,
    NULL, NULL, 43,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_13_weights_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 4300,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_12_bias_array, AI_ARRAY_FORMAT_S32,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_12_weights_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 10000,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    input_11_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    input_10_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_8_bias_array, AI_ARRAY_FORMAT_S32,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_8_weights_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 80000,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    input_7_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 50,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    input_6_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 50,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_4_bias_array, AI_ARRAY_FORMAT_S32,
    NULL, NULL, 50,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_4_weights_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 8000,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    input_3_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 10,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    input_2_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 10,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_0_bias_array, AI_ARRAY_FORMAT_S32,
    NULL, NULL, 10,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_0_weights_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 490,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    input_0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
    NULL, NULL, 1024,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_0_output_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 1690,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    pool_1_fmt_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1690,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    eltwise_2_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1690,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    eltwise_2_fmt_output_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 1690,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    eltwise_3_output_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 1690,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_4_output_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 1250,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    pool_5_fmt_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1250,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    eltwise_6_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1250,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    eltwise_6_fmt_output_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 1250,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    eltwise_7_output_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 1250,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_8_output_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    pool_9_fmt_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    eltwise_10_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    eltwise_10_fmt_output_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    eltwise_11_output_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_12_output_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_13_output_array, AI_ARRAY_FORMAT_S8,
    NULL, NULL, 43,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_13_fmt_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 43,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    nl_14_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 43,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    nl_14_fmt_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
    NULL, NULL, 43,
     AI_STATIC)


AI_STATIC ai_intq_info_list conv2d_8_scratch1_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14130298793315887f), AI_PACK_INTQ_ZP(-128)));   /* Int quant #0 */
AI_STATIC ai_intq_info_list conv2d_4_scratch1_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04002319648861885f), AI_PACK_INTQ_ZP(-128)));   /* Int quant #1 */
AI_STATIC ai_intq_info_list conv2d_0_scratch1_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004687373526394367f), AI_PACK_INTQ_ZP(-128)));   /* Int quant #2 */
AI_STATIC ai_intq_info_list dense_13_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005653419066220522f), AI_PACK_INTQ_ZP(0)));   /* Int quant #3 */
AI_STATIC ai_intq_info_list dense_13_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0045748548582196236f), AI_PACK_INTQ_ZP(0)));   /* Int quant #4 */
AI_STATIC ai_intq_info_list dense_12_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0003414466045796871f), AI_PACK_INTQ_ZP(0)));   /* Int quant #5 */
AI_STATIC ai_intq_info_list dense_12_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004106895532459021f), AI_PACK_INTQ_ZP(0)));   /* Int quant #6 */
AI_STATIC ai_intq_info_list input_11_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012831468135118484f), AI_PACK_INTQ_ZP(0)));   /* Int quant #7 */
AI_STATIC ai_intq_info_list conv2d_8_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 100, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0002110304485540837f, 0.00017544557340443134f, 0.00021164240024518222f, 0.0001799788005882874f, 0.00028332570218481123f, 0.00019792724924627692f, 0.0002022225526161492f, 0.00017952856433112174f, 0.00018272356828674674f, 0.00019750332285184413f, 0.00019009657262358814f, 0.0001979062071768567f, 0.0002049829054158181f, 0.00021190661936998367f, 0.0001806896907510236f, 0.00021345022832974792f, 0.00018119240121450275f, 0.0001394928403897211f, 0.00019115071336273104f, 0.00020241258607711643f, 0.00018870970234274864f, 0.0001677994878264144f, 0.0001924002863233909f, 0.00022277138486970216f, 0.000184636955964379f, 0.0001899402996059507f, 0.00018855059170164168f, 0.0001905318786157295f, 0.00026187513140030205f, 0.0001975083869183436f, 0.00017371997819282115f, 0.00018009320774581283f, 0.0002183877950301394f, 0.00017979428230319172f, 0.0002149180363630876f, 0.00018018823175225407f, 0.00023872635210864246f, 0.00016847210645209998f, 0.00021008503972552717f, 0.00019040692131966352f, 0.00018786972214002162f, 0.00022214307682588696f, 0.0002572684024926275f, 0.00020924147975165397f, 0.00017221708549186587f, 0.00021153681154828519f, 0.00020205559849273413f, 0.00017921521794050932f, 0.00024977224529720843f, 0.00017253134865313768f, 0.00017373404989484698f, 0.00020094549108762294f, 0.00022041439660824835f, 0.000182239746209234f, 0.00020668748766183853f, 0.00019166698621120304f, 0.0002474328503012657f, 0.00018212074064649642f, 0.00022398913279175758f, 0.00023818621411919594f, 0.00025716013624332845f, 0.00018328832811675966f, 0.000191867642570287f, 0.0002516163804102689f, 0.00018192997958976775f, 0.00023640987637918442f, 0.00018790991452988237f, 0.00021480956638697535f, 0.00021507570636458695f, 0.00018804390856530517f, 0.0002086551976390183f, 0.00022531762078870088f, 0.00020979635883122683f, 0.00018529828230384737f, 0.00021793972700834274f, 0.00019026869267690927f, 0.00017688731895759702f, 0.0002469460014253855f, 0.00017358059994876385f, 0.00019883556524291635f, 0.00019608480215538293f, 0.0001925434044096619f, 0.000225125317228958f, 0.00018107634969055653f, 0.00018784786516334862f, 0.00019063234503846616f, 0.0001725732145132497f, 0.00019286639872007072f, 0.00020768196554854512f, 0.00020939311070833355f, 0.00018044296302832663f, 0.0001986209535971284f, 0.00018754742632154375f, 0.0001841980847530067f, 0.00022788307978771627f, 0.00021540159650612622f, 0.00020728939853142947f, 0.0002288944087922573f, 0.00021057247067801654f, 0.0001781405444489792f), AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));   /* Int quant #8 */
AI_STATIC ai_intq_info_list conv2d_8_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 100, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002161702373996377f, 0.0017971866764128208f, 0.00216797087341547f, 0.0018436230020597577f, 0.002902262844145298f, 0.002027479000389576f, 0.002071478171274066f, 0.00183901097625494f, 0.0018717391649261117f, 0.0020231364760547876f, 0.0019472650019451976f, 0.0020272633992135525f, 0.0020997540559619665f, 0.0021706775296479464f, 0.001850905129685998f, 0.002186489524319768f, 0.0018560546450316906f, 0.0014289028476923704f, 0.001958063105121255f, 0.002073424868285656f, 0.0019330584909766912f, 0.0017188636120408773f, 0.00197086320258677f, 0.002281971275806427f, 0.00189133919775486f, 0.0019456641748547554f, 0.0019314286764711142f, 0.0019517240580171347f, 0.0026825328823179007f, 0.0020231883972883224f, 0.0017795104067772627f, 0.0018447949551045895f, 0.0022370677907019854f, 0.0018417328828945756f, 0.002201525028795004f, 0.0018457683036103845f, 0.0024454069789499044f, 0.0017257535364478827f, 0.002152018016204238f, 0.0019504440715536475f, 0.0019244541181251407f, 0.002275535138323903f, 0.0026353434659540653f, 0.002143376972526312f, 0.001764115528203547f, 0.002166889375075698f, 0.0020697680301964283f, 0.0018358011730015278f, 0.0025585561525076628f, 0.0017673346446827054f, 0.0017796546453610063f, 0.0020583965815603733f, 0.0022578274365514517f, 0.0018667832482606173f, 0.002117214957252145f, 0.001963351620361209f, 0.002534592291340232f, 0.0018655641470104456f, 0.0022944454103708267f, 0.002439873991534114f, 0.002634234493598342f, 0.0018775244243443012f, 0.0019654070492833853f, 0.002577446633949876f, 0.0018636101158335805f, 0.0024216780439019203f, 0.0019248658791184425f, 0.0022004139609634876f, 0.002203140174970031f, 0.001926238415762782f, 0.002137371338903904f, 0.002308053895831108f, 0.00214906083419919f, 0.001898113521747291f, 0.0022324780002236366f, 0.0019490281119942665f, 0.0018119552405551076f, 0.0025296052917838097f, 0.0017780826892703772f, 0.002036783378571272f, 0.0020086057484149933f, 0.0019723293371498585f, 0.0023060839157551527f, 0.0018548659281805158f, 0.0019242302514612675f, 0.0019527531694620848f, 0.0017677635187283158f, 0.00197563786059618f, 0.0021274019964039326f, 0.002144930185750127f, 0.0018483777530491352f, 0.002034584991633892f, 0.0019211526960134506f, 0.0018868434708565474f, 0.0023343332577496767f, 0.00220647850073874f, 0.0021233807783573866f, 0.002344692824408412f, 0.002157011069357395f, 0.001824792823754251f), AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));   /* Int quant #9 */
AI_STATIC ai_intq_info_list input_7_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0077092754654586315f), AI_PACK_INTQ_ZP(0)));   /* Int quant #10 */
AI_STATIC ai_intq_info_list conv2d_4_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 50, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(8.656024874653667e-05f, 0.000165149787790142f, 7.023826765362173e-05f, 7.459556945832446e-05f, 5.842583777848631e-05f, 8.877414802554995e-05f, 0.00010154784831684083f, 8.104198786895722e-05f, 6.910468073328957e-05f, 6.964628846617416e-05f, 8.63209497765638e-05f, 9.876787953544408e-05f, 7.154283957788721e-05f, 9.249738650396466e-05f, 9.776046499609947e-05f, 8.792220614850521e-05f, 0.00010654557263478637f, 7.480954809579998e-05f, 7.21180549589917e-05f, 0.00010470044071553275f, 8.882621477823704e-05f, 7.513206946896389e-05f, 7.334152178373188e-05f, 7.301241566892713e-05f, 8.738126052776352e-05f, 5.174620673642494e-05f, 0.0001142624460044317f, 8.018469816306606e-05f, 7.673617801629007e-05f, 5.864254853804596e-05f, 7.698105764575303e-05f, 7.258683763211593e-05f, 0.00010217228555120528f, 8.850820449879393e-05f, 8.442310354439542e-05f, 7.046343671390787e-05f, 8.934792276704684e-05f, 7.545347034465522e-05f, 0.00012713288015220314f, 9.161881462205201e-05f, 7.150152669055387e-05f, 6.782361015211791e-05f, 7.691072096349671e-05f, 7.861831545596942e-05f, 6.442734593292698e-05f, 6.414890958694741e-05f, 7.70420883782208e-05f, 0.00010547781857894734f, 7.778693543514237e-05f, 7.457687024725601e-05f), AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));   /* Int quant #11 */
AI_STATIC ai_intq_info_list conv2d_4_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 50, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002167004393413663f, 0.004134465008974075f, 0.0017583895241841674f, 0.0018674731254577637f, 0.001462669693864882f, 0.0022224285639822483f, 0.002542213536798954f, 0.0020288568921387196f, 0.00173001061193645f, 0.001743569620884955f, 0.002161013660952449f, 0.002472618129104376f, 0.001791049144230783f, 0.0023156385868787766f, 0.0024473979137837887f, 0.002201100578531623f, 0.002667329739779234f, 0.0018728298600763083f, 0.0018054493702948093f, 0.002621137537062168f, 0.0022237321827560663f, 0.0018809041939675808f, 0.0018360784742981195f, 0.0018278394127264619f, 0.002187558216974139f, 0.0012954475823789835f, 0.0028605188708752394f, 0.0020073947962373495f, 0.0019210624741390347f, 0.0014680949971079826f, 0.0019271929049864411f, 0.0018171851988881826f, 0.002557846251875162f, 0.002215770771726966f, 0.002113501774147153f, 0.001764026703312993f, 0.00223679281771183f, 0.0018889502389356494f, 0.0031827257480472326f, 0.0022936437744647264f, 0.0017900147940963507f, 0.0016979394713416696f, 0.0019254321232438087f, 0.0019681809935718775f, 0.001612915308214724f, 0.0016059447079896927f, 0.0019287208560854197f, 0.0026405989192426205f, 0.001947367680259049f, 0.0018670050194486976f), AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));   /* Int quant #12 */
AI_STATIC ai_intq_info_list input_3_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004939779173582792f), AI_PACK_INTQ_ZP(0)));   /* Int quant #13 */
AI_STATIC ai_intq_info_list conv2d_0_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 10, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(2.4666542230988853e-05f, 6.507856141979573e-06f, 5.557933036470786e-06f, 5.273053830023855e-06f, 5.119032266520662e-06f, 4.1389503167010844e-05f, 6.03050693825935e-06f, 7.719323548371904e-06f, 5.460772172227735e-06f, 5.225969744060421e-06f), AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));   /* Int quant #14 */
AI_STATIC ai_intq_info_list conv2d_0_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 10, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0062905969098210335f, 0.001659669098444283f, 0.0014174145180732012f, 0.0013447630917653441f, 0.0013054836308583617f, 0.010555378161370754f, 0.0015379328979179263f, 0.0019686243031173944f, 0.0013926360988989472f, 0.001332755433395505f), AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));   /* Int quant #15 */
AI_STATIC ai_intq_info_list input_0_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00392117677256465f), AI_PACK_INTQ_ZP(-128)));   /* Int quant #16 */
AI_STATIC ai_intq_info_list conv2d_0_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004687373526394367f), AI_PACK_INTQ_ZP(-128)));   /* Int quant #17 */
AI_STATIC ai_intq_info_list eltwise_2_fmt_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.038692228496074677f), AI_PACK_INTQ_ZP(-128)));   /* Int quant #18 */
AI_STATIC ai_intq_info_list eltwise_3_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03994465619325638f), AI_PACK_INTQ_ZP(-112)));   /* Int quant #19 */
AI_STATIC ai_intq_info_list conv2d_4_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04002319648861885f), AI_PACK_INTQ_ZP(-128)));   /* Int quant #20 */
AI_STATIC ai_intq_info_list eltwise_6_fmt_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09601124376058578f), AI_PACK_INTQ_ZP(-128)));   /* Int quant #21 */
AI_STATIC ai_intq_info_list eltwise_7_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0976223424077034f), AI_PACK_INTQ_ZP(-118)));   /* Int quant #22 */
AI_STATIC ai_intq_info_list conv2d_8_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14130298793315887f), AI_PACK_INTQ_ZP(-128)));   /* Int quant #23 */
AI_STATIC ai_intq_info_list eltwise_10_fmt_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08025788515806198f), AI_PACK_INTQ_ZP(-128)));   /* Int quant #24 */
AI_STATIC ai_intq_info_list eltwise_11_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0831398293375969f), AI_PACK_INTQ_ZP(-108)));   /* Int quant #25 */
AI_STATIC ai_intq_info_list dense_12_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12357591837644577f), AI_PACK_INTQ_ZP(-128)));   /* Int quant #26 */
AI_STATIC ai_intq_info_list dense_13_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5155251622200012f), AI_PACK_INTQ_ZP(-16)));   /* Int quant #27 */
AI_STATIC ai_intq_info_list nl_14_fmt_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f), AI_PACK_INTQ_ZP(-128)));   /* Int quant #28 */


/**  Tensor declarations section  *********************************************/
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_scratch1, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 2, 2), AI_STRIDE_INIT(4, 1, 1, 100, 200),
  1, &conv2d_8_scratch1_array, &conv2d_8_scratch1_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 4600, 1, 1), AI_STRIDE_INIT(4, 1, 1, 4600, 4600),
  1, &conv2d_8_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch1, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 10, 2), AI_STRIDE_INIT(4, 1, 1, 50, 500),
  1, &conv2d_4_scratch1_array, &conv2d_4_scratch1_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1340, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1340, 1340),
  1, &conv2d_4_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_scratch1, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 26, 2), AI_STRIDE_INIT(4, 1, 1, 10, 260),
  1, &conv2d_0_scratch1_array, &conv2d_0_scratch1_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 1, 1, 336, 336),
  1, &conv2d_0_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_13_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 43, 1, 1), AI_STRIDE_INIT(4, 4, 4, 172, 172),
  1, &dense_13_bias_array, &dense_13_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  dense_13_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 100, 43, 1, 1), AI_STRIDE_INIT(4, 1, 100, 4300, 4300),
  1, &dense_13_weights_array, &dense_13_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  dense_12_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &dense_12_bias_array, &dense_12_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  dense_12_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 100, 100, 1, 1), AI_STRIDE_INIT(4, 1, 100, 10000, 10000),
  1, &dense_12_weights_array, &dense_12_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  input_11, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &input_11_array, &input_11_intq)
AI_TENSOR_OBJ_DECLARE(
  input_10, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &input_10_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &conv2d_8_bias_array, &conv2d_8_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 50, 4, 4, 100), AI_STRIDE_INIT(4, 1, 50, 200, 800),
  1, &conv2d_8_weights_array, &conv2d_8_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  input_7, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 1, 1, 50, 50),
  1, &input_7_array, &input_7_intq)
AI_TENSOR_OBJ_DECLARE(
  input_6, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &input_6_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &conv2d_4_bias_array, &conv2d_4_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 10, 4, 4, 50), AI_STRIDE_INIT(4, 1, 10, 40, 160),
  1, &conv2d_4_weights_array, &conv2d_4_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  input_3, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 1, 1, 10, 10),
  1, &input_3_array, &input_3_intq)
AI_TENSOR_OBJ_DECLARE(
  input_2, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &input_2_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_0_bias_array, &conv2d_0_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 7, 7, 10), AI_STRIDE_INIT(4, 1, 1, 7, 49),
  1, &conv2d_0_weights_array, &conv2d_0_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  input_0_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1, 32, 32), AI_STRIDE_INIT(4, 1, 1, 1, 32),
  1, &input_0_output_array, &input_0_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 13, 13), AI_STRIDE_INIT(4, 1, 1, 10, 130),
  1, &conv2d_0_output_array, &conv2d_0_output_intq)
AI_TENSOR_OBJ_DECLARE(
  pool_1_fmt_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 13, 13), AI_STRIDE_INIT(4, 4, 4, 40, 520),
  1, &pool_1_fmt_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  eltwise_2_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 13, 13), AI_STRIDE_INIT(4, 4, 4, 40, 520),
  1, &eltwise_2_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  eltwise_2_fmt_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 13, 13), AI_STRIDE_INIT(4, 1, 1, 10, 130),
  1, &eltwise_2_fmt_output_array, &eltwise_2_fmt_output_intq)
AI_TENSOR_OBJ_DECLARE(
  eltwise_3_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 13, 13), AI_STRIDE_INIT(4, 1, 1, 10, 130),
  1, &eltwise_3_output_array, &eltwise_3_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 5, 5), AI_STRIDE_INIT(4, 1, 1, 50, 250),
  1, &conv2d_4_output_array, &conv2d_4_output_intq)
AI_TENSOR_OBJ_DECLARE(
  pool_5_fmt_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 5, 5), AI_STRIDE_INIT(4, 4, 4, 200, 1000),
  1, &pool_5_fmt_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  eltwise_6_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 5, 5), AI_STRIDE_INIT(4, 4, 4, 200, 1000),
  1, &eltwise_6_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  eltwise_6_fmt_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 5, 5), AI_STRIDE_INIT(4, 1, 1, 50, 250),
  1, &eltwise_6_fmt_output_array, &eltwise_6_fmt_output_intq)
AI_TENSOR_OBJ_DECLARE(
  eltwise_7_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 5, 5), AI_STRIDE_INIT(4, 1, 1, 50, 250),
  1, &eltwise_7_output_array, &eltwise_7_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &conv2d_8_output_array, &conv2d_8_output_intq)
AI_TENSOR_OBJ_DECLARE(
  pool_9_fmt_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &pool_9_fmt_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  eltwise_10_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &eltwise_10_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  eltwise_10_fmt_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &eltwise_10_fmt_output_array, &eltwise_10_fmt_output_intq)
AI_TENSOR_OBJ_DECLARE(
  eltwise_11_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &eltwise_11_output_array, &eltwise_11_output_intq)
AI_TENSOR_OBJ_DECLARE(
  dense_12_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &dense_12_output_array, &dense_12_output_intq)
AI_TENSOR_OBJ_DECLARE(
  dense_13_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 43, 1, 1), AI_STRIDE_INIT(4, 1, 1, 43, 43),
  1, &dense_13_output_array, &dense_13_output_intq)
AI_TENSOR_OBJ_DECLARE(
  dense_13_fmt_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 43, 1, 1), AI_STRIDE_INIT(4, 4, 4, 172, 172),
  1, &dense_13_fmt_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  nl_14_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 43, 1, 1), AI_STRIDE_INIT(4, 4, 4, 172, 172),
  1, &nl_14_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  nl_14_fmt_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 43, 1, 1), AI_STRIDE_INIT(4, 1, 1, 43, 43),
  1, &nl_14_fmt_output_array, &nl_14_fmt_output_intq)


/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&input_0_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_0_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_0_weights, &conv2d_0_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_0_scratch0, &conv2d_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_0_layer, 0,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool_integer,
  &AI_NET_OBJ_INSTANCE, &pool_1_fmt_layer, AI_STATIC,
  .tensors = &conv2d_0_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_integer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_1_fmt_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_0_output),
  AI_TENSOR_LIST_ENTRY(&pool_1_fmt_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_1_fmt_layer, 1,
  NL_TYPE,
  nl, node_convert,
  &AI_NET_OBJ_INSTANCE, &eltwise_2_layer, AI_STATIC,
  .tensors = &pool_1_fmt_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&pool_1_fmt_output, &input_2),
  AI_TENSOR_LIST_ENTRY(&eltwise_2_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_2_layer, 2,
  ELTWISE_TYPE,
  eltwise, forward_eltwise,
  &AI_NET_OBJ_INSTANCE, &eltwise_2_fmt_layer, AI_STATIC,
  .tensors = &eltwise_2_chain, 
  .operation = ai_mul, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_2_fmt_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&eltwise_2_output),
  AI_TENSOR_LIST_ENTRY(&eltwise_2_fmt_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_2_fmt_layer, 2,
  NL_TYPE,
  nl, node_convert,
  &AI_NET_OBJ_INSTANCE, &eltwise_3_layer, AI_STATIC,
  .tensors = &eltwise_2_fmt_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&eltwise_2_fmt_output, &input_3),
  AI_TENSOR_LIST_ENTRY(&eltwise_3_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_3_layer, 3,
  ELTWISE_TYPE,
  eltwise, forward_add_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_4_layer, AI_STATIC,
  .tensors = &eltwise_3_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&eltwise_3_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_4_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_4_weights, &conv2d_4_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_4_scratch0, &conv2d_4_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_layer, 4,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool_integer,
  &AI_NET_OBJ_INSTANCE, &pool_5_fmt_layer, AI_STATIC,
  .tensors = &conv2d_4_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_integer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_5_fmt_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_4_output),
  AI_TENSOR_LIST_ENTRY(&pool_5_fmt_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_5_fmt_layer, 5,
  NL_TYPE,
  nl, node_convert,
  &AI_NET_OBJ_INSTANCE, &eltwise_6_layer, AI_STATIC,
  .tensors = &pool_5_fmt_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&pool_5_fmt_output, &input_6),
  AI_TENSOR_LIST_ENTRY(&eltwise_6_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_6_layer, 6,
  ELTWISE_TYPE,
  eltwise, forward_eltwise,
  &AI_NET_OBJ_INSTANCE, &eltwise_6_fmt_layer, AI_STATIC,
  .tensors = &eltwise_6_chain, 
  .operation = ai_mul, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_6_fmt_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&eltwise_6_output),
  AI_TENSOR_LIST_ENTRY(&eltwise_6_fmt_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_6_fmt_layer, 6,
  NL_TYPE,
  nl, node_convert,
  &AI_NET_OBJ_INSTANCE, &eltwise_7_layer, AI_STATIC,
  .tensors = &eltwise_6_fmt_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&eltwise_6_fmt_output, &input_7),
  AI_TENSOR_LIST_ENTRY(&eltwise_7_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_7_layer, 7,
  ELTWISE_TYPE,
  eltwise, forward_add_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_8_layer, AI_STATIC,
  .tensors = &eltwise_7_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&eltwise_7_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_8_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_8_weights, &conv2d_8_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_8_scratch0, &conv2d_8_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_8_layer, 8,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool_integer,
  &AI_NET_OBJ_INSTANCE, &pool_9_fmt_layer, AI_STATIC,
  .tensors = &conv2d_8_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_integer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_9_fmt_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_8_output),
  AI_TENSOR_LIST_ENTRY(&pool_9_fmt_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_9_fmt_layer, 9,
  NL_TYPE,
  nl, node_convert,
  &AI_NET_OBJ_INSTANCE, &eltwise_10_layer, AI_STATIC,
  .tensors = &pool_9_fmt_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&pool_9_fmt_output, &input_10),
  AI_TENSOR_LIST_ENTRY(&eltwise_10_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_10_layer, 10,
  ELTWISE_TYPE,
  eltwise, forward_eltwise,
  &AI_NET_OBJ_INSTANCE, &eltwise_10_fmt_layer, AI_STATIC,
  .tensors = &eltwise_10_chain, 
  .operation = ai_mul, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_10_fmt_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&eltwise_10_output),
  AI_TENSOR_LIST_ENTRY(&eltwise_10_fmt_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_10_fmt_layer, 10,
  NL_TYPE,
  nl, node_convert,
  &AI_NET_OBJ_INSTANCE, &eltwise_11_layer, AI_STATIC,
  .tensors = &eltwise_10_fmt_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&eltwise_10_fmt_output, &input_11),
  AI_TENSOR_LIST_ENTRY(&eltwise_11_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_11_layer, 11,
  ELTWISE_TYPE,
  eltwise, forward_add_integer,
  &AI_NET_OBJ_INSTANCE, &dense_12_layer, AI_STATIC,
  .tensors = &eltwise_11_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&eltwise_11_output),
  AI_TENSOR_LIST_ENTRY(&dense_12_output),
  AI_TENSOR_LIST_ENTRY(&dense_12_weights, &dense_12_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_12_layer, 12,
  DENSE_TYPE,
  dense, forward_dense_integer,
  &AI_NET_OBJ_INSTANCE, &dense_13_layer, AI_STATIC,
  .tensors = &dense_12_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&dense_12_output),
  AI_TENSOR_LIST_ENTRY(&dense_13_output),
  AI_TENSOR_LIST_ENTRY(&dense_13_weights, &dense_13_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_13_layer, 13,
  DENSE_TYPE,
  dense, forward_dense_integer,
  &AI_NET_OBJ_INSTANCE, &dense_13_fmt_layer, AI_STATIC,
  .tensors = &dense_13_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_13_fmt_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&dense_13_output),
  AI_TENSOR_LIST_ENTRY(&dense_13_fmt_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_13_fmt_layer, 13,
  NL_TYPE,
  nl, node_convert,
  &AI_NET_OBJ_INSTANCE, &nl_14_layer, AI_STATIC,
  .tensors = &dense_13_fmt_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&dense_13_fmt_output),
  AI_TENSOR_LIST_ENTRY(&nl_14_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_14_layer, 14,
  NL_TYPE,
  nl, forward_sm,
  &AI_NET_OBJ_INSTANCE, &nl_14_fmt_layer, AI_STATIC,
  .tensors = &nl_14_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_14_fmt_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&nl_14_output),
  AI_TENSOR_LIST_ENTRY(&nl_14_fmt_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_14_fmt_layer, 14,
  NL_TYPE,
  nl, node_convert,
  &AI_NET_OBJ_INSTANCE, &nl_14_fmt_layer, AI_STATIC,
  .tensors = &nl_14_fmt_chain, 
)


AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 104808, 1,
                     NULL),
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 13520, 1,
                     NULL),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_TFLITE_IN_NUM, &input_0_output),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_TFLITE_OUT_NUM, &nl_14_fmt_output),
  &conv2d_0_layer, 0, NULL)



AI_DECLARE_STATIC
ai_bool tflite_configure_activations(
  ai_network* net_ctx, const ai_buffer* activation_buffer)
{
  AI_ASSERT(net_ctx &&  activation_buffer && activation_buffer->data)

  ai_ptr activations = AI_PTR(AI_PTR_ALIGN(activation_buffer->data, 4));
  AI_ASSERT(activations)
  AI_UNUSED(net_ctx)

  {
    /* Updating activations (byte) offsets */
    conv2d_8_scratch1_array.data = AI_PTR(activations + 5852);
    conv2d_8_scratch1_array.data_start = AI_PTR(activations + 5852);
    conv2d_8_scratch0_array.data = AI_PTR(activations + 1252);
    conv2d_8_scratch0_array.data_start = AI_PTR(activations + 1252);
    conv2d_4_scratch1_array.data = AI_PTR(activations + 3032);
    conv2d_4_scratch1_array.data_start = AI_PTR(activations + 3032);
    conv2d_4_scratch0_array.data = AI_PTR(activations + 1692);
    conv2d_4_scratch0_array.data_start = AI_PTR(activations + 1692);
    conv2d_0_scratch1_array.data = AI_PTR(activations + 336);
    conv2d_0_scratch1_array.data_start = AI_PTR(activations + 336);
    conv2d_0_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_0_scratch0_array.data_start = AI_PTR(activations + 0);
    input_0_output_array.data = AI_PTR(NULL);
    input_0_output_array.data_start = AI_PTR(NULL);
    conv2d_0_output_array.data = AI_PTR(activations + 6760);
    conv2d_0_output_array.data_start = AI_PTR(activations + 6760);
    pool_1_fmt_output_array.data = AI_PTR(activations + 0);
    pool_1_fmt_output_array.data_start = AI_PTR(activations + 0);
    eltwise_2_output_array.data = AI_PTR(activations + 6760);
    eltwise_2_output_array.data_start = AI_PTR(activations + 6760);
    eltwise_2_fmt_output_array.data = AI_PTR(activations + 6760);
    eltwise_2_fmt_output_array.data_start = AI_PTR(activations + 6760);
    eltwise_3_output_array.data = AI_PTR(activations + 0);
    eltwise_3_output_array.data_start = AI_PTR(activations + 0);
    conv2d_4_output_array.data = AI_PTR(activations + 5000);
    conv2d_4_output_array.data_start = AI_PTR(activations + 5000);
    pool_5_fmt_output_array.data = AI_PTR(activations + 0);
    pool_5_fmt_output_array.data_start = AI_PTR(activations + 0);
    eltwise_6_output_array.data = AI_PTR(activations + 5000);
    eltwise_6_output_array.data_start = AI_PTR(activations + 5000);
    eltwise_6_fmt_output_array.data = AI_PTR(activations + 5000);
    eltwise_6_fmt_output_array.data_start = AI_PTR(activations + 5000);
    eltwise_7_output_array.data = AI_PTR(activations + 0);
    eltwise_7_output_array.data_start = AI_PTR(activations + 0);
    conv2d_8_output_array.data = AI_PTR(activations + 6252);
    conv2d_8_output_array.data_start = AI_PTR(activations + 6252);
    pool_9_fmt_output_array.data = AI_PTR(activations + 0);
    pool_9_fmt_output_array.data_start = AI_PTR(activations + 0);
    eltwise_10_output_array.data = AI_PTR(activations + 400);
    eltwise_10_output_array.data_start = AI_PTR(activations + 400);
    eltwise_10_fmt_output_array.data = AI_PTR(activations + 400);
    eltwise_10_fmt_output_array.data_start = AI_PTR(activations + 400);
    eltwise_11_output_array.data = AI_PTR(activations + 0);
    eltwise_11_output_array.data_start = AI_PTR(activations + 0);
    dense_12_output_array.data = AI_PTR(activations + 100);
    dense_12_output_array.data_start = AI_PTR(activations + 100);
    dense_13_output_array.data = AI_PTR(activations + 0);
    dense_13_output_array.data_start = AI_PTR(activations + 0);
    dense_13_fmt_output_array.data = AI_PTR(activations + 44);
    dense_13_fmt_output_array.data_start = AI_PTR(activations + 44);
    nl_14_output_array.data = AI_PTR(activations + 44);
    nl_14_output_array.data_start = AI_PTR(activations + 44);
    nl_14_fmt_output_array.data = AI_PTR(NULL);
    nl_14_fmt_output_array.data_start = AI_PTR(NULL);
    
  }
  return true;
}



AI_DECLARE_STATIC
ai_bool tflite_configure_weights(
  ai_network* net_ctx, const ai_buffer* weights_buffer)
{
  AI_ASSERT(net_ctx &&  weights_buffer && weights_buffer->data)

  ai_ptr weights = AI_PTR(weights_buffer->data);
  AI_ASSERT(weights)
  AI_UNUSED(net_ctx)

  {
    /* Updating weights (byte) offsets */
    
    dense_13_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_13_bias_array.data = AI_PTR(weights + 104636);
    dense_13_bias_array.data_start = AI_PTR(weights + 104636);
    dense_13_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_13_weights_array.data = AI_PTR(weights + 100336);
    dense_13_weights_array.data_start = AI_PTR(weights + 100336);
    dense_12_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_12_bias_array.data = AI_PTR(weights + 99936);
    dense_12_bias_array.data_start = AI_PTR(weights + 99936);
    dense_12_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_12_weights_array.data = AI_PTR(weights + 89936);
    dense_12_weights_array.data_start = AI_PTR(weights + 89936);
    input_11_array.format |= AI_FMT_FLAG_CONST;
    input_11_array.data = AI_PTR(weights + 89836);
    input_11_array.data_start = AI_PTR(weights + 89836);
    input_10_array.format |= AI_FMT_FLAG_CONST;
    input_10_array.data = AI_PTR(weights + 89436);
    input_10_array.data_start = AI_PTR(weights + 89436);
    conv2d_8_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_bias_array.data = AI_PTR(weights + 89036);
    conv2d_8_bias_array.data_start = AI_PTR(weights + 89036);
    conv2d_8_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_weights_array.data = AI_PTR(weights + 9036);
    conv2d_8_weights_array.data_start = AI_PTR(weights + 9036);
    input_7_array.format |= AI_FMT_FLAG_CONST;
    input_7_array.data = AI_PTR(weights + 8984);
    input_7_array.data_start = AI_PTR(weights + 8984);
    input_6_array.format |= AI_FMT_FLAG_CONST;
    input_6_array.data = AI_PTR(weights + 8784);
    input_6_array.data_start = AI_PTR(weights + 8784);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(weights + 8584);
    conv2d_4_bias_array.data_start = AI_PTR(weights + 8584);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(weights + 584);
    conv2d_4_weights_array.data_start = AI_PTR(weights + 584);
    input_3_array.format |= AI_FMT_FLAG_CONST;
    input_3_array.data = AI_PTR(weights + 572);
    input_3_array.data_start = AI_PTR(weights + 572);
    input_2_array.format |= AI_FMT_FLAG_CONST;
    input_2_array.data = AI_PTR(weights + 532);
    input_2_array.data_start = AI_PTR(weights + 532);
    conv2d_0_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_bias_array.data = AI_PTR(weights + 492);
    conv2d_0_bias_array.data_start = AI_PTR(weights + 492);
    conv2d_0_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_weights_array.data = AI_PTR(weights + 0);
    conv2d_0_weights_array.data_start = AI_PTR(weights + 0);
  }

  return true;
}


/**  PUBLIC APIs SECTION  *****************************************************/

AI_API_ENTRY
ai_bool ai_tflite_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if ( report && net_ctx )
  {
    ai_network_report r = {
      .model_name        = AI_TFLITE_MODEL_NAME,
      .model_signature   = AI_TFLITE_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = {AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR,
                            AI_TOOLS_API_VERSION_MICRO, 0x0},

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 1496917,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .activations       = AI_STRUCT_INIT,
      .params            = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if ( !ai_platform_api_get_network_report(network, &r) ) return false;

    *report = r;
    return true;
  }

  return false;
}

AI_API_ENTRY
ai_error ai_tflite_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_tflite_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_handle ai_tflite_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_tflite_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if ( !net_ctx ) return false;

  ai_bool ok = true;
  ok &= tflite_configure_weights(net_ctx, &params->params);
  ok &= tflite_configure_activations(net_ctx, &params->activations);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_tflite_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_tflite_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}

#undef AI_TFLITE_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

