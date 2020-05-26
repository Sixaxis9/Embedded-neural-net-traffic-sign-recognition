/**
  ******************************************************************************
  * @file    keras_n_bn.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Tue May 26 20:29:03 2020
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


#include "keras_n_bn.h"

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
#define AI_NET_OBJ_INSTANCE g_keras_n_bn
 
#undef AI_KERAS_N_BN_MODEL_SIGNATURE
#define AI_KERAS_N_BN_MODEL_SIGNATURE     "4345c1e519e258e4030c4ca723664bc0"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     "(rev-5.0.0)"
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Tue May 26 20:29:03 2020"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_KERAS_N_BN_N_BATCHES
#define AI_KERAS_N_BN_N_BATCHES         (1)

/**  Forward network declaration section  *************************************/
AI_STATIC ai_network AI_NET_OBJ_INSTANCE;


/**  Forward network array declarations  **************************************/
AI_STATIC ai_array conv2d_6_scratch0_array;   /* Array #0 */
AI_STATIC ai_array conv2d_5_scratch0_array;   /* Array #1 */
AI_STATIC ai_array conv2d_4_scratch0_array;   /* Array #2 */
AI_STATIC ai_array dense_4_bias_array;   /* Array #3 */
AI_STATIC ai_array dense_4_weights_array;   /* Array #4 */
AI_STATIC ai_array dense_3_bias_array;   /* Array #5 */
AI_STATIC ai_array dense_3_weights_array;   /* Array #6 */
AI_STATIC ai_array batch_normalization_3_bias_array;   /* Array #7 */
AI_STATIC ai_array batch_normalization_3_scale_array;   /* Array #8 */
AI_STATIC ai_array conv2d_6_bias_array;   /* Array #9 */
AI_STATIC ai_array conv2d_6_weights_array;   /* Array #10 */
AI_STATIC ai_array batch_normalization_2_bias_array;   /* Array #11 */
AI_STATIC ai_array batch_normalization_2_scale_array;   /* Array #12 */
AI_STATIC ai_array conv2d_5_bias_array;   /* Array #13 */
AI_STATIC ai_array conv2d_5_weights_array;   /* Array #14 */
AI_STATIC ai_array batch_normalization_1_bias_array;   /* Array #15 */
AI_STATIC ai_array batch_normalization_1_scale_array;   /* Array #16 */
AI_STATIC ai_array conv2d_4_bias_array;   /* Array #17 */
AI_STATIC ai_array conv2d_4_weights_array;   /* Array #18 */
AI_STATIC ai_array input_0_output_array;   /* Array #19 */
AI_STATIC ai_array conv2d_4_output_array;   /* Array #20 */
AI_STATIC ai_array batch_normalization_1_output_array;   /* Array #21 */
AI_STATIC ai_array conv2d_5_output_array;   /* Array #22 */
AI_STATIC ai_array batch_normalization_2_output_array;   /* Array #23 */
AI_STATIC ai_array conv2d_6_output_array;   /* Array #24 */
AI_STATIC ai_array batch_normalization_3_output_array;   /* Array #25 */
AI_STATIC ai_array dense_3_output_array;   /* Array #26 */
AI_STATIC ai_array activation_9_output_array;   /* Array #27 */
AI_STATIC ai_array dense_4_output_array;   /* Array #28 */
AI_STATIC ai_array activation_10_output_array;   /* Array #29 */


/**  Forward network tensor declarations  *************************************/
AI_STATIC ai_tensor conv2d_6_scratch0;   /* Tensor #0 */
AI_STATIC ai_tensor conv2d_5_scratch0;   /* Tensor #1 */
AI_STATIC ai_tensor conv2d_4_scratch0;   /* Tensor #2 */
AI_STATIC ai_tensor dense_4_bias;   /* Tensor #3 */
AI_STATIC ai_tensor dense_4_weights;   /* Tensor #4 */
AI_STATIC ai_tensor dense_3_bias;   /* Tensor #5 */
AI_STATIC ai_tensor dense_3_weights;   /* Tensor #6 */
AI_STATIC ai_tensor batch_normalization_3_bias;   /* Tensor #7 */
AI_STATIC ai_tensor batch_normalization_3_scale;   /* Tensor #8 */
AI_STATIC ai_tensor conv2d_6_bias;   /* Tensor #9 */
AI_STATIC ai_tensor conv2d_6_weights;   /* Tensor #10 */
AI_STATIC ai_tensor batch_normalization_2_bias;   /* Tensor #11 */
AI_STATIC ai_tensor batch_normalization_2_scale;   /* Tensor #12 */
AI_STATIC ai_tensor conv2d_5_bias;   /* Tensor #13 */
AI_STATIC ai_tensor conv2d_5_weights;   /* Tensor #14 */
AI_STATIC ai_tensor batch_normalization_1_bias;   /* Tensor #15 */
AI_STATIC ai_tensor batch_normalization_1_scale;   /* Tensor #16 */
AI_STATIC ai_tensor conv2d_4_bias;   /* Tensor #17 */
AI_STATIC ai_tensor conv2d_4_weights;   /* Tensor #18 */
AI_STATIC ai_tensor input_0_output;   /* Tensor #19 */
AI_STATIC ai_tensor conv2d_4_output;   /* Tensor #20 */
AI_STATIC ai_tensor batch_normalization_1_output;   /* Tensor #21 */
AI_STATIC ai_tensor conv2d_5_output;   /* Tensor #22 */
AI_STATIC ai_tensor batch_normalization_2_output;   /* Tensor #23 */
AI_STATIC ai_tensor conv2d_6_output;   /* Tensor #24 */
AI_STATIC ai_tensor batch_normalization_3_output;   /* Tensor #25 */
AI_STATIC ai_tensor dense_3_output;   /* Tensor #26 */
AI_STATIC ai_tensor activation_9_output;   /* Tensor #27 */
AI_STATIC ai_tensor dense_4_output;   /* Tensor #28 */
AI_STATIC ai_tensor activation_10_output;   /* Tensor #29 */


/**  Forward network tensor chain declarations  *******************************/
AI_STATIC_CONST ai_tensor_chain conv2d_4_chain;   /* Chain #0 */
AI_STATIC_CONST ai_tensor_chain batch_normalization_1_chain;   /* Chain #1 */
AI_STATIC_CONST ai_tensor_chain conv2d_5_chain;   /* Chain #2 */
AI_STATIC_CONST ai_tensor_chain batch_normalization_2_chain;   /* Chain #3 */
AI_STATIC_CONST ai_tensor_chain conv2d_6_chain;   /* Chain #4 */
AI_STATIC_CONST ai_tensor_chain batch_normalization_3_chain;   /* Chain #5 */
AI_STATIC_CONST ai_tensor_chain dense_3_chain;   /* Chain #6 */
AI_STATIC_CONST ai_tensor_chain activation_9_chain;   /* Chain #7 */
AI_STATIC_CONST ai_tensor_chain dense_4_chain;   /* Chain #8 */
AI_STATIC_CONST ai_tensor_chain activation_10_chain;   /* Chain #9 */


/**  Forward network layer declarations  **************************************/
AI_STATIC ai_layer_conv2d_nl_pool conv2d_4_layer; /* Layer #0 */
AI_STATIC ai_layer_bn batch_normalization_1_layer; /* Layer #1 */
AI_STATIC ai_layer_conv2d_nl_pool conv2d_5_layer; /* Layer #2 */
AI_STATIC ai_layer_bn batch_normalization_2_layer; /* Layer #3 */
AI_STATIC ai_layer_conv2d_nl_pool conv2d_6_layer; /* Layer #4 */
AI_STATIC ai_layer_bn batch_normalization_3_layer; /* Layer #5 */
AI_STATIC ai_layer_dense dense_3_layer; /* Layer #6 */
AI_STATIC ai_layer_nl activation_9_layer; /* Layer #7 */
AI_STATIC ai_layer_dense dense_4_layer; /* Layer #8 */
AI_STATIC ai_layer_nl activation_10_layer; /* Layer #9 */


/**  Array declarations section  **********************************************/
AI_ARRAY_OBJ_DECLARE(
    conv2d_6_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 400,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_5_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1000,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_4_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 520,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_4_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 43,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_4_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 4300,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_3_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_3_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 10000,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    batch_normalization_3_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    batch_normalization_3_scale_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_6_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_6_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 80000,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    batch_normalization_2_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 50,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    batch_normalization_2_scale_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 50,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_5_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 50,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_5_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 8000,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    batch_normalization_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 10,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    batch_normalization_1_scale_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 10,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_4_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 10,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_4_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 490,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    input_0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
    NULL, NULL, 1024,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_4_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1690,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    batch_normalization_1_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1690,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_5_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1250,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    batch_normalization_2_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1250,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_6_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    batch_normalization_3_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_3_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    activation_9_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 100,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_4_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 43,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    activation_10_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
    NULL, NULL, 43,
     AI_STATIC)




/**  Tensor declarations section  *********************************************/
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 2, 2), AI_STRIDE_INIT(4, 4, 4, 400, 800),
  1, &conv2d_6_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 10, 2), AI_STRIDE_INIT(4, 4, 4, 200, 2000),
  1, &conv2d_5_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 26, 2), AI_STRIDE_INIT(4, 4, 4, 40, 1040),
  1, &conv2d_4_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_4_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 43, 1, 1), AI_STRIDE_INIT(4, 4, 4, 172, 172),
  1, &dense_4_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_4_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 100, 43, 1, 1), AI_STRIDE_INIT(4, 4, 400, 17200, 17200),
  1, &dense_4_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_3_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &dense_3_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_3_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 100, 100, 1, 1), AI_STRIDE_INIT(4, 4, 400, 40000, 40000),
  1, &dense_3_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_3_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &batch_normalization_3_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_3_scale, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &batch_normalization_3_scale_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &conv2d_6_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 50, 4, 4, 100), AI_STRIDE_INIT(4, 4, 200, 800, 3200),
  1, &conv2d_6_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_2_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &batch_normalization_2_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_2_scale, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &batch_normalization_2_scale_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &conv2d_5_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 10, 4, 4, 50), AI_STRIDE_INIT(4, 4, 40, 160, 640),
  1, &conv2d_5_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_1_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &batch_normalization_1_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_1_scale, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &batch_normalization_1_scale_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv2d_4_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 7, 7, 10), AI_STRIDE_INIT(4, 4, 4, 28, 196),
  1, &conv2d_4_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  input_0_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1, 32, 32), AI_STRIDE_INIT(4, 4, 4, 4, 128),
  1, &input_0_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 13, 13), AI_STRIDE_INIT(4, 4, 4, 40, 520),
  1, &conv2d_4_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_1_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 13, 13), AI_STRIDE_INIT(4, 4, 4, 40, 520),
  1, &batch_normalization_1_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 5, 5), AI_STRIDE_INIT(4, 4, 4, 200, 1000),
  1, &conv2d_5_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_2_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 50, 5, 5), AI_STRIDE_INIT(4, 4, 4, 200, 1000),
  1, &batch_normalization_2_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &conv2d_6_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  batch_normalization_3_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &batch_normalization_3_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_3_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &dense_3_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  activation_9_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &activation_9_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_4_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 43, 1, 1), AI_STRIDE_INIT(4, 4, 4, 172, 172),
  1, &dense_4_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  activation_10_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 43, 1, 1), AI_STRIDE_INIT(4, 4, 4, 172, 172),
  1, &activation_10_output_array, NULL)


/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&input_0_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_4_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_4_weights, &conv2d_4_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_4_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_layer, 0,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &AI_NET_OBJ_INSTANCE, &batch_normalization_1_layer, AI_STATIC,
  .tensors = &conv2d_4_chain, 
  .groups = 1, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  batch_normalization_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_4_output),
  AI_TENSOR_LIST_ENTRY(&batch_normalization_1_output),
  AI_TENSOR_LIST_ENTRY(&batch_normalization_1_scale, &batch_normalization_1_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  batch_normalization_1_layer, 3,
  BN_TYPE,
  bn, forward_bn,
  &AI_NET_OBJ_INSTANCE, &conv2d_5_layer, AI_STATIC,
  .tensors = &batch_normalization_1_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&batch_normalization_1_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_5_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_5_weights, &conv2d_5_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_5_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_5_layer, 4,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &AI_NET_OBJ_INSTANCE, &batch_normalization_2_layer, AI_STATIC,
  .tensors = &conv2d_5_chain, 
  .groups = 1, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  batch_normalization_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_5_output),
  AI_TENSOR_LIST_ENTRY(&batch_normalization_2_output),
  AI_TENSOR_LIST_ENTRY(&batch_normalization_2_scale, &batch_normalization_2_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  batch_normalization_2_layer, 7,
  BN_TYPE,
  bn, forward_bn,
  &AI_NET_OBJ_INSTANCE, &conv2d_6_layer, AI_STATIC,
  .tensors = &batch_normalization_2_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&batch_normalization_2_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_6_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_6_weights, &conv2d_6_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_6_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_6_layer, 8,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &AI_NET_OBJ_INSTANCE, &batch_normalization_3_layer, AI_STATIC,
  .tensors = &conv2d_6_chain, 
  .groups = 1, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  batch_normalization_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_6_output),
  AI_TENSOR_LIST_ENTRY(&batch_normalization_3_output),
  AI_TENSOR_LIST_ENTRY(&batch_normalization_3_scale, &batch_normalization_3_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  batch_normalization_3_layer, 11,
  BN_TYPE,
  bn, forward_bn,
  &AI_NET_OBJ_INSTANCE, &dense_3_layer, AI_STATIC,
  .tensors = &batch_normalization_3_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&batch_normalization_3_output),
  AI_TENSOR_LIST_ENTRY(&dense_3_output),
  AI_TENSOR_LIST_ENTRY(&dense_3_weights, &dense_3_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_3_layer, 13,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &activation_9_layer, AI_STATIC,
  .tensors = &dense_3_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&dense_3_output),
  AI_TENSOR_LIST_ENTRY(&activation_9_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_9_layer, 14,
  NL_TYPE,
  nl, forward_relu,
  &AI_NET_OBJ_INSTANCE, &dense_4_layer, AI_STATIC,
  .tensors = &activation_9_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&activation_9_output),
  AI_TENSOR_LIST_ENTRY(&dense_4_output),
  AI_TENSOR_LIST_ENTRY(&dense_4_weights, &dense_4_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_4_layer, 15,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &activation_10_layer, AI_STATIC,
  .tensors = &dense_4_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&dense_4_output),
  AI_TENSOR_LIST_ENTRY(&activation_10_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_10_layer, 16,
  NL_TYPE,
  nl, forward_sm,
  &AI_NET_OBJ_INSTANCE, &activation_10_layer, AI_STATIC,
  .tensors = &activation_10_chain, 
)


AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 413652, 1,
                     NULL),
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 11960, 1,
                     NULL),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_KERAS_N_BN_IN_NUM, &input_0_output),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_KERAS_N_BN_OUT_NUM, &activation_10_output),
  &conv2d_4_layer, 0, NULL)



AI_DECLARE_STATIC
ai_bool keras_n_bn_configure_activations(
  ai_network* net_ctx, const ai_buffer* activation_buffer)
{
  AI_ASSERT(net_ctx &&  activation_buffer && activation_buffer->data)

  ai_ptr activations = AI_PTR(AI_PTR_ALIGN(activation_buffer->data, 4));
  AI_ASSERT(activations)
  AI_UNUSED(net_ctx)

  {
    /* Updating activations (byte) offsets */
    conv2d_6_scratch0_array.data = AI_PTR(activations + 7960);
    conv2d_6_scratch0_array.data_start = AI_PTR(activations + 7960);
    conv2d_5_scratch0_array.data = AI_PTR(activations + 7960);
    conv2d_5_scratch0_array.data_start = AI_PTR(activations + 7960);
    conv2d_4_scratch0_array.data = AI_PTR(activations + 7960);
    conv2d_4_scratch0_array.data_start = AI_PTR(activations + 7960);
    input_0_output_array.data = AI_PTR(NULL);
    input_0_output_array.data_start = AI_PTR(NULL);
    conv2d_4_output_array.data = AI_PTR(activations + 1200);
    conv2d_4_output_array.data_start = AI_PTR(activations + 1200);
    batch_normalization_1_output_array.data = AI_PTR(activations + 1200);
    batch_normalization_1_output_array.data_start = AI_PTR(activations + 1200);
    conv2d_5_output_array.data = AI_PTR(activations + 0);
    conv2d_5_output_array.data_start = AI_PTR(activations + 0);
    batch_normalization_2_output_array.data = AI_PTR(activations + 0);
    batch_normalization_2_output_array.data_start = AI_PTR(activations + 0);
    conv2d_6_output_array.data = AI_PTR(activations + 7560);
    conv2d_6_output_array.data_start = AI_PTR(activations + 7560);
    batch_normalization_3_output_array.data = AI_PTR(activations + 7560);
    batch_normalization_3_output_array.data_start = AI_PTR(activations + 7560);
    dense_3_output_array.data = AI_PTR(activations + 7160);
    dense_3_output_array.data_start = AI_PTR(activations + 7160);
    activation_9_output_array.data = AI_PTR(activations + 7160);
    activation_9_output_array.data_start = AI_PTR(activations + 7160);
    dense_4_output_array.data = AI_PTR(activations + 6988);
    dense_4_output_array.data_start = AI_PTR(activations + 6988);
    activation_10_output_array.data = AI_PTR(NULL);
    activation_10_output_array.data_start = AI_PTR(NULL);
    
  }
  return true;
}



AI_DECLARE_STATIC
ai_bool keras_n_bn_configure_weights(
  ai_network* net_ctx, const ai_buffer* weights_buffer)
{
  AI_ASSERT(net_ctx &&  weights_buffer && weights_buffer->data)

  ai_ptr weights = AI_PTR(weights_buffer->data);
  AI_ASSERT(weights)
  AI_UNUSED(net_ctx)

  {
    /* Updating weights (byte) offsets */
    
    dense_4_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_4_bias_array.data = AI_PTR(weights + 413480);
    dense_4_bias_array.data_start = AI_PTR(weights + 413480);
    dense_4_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_4_weights_array.data = AI_PTR(weights + 396280);
    dense_4_weights_array.data_start = AI_PTR(weights + 396280);
    dense_3_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_3_bias_array.data = AI_PTR(weights + 395880);
    dense_3_bias_array.data_start = AI_PTR(weights + 395880);
    dense_3_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_3_weights_array.data = AI_PTR(weights + 355880);
    dense_3_weights_array.data_start = AI_PTR(weights + 355880);
    batch_normalization_3_bias_array.format |= AI_FMT_FLAG_CONST;
    batch_normalization_3_bias_array.data = AI_PTR(weights + 355480);
    batch_normalization_3_bias_array.data_start = AI_PTR(weights + 355480);
    batch_normalization_3_scale_array.format |= AI_FMT_FLAG_CONST;
    batch_normalization_3_scale_array.data = AI_PTR(weights + 355080);
    batch_normalization_3_scale_array.data_start = AI_PTR(weights + 355080);
    conv2d_6_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_6_bias_array.data = AI_PTR(weights + 354680);
    conv2d_6_bias_array.data_start = AI_PTR(weights + 354680);
    conv2d_6_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_6_weights_array.data = AI_PTR(weights + 34680);
    conv2d_6_weights_array.data_start = AI_PTR(weights + 34680);
    batch_normalization_2_bias_array.format |= AI_FMT_FLAG_CONST;
    batch_normalization_2_bias_array.data = AI_PTR(weights + 34480);
    batch_normalization_2_bias_array.data_start = AI_PTR(weights + 34480);
    batch_normalization_2_scale_array.format |= AI_FMT_FLAG_CONST;
    batch_normalization_2_scale_array.data = AI_PTR(weights + 34280);
    batch_normalization_2_scale_array.data_start = AI_PTR(weights + 34280);
    conv2d_5_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_bias_array.data = AI_PTR(weights + 34080);
    conv2d_5_bias_array.data_start = AI_PTR(weights + 34080);
    conv2d_5_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_weights_array.data = AI_PTR(weights + 2080);
    conv2d_5_weights_array.data_start = AI_PTR(weights + 2080);
    batch_normalization_1_bias_array.format |= AI_FMT_FLAG_CONST;
    batch_normalization_1_bias_array.data = AI_PTR(weights + 2040);
    batch_normalization_1_bias_array.data_start = AI_PTR(weights + 2040);
    batch_normalization_1_scale_array.format |= AI_FMT_FLAG_CONST;
    batch_normalization_1_scale_array.data = AI_PTR(weights + 2000);
    batch_normalization_1_scale_array.data_start = AI_PTR(weights + 2000);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(weights + 1960);
    conv2d_4_bias_array.data_start = AI_PTR(weights + 1960);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(weights + 0);
    conv2d_4_weights_array.data_start = AI_PTR(weights + 0);
  }

  return true;
}


/**  PUBLIC APIs SECTION  *****************************************************/

AI_API_ENTRY
ai_bool ai_keras_n_bn_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if ( report && net_ctx )
  {
    ai_network_report r = {
      .model_name        = AI_KERAS_N_BN_MODEL_NAME,
      .model_signature   = AI_KERAS_N_BN_MODEL_SIGNATURE,
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
      
      .n_macc            = 1496845,
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
ai_error ai_keras_n_bn_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_keras_n_bn_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_handle ai_keras_n_bn_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_keras_n_bn_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if ( !net_ctx ) return false;

  ai_bool ok = true;
  ok &= keras_n_bn_configure_weights(net_ctx, &params->params);
  ok &= keras_n_bn_configure_activations(net_ctx, &params->activations);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_keras_n_bn_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_keras_n_bn_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}

#undef AI_KERAS_N_BN_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

