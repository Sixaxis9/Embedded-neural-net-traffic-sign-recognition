/*
 * inference_w.c
 *
 *  Created on: May 26, 2020
 *      Author: Marco
 */


#include "inference_w.h"
#include "app_x-cube-ai.h"

  uint8_t x_data[BUFFER_SIZE] = {};
  float output_arr[43] = {};
  int8_t output_arr_int[43] = {};
  int array_pos = 0;
  uint8_t byte_received = 0;


void inference_w(UART_HandleTypeDef huart1){
	  int uart_status = HAL_UART_Receive(&huart1, &byte_received, sizeof(uint8_t), 100);
	  if (uart_status == HAL_OK)
	  {
	  	x_data[array_pos] = byte_received;
	  	array_pos++;
	  }
	  if (array_pos == BUFFER_SIZE)
	  {
	  	//HAL_Delay(500);
	  	//HAL_UART_Transmit(&huart1, x_data, BUFFER_SIZE*sizeof(uint8_t), 2000);
	  	array_pos = 0;

	  	// instantiate float array
	  	float fArray[BUFFER_SIZE];

	  	// step through each element of integer array, and copy into float array as float
	  	for (int i = 0; i < BUFFER_SIZE; i++)
	  	{
	  		fArray[i] = ((float)x_data[i]) / 255;
	  	}

	  	ai_i8 x_data_ai[4 * BUFFER_SIZE] = {};
	  	ai_i8 y_data_ai[4 * 43] = {};
	  	uint8_t best_output = 0;
	  	float accuracy = 0;
	  	memcpy(x_data_ai, fArray, 4 * BUFFER_SIZE);

	  	ResetTimer();
	  	StartTimer();
	  	aiRun(x_data_ai, y_data_ai);
	  	StopTimer();
	  	uint32_t cycles_count = getCycles();
	  	memcpy(output_arr, y_data_ai, sizeof(float) * 43);
	  	for (int i = 0; i < 43; i++)
	  	{
	  		if (accuracy < output_arr[i])
	  		{
	  			accuracy = output_arr[i];
	  			best_output = i;
	  		}
	  	}
	  	HAL_UART_Transmit(&huart1, &best_output, 1, 200);
	  	HAL_UART_Transmit(&huart1, &accuracy, sizeof(float), 200);
	  	HAL_UART_Transmit(&huart1, &cycles_count, sizeof(uint32_t), 200);
	  }
	  }


void inference_w_a(UART_HandleTypeDef huart1){
		  int uart_status = HAL_UART_Receive(&huart1, &byte_received, sizeof(uint8_t), 100);
		  	  if (uart_status == HAL_OK)
		  	  {
		  	  	x_data[array_pos] = byte_received;
		  	  	array_pos++;
		  	  }
		  	  if (array_pos == BUFFER_SIZE)
		  	  {
		  	  	//HAL_Delay(500);
		  	  	//HAL_UART_Transmit(&huart1, x_data, BUFFER_SIZE*sizeof(uint8_t), 2000);
		  	  	array_pos = 0;

		  	  int8_t fArray[BUFFER_SIZE];

		  	  		  	// step through each element of integer array, and copy into float array as float
		  	  		  	for (int i = 0; i < BUFFER_SIZE; i++)
		  	  		  	{
		  	  		  		fArray[i] = (int8_t) x_data[i] >> 1 & 0b01111111;
		  	  		  	}

		  	  	ai_i8 x_data_ai[BUFFER_SIZE] = {};
		  	  	ai_i8 y_data_ai[43] = {};
		  	  	uint8_t best_output = 0;
		  	  	float accuracy = 0;
		  	  	memcpy(x_data_ai, fArray, BUFFER_SIZE);

		  	  	ResetTimer();
		  	  	StartTimer();
		  	  	aiRun(x_data_ai, y_data_ai);
		  	  	StopTimer();
		  	  	uint32_t cycles_count = getCycles();
		  	  	memcpy(output_arr_int, y_data_ai, 43);
		  	  	for (int i = 0; i < 43; i++)
		  	  	{
		  	  		if (accuracy < output_arr_int[i])
		  	  		{
		  	  			accuracy = output_arr_int[i];
		  	  			best_output = i;
		  	  		}
		  	  	}
		  	  	HAL_UART_Transmit(&huart1, &best_output, 1, 200);
		  	  	HAL_UART_Transmit(&huart1, &accuracy, sizeof(float), 200);
		  	  	HAL_UART_Transmit(&huart1, &cycles_count, sizeof(uint32_t), 200);
		  	  }
	  }
