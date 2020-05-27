################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../X-CUBE-AI/App/app_x-cube-ai.c \
../X-CUBE-AI/App/tflite.c \
../X-CUBE-AI/App/tflite_data.c 

OBJS += \
./X-CUBE-AI/App/app_x-cube-ai.o \
./X-CUBE-AI/App/tflite.o \
./X-CUBE-AI/App/tflite_data.o 

C_DEPS += \
./X-CUBE-AI/App/app_x-cube-ai.d \
./X-CUBE-AI/App/tflite.d \
./X-CUBE-AI/App/tflite_data.d 


# Each subdirectory must supply rules for building sources it contributes
X-CUBE-AI/App/app_x-cube-ai.o: ../X-CUBE-AI/App/app_x-cube-ai.c
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DUSE_HAL_DRIVER -DDEBUG -DSTM32L475xx -c -I../Drivers/CMSIS/Include -I../Core/Inc -I../Middlewares/ST/AI/Inc -I../X-CUBE-AI -I../Drivers/CMSIS/Device/ST/STM32L4xx/Include -I../Drivers/STM32L4xx_HAL_Driver/Inc -I../X-CUBE-AI/App -I../Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"X-CUBE-AI/App/app_x-cube-ai.d" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
X-CUBE-AI/App/tflite.o: ../X-CUBE-AI/App/tflite.c
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DUSE_HAL_DRIVER -DDEBUG -DSTM32L475xx -c -I../Drivers/CMSIS/Include -I../Core/Inc -I../Middlewares/ST/AI/Inc -I../X-CUBE-AI -I../Drivers/CMSIS/Device/ST/STM32L4xx/Include -I../Drivers/STM32L4xx_HAL_Driver/Inc -I../X-CUBE-AI/App -I../Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"X-CUBE-AI/App/tflite.d" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
X-CUBE-AI/App/tflite_data.o: ../X-CUBE-AI/App/tflite_data.c
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DUSE_HAL_DRIVER -DDEBUG -DSTM32L475xx -c -I../Drivers/CMSIS/Include -I../Core/Inc -I../Middlewares/ST/AI/Inc -I../X-CUBE-AI -I../Drivers/CMSIS/Device/ST/STM32L4xx/Include -I../Drivers/STM32L4xx_HAL_Driver/Inc -I../X-CUBE-AI/App -I../Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"X-CUBE-AI/App/tflite_data.d" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

