################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include X-CUBE-AI/App/subdir.mk
-include Drivers/STM32F7xx_HAL_Driver/Src/subdir.mk
-include Core/Startup/subdir.mk
-include Core/Src/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 
EXECUTABLES += \
TFLite_Template_F7.elf \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
TFLite_Template_F7.list \

OBJCOPY_BIN += \
TFLite_Template_F7.bin \


# All Target
all: TFLite_Template_F7.elf secondary-outputs

# Tool invocations
TFLite_Template_F7.elf: $(OBJS) $(USER_OBJS) C:\Users\Marco\Documents\GitHub\Embedded-neural-net-traffic-sign-recognition\Firmware\STM32F723\TFLite_Template_F7\STM32F723IEKX_FLASH.ld
	arm-none-eabi-gcc -o "TFLite_Template_F7.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m7 -T"C:\Users\Marco\Documents\GitHub\Embedded-neural-net-traffic-sign-recognition\Firmware\STM32F723\TFLite_Template_F7\STM32F723IEKX_FLASH.ld" --specs=nosys.specs -Wl,-Map="TFLite_Template_F7.map" -Wl,--gc-sections -static -L../Middlewares/ST/AI/Lib --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -Wl,--start-group -lc -lm -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

TFLite_Template_F7.list: $(EXECUTABLES)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "TFLite_Template_F7.list"
	@echo 'Finished building: $@'
	@echo ' '

TFLite_Template_F7.bin: $(EXECUTABLES)
	arm-none-eabi-objcopy  -O binary $(EXECUTABLES) "TFLite_Template_F7.bin"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) *
	-@echo ' '

secondary-outputs: $(SIZE_OUTPUT) $(OBJDUMP_LIST) $(OBJCOPY_BIN)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
