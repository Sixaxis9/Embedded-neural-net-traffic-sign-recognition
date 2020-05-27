################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include X-CUBE-AI/App/subdir.mk
-include Drivers/STM32L4xx_HAL_Driver/Src/subdir.mk
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
Keras_Template.elf \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
Keras_Template.list \

OBJCOPY_BIN += \
Keras_Template.bin \


# All Target
all: Keras_Template.elf secondary-outputs

# Tool invocations
Keras_Template.elf: $(OBJS) $(USER_OBJS) C:\Users\Marco\Documents\GitHub\Embedded-neural-net-traffic-sign-recognition\Firmware\Keras_Template\STM32L475VGTX_FLASH.ld
	arm-none-eabi-gcc -o "Keras_Template.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m4 -T"C:\Users\Marco\Documents\GitHub\Embedded-neural-net-traffic-sign-recognition\Firmware\Keras_Template\STM32L475VGTX_FLASH.ld" --specs=nosys.specs -Wl,-Map="Keras_Template.map" -Wl,--gc-sections -static -L../Middlewares/ST/AI/Lib --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -Wl,--start-group -lc -lm -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

Keras_Template.list: $(EXECUTABLES)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "Keras_Template.list"
	@echo 'Finished building: $@'
	@echo ' '

Keras_Template.bin: $(EXECUTABLES)
	arm-none-eabi-objcopy  -O binary $(EXECUTABLES) "Keras_Template.bin"
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
