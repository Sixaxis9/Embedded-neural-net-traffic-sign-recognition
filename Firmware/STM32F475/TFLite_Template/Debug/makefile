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
TFLite_Template.elf \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
TFLite_Template.list \

OBJCOPY_BIN += \
TFLite_Template.bin \


# All Target
all: TFLite_Template.elf secondary-outputs

# Tool invocations
TFLite_Template.elf: $(OBJS) $(USER_OBJS) C:\Users\Marco\Documents\GitHub\Embedded-neural-net-traffic-sign-recognition\Firmware\TFLite_Template\STM32L475VGTX_FLASH.ld
	arm-none-eabi-gcc -o "TFLite_Template.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m4 -T"C:\Users\Marco\Documents\GitHub\Embedded-neural-net-traffic-sign-recognition\Firmware\TFLite_Template\STM32L475VGTX_FLASH.ld" --specs=nosys.specs -Wl,-Map="TFLite_Template.map" -Wl,--gc-sections -static -L../Middlewares/ST/AI/Lib --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -Wl,--start-group -lc -lm -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

TFLite_Template.list: $(EXECUTABLES)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "TFLite_Template.list"
	@echo 'Finished building: $@'
	@echo ' '

TFLite_Template.bin: $(EXECUTABLES)
	arm-none-eabi-objcopy  -O binary $(EXECUTABLES) "TFLite_Template.bin"
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
