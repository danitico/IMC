################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../imc/PerceptronMulticapa.cpp 

OBJS += \
./imc/PerceptronMulticapa.o 

CPP_DEPS += \
./imc/PerceptronMulticapa.d 


# Each subdirectory must supply rules for building sources it contributes
imc/%.o: ../imc/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


