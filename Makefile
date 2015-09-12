RM := rm -rf

CC=g++
NVCC=nvcc
INCL_DIR=./include
CFLAGS=-c -O2 -I $(INCL_DIR) -w
COMPUTE=$(if $(compute),$(compute),20)
SM=$(if $(sm),$(sm),20)
GENCODE=-gencode arch=compute_$(COMPUTE),code=compute_$(COMPUTE) -gencode arch=compute_$(COMPUTE),code=sm_$(SM)
LDFLAGS=-I $(INCL_DIR)

CU_SRCS += \
src/kernel.cu \
src/unit_1.cu

SRCS += \
src/main.cpp

OBJS=$(SRCS:.cpp=.o)
#OBJS+= $(SRCS:.c=.o)
CU_OBJS=$(CU_SRCS:.cu=.o)

EXECUTABLE=vortex

# All Target
all: $(SRCS) $(CU_SRCS) $(EXECUTABLE)

%.o : %.cu
	@echo 'Building file: $<'
	@echo 'Invoking: $(NVCC) Compiler'
	$(NVCC) $(CFLAGS) $(GENCODE) -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o : %.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: $(CC) Compiler'
	$(CC) $(CFLAGS) -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

$(EXECUTABLE): $(OBJS) $(CU_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC Linker'
	$(NVCC) $(LDFLAGS) $(GENCODE) -o $(EXECUTABLE) $(OBJS) $(CU_OBJS)
	@echo 'Finished building target: $@'
	@echo ' '

clean:
	$(RM) $(EXECUTABLE) $(OBJS) $(CU_OBJS)
