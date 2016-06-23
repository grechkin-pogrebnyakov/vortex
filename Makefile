RM := rm -rf

CC=gcc
NVCC=/usr/local/cuda/bin/nvcc
LD=gcc
BASE_DIR=.
INCL_DIR=$(BASE_DIR)/include
SRC_DIR=$(BASE_DIR)/src
BUILD_DIR=$(BASE_DIR)/build
CUDA_INCL_DIR=/usr/local/cuda/include
CUDA_LIB_DIR=/usr/local/cuda/lib64
OPTIMIZATION=$(if $(DEBUG),-O0 -g,-O2)
TREE=$(if $(NO_TREE), -DNO_TREE, )
CFLAGS=-c $(OPTIMIZATION) -I$(INCL_DIR) -I$(CUDA_INCL_DIR) $(TREE) -std=gnu99 -Wno-unused-result -Werror
CUFLAGS=-dc $(OPTIMIZATION) -I$(INCL_DIR) $(TREE) -w
COMPUTE=$(if $(compute),$(compute),20)
SM=$(if $(sm),$(sm),20)
GENCODE=-gencode arch=compute_$(COMPUTE),code=compute_$(COMPUTE) -gencode arch=compute_$(COMPUTE),code=sm_$(SM)
LDFLAGS=-I$(INCL_DIR) -I$(CUDA_INCL_DIR) -L$(CUDA_LIB_DIR) -lcudart -lm

CU_SRCS = \
$(SRC_DIR)/kernel.cu \
$(SRC_DIR)/unit_1.cu

SRCS = \
$(SRC_DIR)/main.c

OBJS=$(addprefix $(BUILD_DIR)/,$(notdir $(SRCS:.c=.o)))
CU_OBJS=$(addprefix $(BUILD_DIR)/cuda/,$(notdir $(CU_SRCS:.cu=.o)))

EXECUTABLE=$(BASE_DIR)/vortex

# All Target
all: build_dir $(EXECUTABLE)

build_dir:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/cuda

$(BUILD_DIR)/cuda/%.o : $(SRC_DIR)/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: $(NVCC) Compiler'
	$(NVCC) $(CUFLAGS) $(GENCODE) -o $@ $<
	@echo 'Finished building: $<'
	@echo ' '

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: $(CC) Compiler'
	$(CC) $(CFLAGS) -o $@ $<
	@echo 'Finished building: $<'
	@echo ' '

$(BUILD_DIR)/culink.o: $(CU_OBJS)
	@echo 'building $@'
	$(NVCC) $(LDFLAGS) -dlink $(GENCODE) -o $@ $(CU_OBJS)
	@echo 'Finished building: $@'
	@echo ' '

$(EXECUTABLE): $(OBJS) $(BUILD_DIR)/culink.o
	@echo 'Building target: $@'
	@echo 'Invoking: $(LD) Linker'
	$(LD) $(OBJS) $(CU_OBJS) $(BUILD_DIR)/culink.o $(LDFLAGS) -o $(EXECUTABLE)
	@echo 'Finished building target: $@'
	@echo ' '

clean:
	$(RM) $(EXECUTABLE) $(BUILD_DIR)
