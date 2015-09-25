RM := rm -rf

CC=gcc
NVCC=nvcc
LD=gcc
INCL_DIR=./include
OPTIMIZATION=$(if $(DEBUG),-O0 -g,-O2)
CFLAGS=-c $(OPTIMIZATION) -I $(INCL_DIR) -std=gnu99 -Wno-unused-result -Werror
CUFLAGS=-dc $(OPTIMIZATION) -I $(INCL_DIR) -w
COMPUTE=$(if $(compute),$(compute),20)
SM=$(if $(sm),$(sm),20)
GENCODE=-gencode arch=compute_$(COMPUTE),code=compute_$(COMPUTE) -gencode arch=compute_$(COMPUTE),code=sm_$(SM)
LDFLAGS=-I $(INCL_DIR) -lcudart -lm

CU_SRCS = \
src/kernel.cu \
src/unit_1.cu

SRCS = \
src/main.c

OBJS=$(SRCS:.c=.o)
CU_OBJS=$(CU_SRCS:.cu=.o)

EXECUTABLE=vortex

# All Target
all: $(SRCS) $(CU_SRCS) $(EXECUTABLE)

%.o : %.cu
	@echo 'Building file: $<'
	@echo 'Invoking: $(NVCC) Compiler'
	$(NVCC) $(CUFLAGS) $(GENCODE) -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o : %.c
	@echo 'Building file: $<'
	@echo 'Invoking: $(CC) Compiler'
	$(CC) $(CFLAGS) -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

culink.o: $(CU_OBJS)
	@echo 'building $@'
	$(NVCC) $(LDFLAGS) -dlink $(GENCODE) -o src/$@ $(CU_OBJS)
	@echo 'Finished building: $@'
	@echo ' '

$(EXECUTABLE): $(OBJS) culink.o
	@echo 'Building target: $@'
	@echo 'Invoking: $(LD) Linker'
	$(LD) $(OBJS) $(CU_OBJS) src/culink.o $(LDFLAGS) -o $(EXECUTABLE)
	@echo 'Finished building target: $@'
	@echo ' '

clean:
	$(RM) $(EXECUTABLE) $(OBJS) $(CU_OBJS) src/culink.o
