##############################
# AutoEnc Makefile
##############################

#INSTALLROOT=$(PWD)

CC=gcc
CPP=g++
NVCC=nvcc
INSTALL=install
BINARY=AutoEnc
#BIN=/usr/local/bin
#VERSION = 0.0.1

CPPFLAGS = -O3 -std=c++11
OPENCVLIBS = -I /usr/include/opencv4/ -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc
CPPCUDA = -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lcufft #-lcublas_device
CUDAFLAGS = -gencode arch=compute_50,code=sm_50 --use_fast_math -lineinfo
#CUDAFLAGS = -ccbin g++ -m64 -gencode arch=compute_50,code=sm_50
#for multiple GPU architectures use instead
#CUDAFLAGS = -ccbin g++ -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70



SRC_DIR= ./source
OBJ_DIR= ./obj
SRC_CPP := $(wildcard $(SRC_DIR)/*.cpp)
SRC_CU  := $(wildcard $(SRC_DIR)/*.cu)
SRC_FILES:= $(SRC_CPP) $(SRC_CU)
#OBJ_FILES := $(wildcard $(OBJ_DIR)/*.o
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_CPP)) \
             $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC_CU))
		
all:	AutoEnc
clean:
	@echo "\033[0;33mCleaning up obj directory."
	@rm -f -r $(OBJ_DIR)
print:
	@echo $(SRC_FILES)
	@echo $(OBJ_FILES)

#Output Decorations
COM_COLOR   = \033[0;34m
OBJ_COLOR   = \033[0;36m
OK_COLOR    = \033[0;32m
ERROR_COLOR = \033[0;31m
WARN_COLOR  = \033[0;33m
NO_COLOR    = \033[m
OK_STRING    = "[OK]"
ERROR_STRING = "[ERROR]"
WARN_STRING  = "[WARNING]"
COM_STRING   = "Compiling.........."
EXE_STR      = "Creating" 
EXE_STR1  =    "Executable"
OBJ_NAME = $<
EXE_NAME = $(@F)
#Info printing function 1
define infos
printf "%b" "$(COM_COLOR)$(COM_STRING) $(OBJ_COLOR)$(OBJ_NAME)$(NO_COLOR)\r"; \
$(1) 2> $@.log; \
RESULT=$$?; \
if [ $$RESULT -ne 0 ]; then \
  printf "%-60b%b" "$(COM_COLOR)$(COM_STRING)$(OBJ_COLOR) $(OBJ_NAME)" "$(ERROR_COLOR)$(ERROR_STRING)$(NO_COLOR)\n"   ; \
elif [ -s $@.log ]; then \
  printf "%-60b%b" "$(COM_COLOR)$(COM_STRING)$(OBJ_COLOR) $(OBJ_NAME)" "$(WARN_COLOR)$(WARN_STRING)$(NO_COLOR)\n"   ; \
else  \
  printf "%-60b%b" "$(COM_COLOR)$(COM_STRING)$(OBJ_COLOR) $(OBJ_NAME)" "$(OK_COLOR)$(OK_STRING)$(NO_COLOR)\n"   ; \
fi; \
cat $@.log; \
rm -f $@.log; \
exit $$RESULT
endef
#Info printing function 2
define infos_exe
printf "%b" "$(COM_COLOR)$(EXE_STR) $(EXE_STR1) $(OBJ_COLOR)$(EXE_NAME)$(NO_COLOR)\r"; \
$(1) 2> $@.log; \
RESULT=$$?; \
if [ $$RESULT -ne 0 ]; then \
  printf "%-60b%b" "$(COM_COLOR)$(EXE_STR)$(EXE_STR1)$(OBJ_COLOR) $(EXE_NAME)" "$(ERROR_COLOR)$(ERROR_STRING)$(NO_COLOR)\n"   ; \
elif [ -s $@.log ]; then \
  printf "%-60b%b" "$(COM_COLOR)$(EXE_STR)$(EXE_STR1)$(OBJ_COLOR) $(EXE_NAME)" "$(WARN_COLOR)$(WARN_STRING)$(NO_COLOR)\n"   ; \
else  \
  printf "%-60b%b" "$(COM_COLOR)$(EXE_STR)$(EXE_STR1)$(OBJ_COLOR) $(EXE_NAME)" "$(OK_COLOR)$(OK_STRING)$(NO_COLOR)\n"   ; \
fi; \
cat $@.log; \
rm -f $@.log; \
exit $$RESULT
endef

#COMPILATION
AutoEnc: $(OBJ_FILES)
#@printf "$(COM_COLOR)Creating Executable $(OBJ_COLOR)$@$(NO_COLOR)\n";
	@$(call infos_exe,$(CPP) -o $@ $^ $(OPENCVLIBS) $(CPPCUDA) $(CPPFLAGS))

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
#@printf "$(COM_COLOR)CPP Compiling...... $(OBJ_COLOR)$<$(NO_COLOR)\n";
	@$(call infos,$(CPP) -c -o $@ $< $(CPPFLAGS) $(OPENCVLIBS))

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
#@printf "$(COM_COLOR)CUDA Compiling..... $(OBJ_COLOR)$<$(NO_COLOR)\n";
	@$(call infos,$(NVCC) $(CUDAFLAGS) -c -o $@ $<)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)


#install: AutoEnc
#	$(INSTALL) -s -m 755 -g root -o root $(BINARY) $(BIN) 
#	rm -f $(BIN)/$(BINARY)
