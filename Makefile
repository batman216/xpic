EXE = xpic
CXX = nvcc

NVCFlAG = --expt-relaxed-constexpr --extended-lambda
CXXFLAG = -std=c++20 -Xcompiler -fopenmp
LDFLAG  = -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nccl/include           \
          -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/12.0/nccl/lib               \
          -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/hpcx/hpcx-2.14/ompi/include \
          -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/hpcx/hpcx-2.14/ompi/lib     \
          -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/include                     \
          -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/lib                         \
          -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/hpcx/hpcx-2.14/ompi/include \
          -I/opt/hdf5/include -L/opt/hdf5/lib -lhdf5                                    \
          -lmpi -lopen-rte -lopen-pal -lomp -lnccl -lcusparse
ifeq ($(mode),debug)
	CXXFLAG += -g
endif

SRC = src
BLD = bld
RUN = run
ILD = include

INPUT = xpic.in

$(shell mkdir -p ${BLD})


CPP = ${wildcard ${SRC}/*.cpp}
CU  = ${wildcard ${SRC}/*.cu}
HPP = ${wildcard ${ILD}/*.hpp}

CUOBJ = ${patsubst ${SRC}/%.cu,${BLD}/%.o,${CU}}
CPPOBJ = ${patsubst ${SRC}/%.cpp,${BLD}/%.o,${CPP}}


${BLD}/${EXE}: ${CUOBJ} ${CPPOBJ}
	${CXX} $^ ${NVCFlAG} ${LDFLAG} -o $@

${BLD}/%.o: ${SRC}/%.cu 
	${CXX} ${CXXFLAG} ${LDFLAG} ${NVCFlAG} -c $< -o $@

${BLD}/%.o: ${SRC}/%.cpp 
	${CXX} ${CXXFLAG} ${LDFLAG} ${NVCFlAG} -c $< -o $@


run: ${BLD}/${EXE} ${INPUT}
	mkdir -p ${RUN} && cp $^ ${RUN} && cd ${RUN} && ./${EXE}

clean:
	rm -rf ${BLD}

show:
	echo ${CPP} ${OBJ}
