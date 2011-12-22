
TARGETS = lib/libcutil_x86_64.a imageFilter imageFilter_const

all: $(TARGETS)

imageFilter: imageFilter.cu
	nvcc -arch=sm_13 -O3 -Icommon/inc imageFilter.cu -Llib -lcutil_x86_64 -o imageFilter

imageFilter_const: imageFilter_const.cu
	nvcc -arch=sm_13 -O3 -Icommon/inc imageFilter_const.cu -Llib -lcutil_x86_64 -o imageFilter_const

lib/libcutil_x86_64.a: 
	make -C common

clean:
	make -C common clean
	rm -f $(TARGETS)
