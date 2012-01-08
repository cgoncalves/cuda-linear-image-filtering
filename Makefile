
TARGETS = lib/libcutil_x86_64.a imageFilter_v1 imageFilter_v2 imageFilter_v3 imageFilter_v4 imageFilter_shared

all: $(TARGETS)

imageFilter_v1: imageFilter_v1.cu
	nvcc -arch=sm_13 -O3 -Icommon/inc imageFilter_v1.cu -Llib -lcutil_x86_64 -o imageFilter_v1

imageFilter_v2: imageFilter_v2.cu
	nvcc -arch=sm_13 -O3 -Icommon/inc imageFilter_v2.cu -Llib -lcutil_x86_64 -o imageFilter_v2

imageFilter_v3: imageFilter_v3.cu
	nvcc -arch=sm_13 -O3 -Icommon/inc imageFilter_v3.cu -Llib -lcutil_x86_64 -o imageFilter_v3

imageFilter_v4: imageFilter_v4.cu
	nvcc -arch=sm_13 -O3 -Icommon/inc imageFilter_v4.cu -Llib -lcutil_x86_64 -o imageFilter_v4

imageFilter_shared: imageFilter_shared.cu
	nvcc -arch=sm_13 -O3 -Icommon/inc imageFilter_shared.cu -Llib -lcutil_x86_64 -o imageFilter_shared

lib/libcutil_x86_64.a: 
	make -C common

clean:
	make -C common clean
	rm -f $(TARGETS)
