
TARGETS = lib/libcutil_x86_64.a imageFilter 

all: $(TARGETS)

imageFilter: imageFilter.cu
	nvcc -arch=sm_13 -O3 -Icommon/inc imageFilter.cu -Llib -lcutil_x86_64 -o imageFilter

lib/libcutil_x86_64.a: 
	make -C common

clean:
	make -C common clean
	rm -f $(TARGETS)
