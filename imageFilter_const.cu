
// Based on CUDA SDK template from NVIDIA

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

//#include <assert.h>
//
//#ifndef gpuAssert
//#include <stdio.h>
//#define gpuAssert( condition ) { if( (condition) != 0 ) { fprintf( stderr, "\n FAILURE %s in %s, line %d\n", cudaGetErrorString(condition), __FILE__, __LINE__ ); exit( 1 ); } }
//#endif

// includes, project
#include <cutil_inline.h>

#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

__constant__ float f_const[3*3];

// loads filter coefficients from file fname,
// allocates memory through parray and stores width and height of filter through pwidth and pheight
int loadFilter(char* fname, float** parray, unsigned int *pwidth, unsigned int *pheight)
{
    FILE* fp;

    if( (fp=fopen(fname, "r")) == NULL)
    {
        fprintf(stderr,"Failed to open filter file %s\n",fname);
        return -1;
    }

    if(fscanf(fp,"%u %u",pwidth,pheight)!=2) {
        fprintf(stderr,"Failed to read header of filter file %s\n",fname);
        return -1;
    }

    *parray = (float *) malloc((*pwidth)*(*pheight)*sizeof(float));

    int i;
    for(i=0;i<(*pwidth)*(*pheight);i++)
    {
        if(fscanf(fp,"%f",(*parray+i))!=1) {
            fprintf(stderr,"Failed to read data of filter file %s\n",fname);
            return -1;
        }
    }

    fclose(fp);

    return 0;
}


// filter code to run on the host
void filterHost(unsigned int *h_idata, unsigned int w, unsigned int h,
        float* filter, unsigned int fw, unsigned int fh,
        unsigned int* reference)
{
    int i,j,k,l;

    int fh_2 = fh/2;
    int fw_2 = fw/2;

    for(i=0; i<h; i++) //height image
    {
        for(j=0; j<w; j++) //width image
        {
            float sum = 0;
            for(k=-fh_2; k<=fh_2; k++) //filter height
            {
                for(l=-fw_2; l<=fw_2; l++) //filter width
                {
                    if( (i+k >= 0) && (i+k < h))
                        if( (j+l >=0) && (j+l < w)) {
                            sum += h_idata[(i+k)*w + j+l]*filter[(k+fh/2)*fw + l+fw/2];
                        }

                }
            }
            reference[i*w+j] = min(max(sum,0),255);
        }
    }
}

// y = blockIdx.y * 32
// x = blockIdx.x * 32
__global__ void renderFilteredImage(unsigned int *in, unsigned int w, unsigned int h,
        unsigned int fw, unsigned int fh,
        unsigned int *out) {

    int i, j, k, l;
    int fw_2, fh_2;
    float sum = 0;

    fw_2 = fw/2;
    fh_2 = fh/2;

    i = threadIdx.y + blockIdx.y *blockDim.y;
    j = threadIdx.x + blockIdx.x *blockDim.x;

    for (k =- fh_2; k <= fh_2; k++) //filter height
    {
        for (l =- fw_2; l <= fw_2; l++) //filter width
        {
            if( (i+k >= 0) && (i+k < h))
                if( (j+l >=0) && (j+l < w)) {
                    sum += in[(i+k)*w + j+l] * f_const[(k+fh/2)*fw + l+fw/2];
                }
        }

        out[i*w+j] = min(max(sum,0),255);
    }
}


// filter code to run on the GPU
void filterDevice(unsigned int *h_idata, unsigned int w, unsigned int h,
        float* filter, unsigned int fw, unsigned int fh,
        unsigned int* h_odata)
{
    unsigned int *in, *out;
    //float *f;
    int size = w * h * sizeof(unsigned int);

    dim3 dimGrid(h/fh, w/fw);
    dim3 dimBlock(1, fw);

    cudaMalloc((void **)&in, size);
    //cudaMalloc((void **)&f, fw * fh * sizeof(float));
    cudaMalloc((void **)&out, size);

    cudaMemcpy(in, h_idata, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(f, filter, fw * fh * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(f_const, filter, fw * fh * sizeof(float), 0, cudaMemcpyHostToDevice);

    renderFilteredImage<<<dimGrid, dimBlock>>>(in, w, h, fw, fh, out);

    cudaMemcpy(h_odata, out, size, cudaMemcpyDeviceToHost);

    cudaFree(in);
    //cudaFree(f);
    cudaFree(out);
}

// print command line format
void usage(char *command)
{
    printf("Usage: %s [-h] [-d device] [-i inputfile] [-o outputfile] [-f filterfile]\n",command);
}

// main
int main( int argc, char** argv)
{
    // default command line options
    int deviceId = 1;
    char *fileIn="lena.pgm",*fileOut="lenaOut.pgm",*fileFilter="filter.txt";

    // parse command line arguments
    int opt;
    while( (opt = getopt(argc,argv,"d:i:o:f:h")) !=-1)
    {
        switch(opt)
        {

            case 'd':
                if(sscanf(optarg,"%d",&deviceId)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;

            case 'i':
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }

                fileIn = strdup(optarg);
                break;
            case 'o':
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                fileOut = strdup(optarg);
                break;
            case 'f':
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                fileFilter = strdup(optarg);
                break;
            case 'h':
                usage(argv[0]);
                exit(0);
                break;

        }
    }

    // select cuda device
    cutilSafeCall( cudaSetDevice( deviceId ) );

    // create events to measure host filter time and device filter time
    cudaEvent_t startH, stopH, startD, stopD;
    cudaEventCreate(&startH);
    cudaEventCreate(&stopH);
    cudaEventCreate(&startD);
    cudaEventCreate(&stopD);


    // allocate host memory
    unsigned int* h_idata=NULL;
    unsigned int h,w;
    //load pgm

    if (cutLoadPGMi(fileIn, &h_idata, &w, &h) != CUTTrue) {
        printf("Failed to load image file: %s\n", fileIn);
        exit(1);
    }

    //load filter
    float *filter;
    unsigned int fh, fw;
    if(loadFilter(fileFilter, &filter, &fw, &fh)==-1)
    {
        printf("Failed to load filter file: %s\n",fileFilter);
        exit(1);
    }

    // allocate mem for the result on host side
    unsigned int* h_odata = (unsigned int*) malloc( h*w*sizeof(unsigned int));
    unsigned int* reference = (unsigned int*) malloc( h*w*sizeof(unsigned int));

    // filter at host
    cudaEventRecord( startH, 0 );
    filterHost(h_idata, w, h, filter, fw, fh, reference);
    // filter at host
    cudaEventRecord( stopH, 0 );
    cudaEventSynchronize( stopH );

    // filter at GPU
    cudaEventRecord( startD, 0 );
    filterDevice(h_idata, w, h, filter, fw, fh, h_odata);
    cudaEventRecord( stopD, 0 );
    cudaEventSynchronize( stopD );

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    float timeH, timeD;
    cudaEventElapsedTime( &timeH, startH, stopH );
    printf( "Host processing time: %f (ms)\n", timeH);
    cudaEventElapsedTime( &timeD, startD, stopD );
    printf( "Device processing time: %f (ms)\n", timeD);

    // save output image
    if (cutSavePGMi(fileOut, reference, w, h) != CUTTrue) {
        printf("Failed to save image file: %s\n", fileOut);
        exit(1);
    }

    // cleanup memory
    cutFree( h_idata);
    free( h_odata);
    free( reference);
    free( filter);

    cutilDeviceReset();
}
