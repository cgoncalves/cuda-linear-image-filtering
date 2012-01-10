
// Based on CUDA SDK template from NVIDIA

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

// includes, project
#include <cutil_inline.h>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

#define block_x_extra(x) (x%BLOCK_DIM_X != 0? 1 : 0)
#define block_y_extra(y) (y%BLOCK_DIM_Y != 0? 1 : 0)

extern __shared__ float sharedMem[];

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

__global__ void renderFilteredImage(unsigned int *idata, unsigned int w, unsigned int h,
        float *filter, unsigned int fw, unsigned int fh,
        unsigned int *odata) {

    int i, j, k, l;
    unsigned int blockPadSize, fw_2, fh_2, tmp;
    float *sImage, *sFilter;
    float sum;

    fw_2 = fw/2;
    fh_2 = fh/2;

    blockPadSize = blockDim.x + (fw_2*2);

    sFilter = sharedMem;
    sImage = sharedMem + fw*fh;

    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;

    /*
     * Fill sFilter with data
     * Only the first fw*fh threads are needed to perform this operation!
    */
    if (threadIdx.x < fw && threadIdx.y < fh) {
        tmp = (threadIdx.x * fw) + threadIdx.y;
        sFilter[tmp] = filter[tmp];
    }

    if (i < h && j < w) {

        // current pixel
        sImage[threadIdx.y + fh_2*blockPadSize + threadIdx.x + fw_2] = idata[i*w + j];

        /*
         * fill sImage with padding data
        `*/

        // top left corner
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            for (k=-fh_2; k<=0; k++) //filter height
                for (l=-fw_2; l<=0; l++) //filter width
                    if ((i+k >= 0) && (i+k < h) && (j+l >=0) && (j+l < w))
                        sImage[(threadIdx.y + fh_2+k)*blockPadSize + threadIdx.x + fw_2+l] = idata[(i+k)*w + j+l];
        }

        // bottom left corner
        else if (threadIdx.x == 0 && (threadIdx.y == (blockDim.y -1)))
        {
            for (k=0; k<=fh_2; k++) //filter height
                for (l=-fw_2; l<=0; l++) //filter width
                    if ((i+k >= 0) && (i+k < h) && (j+l >=0) && (j+l < w))
                        sImage[(threadIdx.y + fh_2+k)*blockPadSize + threadIdx.x + fw_2+l] = idata[(i+k)*w + j+l];
        }

        // remaining left side
        else if (threadIdx.x == 0)
        {
            for (l=-fw_2; l<=0; l++) //filter height
                if ((j+l >=0) && (j+l < w))
                    sImage[threadIdx.y + fh_2*blockPadSize + threadIdx.x + fw_2+l] = idata[i*w + j+l];
        }

        // top right corner
        else if ((threadIdx.x == (blockDim.x -1)) && (threadIdx.y == 0))
        {
            for (k=-fh_2; k<=0; k++) //filter height
                for (l=0; l<=fw_2; l++) //filter width
                    if ((i+k >= 0) && (i+k < h) && (j+l >=0) && (j+l < w))
                        sImage[(threadIdx.y + fh_2+k)*blockPadSize + threadIdx.x + fw_2+l] = idata[(i+k)*w + j+l];
        }

        // bottom right corner
        else if ((threadIdx.x == (blockDim.x -1)) && (threadIdx.y == (blockDim.y -1)))
        {
            for (k=0; k<=fh_2; k++) //filter height
                for (l=0; l<=fw_2; l++) //filter width
                    if ((i+k >= 0) && (i+k < h) && (j+l >=0) && (j+l < w))
                        sImage[(threadIdx.y + fh_2+k)*blockPadSize + threadIdx.x + fw_2+l] = idata[(i+k)*w + j+l];
        }

        // remaining right side
        else if (threadIdx.x == (blockDim.x -1))
        {
            for (l=0; l<=fw_2; l++) //filter height
                if ((j+l >=0) && (j+l < w))
                    sImage[threadIdx.y + fh_2*blockPadSize + threadIdx.x + fw_2+l] = idata[i*w + j+l];
        }

        // top excluding corners
        else if ((threadIdx.y == 0) && (threadIdx.x != 0) && (threadIdx.x != (blockDim.x - 1)))
        {
            for (k=-fh_2; k<=0; k++) //filter width
                if ((i+k >= 0) && (i+k < h))
                    sImage[(threadIdx.y + fh_2+k)*blockPadSize + threadIdx.x + fw_2] = idata[(i+k)*w + j];
        }

        // bottom excluding corners
        else if ((threadIdx.y == (blockDim.y - 1)) && (threadIdx.x != 0) && (threadIdx.x != (blockDim.x - 1)))
        {
            for (k=0; k<=fh_2; k++) //filter width
                if ((i+k >= 0) && (i+k < h))
                    sImage[(threadIdx.y + fh_2+k)*blockPadSize + threadIdx.x + fw_2] = idata[(i+k)*w + j];
        }

        /*
         * Wait for all threads.
         *
         * Wait for all threads processing the code above first so that we can
         * ensure all shared data (filter and image) are synced.
        */
        __syncthreads();

        sum = 0;
        for (k=-fh_2; k<=fh_2; k++) { //filter height
            for (l=-fw_2; l<=fw_2; l++) { //filter width
                if ( (i+k >= 0) && (i+k < h))
                    if ( (j+l >=0) && (j+l < w))
                        sum += sImage[(threadIdx.y + fh_2+k)*blockPadSize + threadIdx.x + fw_2+l] * sFilter[(k+fh_2)*fw + l+fw_2];
            }
        }

        odata[i*w + j] = min(max((unsigned int)sum,0),255);
    }
}

void filterDevice(unsigned int *h_idata, unsigned int w, unsigned int h,
                  float* filter, unsigned int fw, unsigned int fh,
                  unsigned int* h_odata)
{
    unsigned int *d_idata, *d_odata, data_size, dim_grid_x, dim_grid_y, sharedSize, filter_size;
    float *f;

    data_size = w * h * sizeof(unsigned int);
    filter_size = fw * fh * sizeof(float);

    dim_grid_x = w/BLOCK_DIM_X + block_x_extra(w);
    dim_grid_y = h/BLOCK_DIM_Y + block_y_extra(h);

    dim3 dimGrid(dim_grid_x, dim_grid_y);
    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);

    // how much shared memory we need to reserve (= filter size + image size with padding)
    sharedSize = filter_size + ( (BLOCK_DIM_Y + fh-1) * (BLOCK_DIM_X + fw-1) );

    // memory allocation
    cudaMalloc((void**)&d_idata, data_size);
    cudaMalloc((void**)&f, filter_size);
    cudaMalloc((void**)&d_odata, data_size);

    // copy image and filter to device (CPU->GPU)
    cudaMemcpy(d_idata, h_idata, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(f, filter, filter_size, cudaMemcpyHostToDevice);

    // render filtered image on GPU
    renderFilteredImage<<<dimGrid, dimBlock, sharedSize>>>(d_idata, w, h, f, fw, fh, d_odata);

    // copy result from device to host (GPU->CPU)
    cudaMemcpy(h_odata, d_odata, data_size, cudaMemcpyDeviceToHost);

    // free allocated memory
    cudaFree(f);
    cudaFree(d_idata);
    cudaFree(d_odata);
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
    int deviceId = 0;
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

    // save output image
    if (cutSavePGMi("lenaGPU.pgm", reference, w, h) != CUTTrue) {
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
