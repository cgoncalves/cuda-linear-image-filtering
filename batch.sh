#!/usr/bin/env bash

IMAGE_FILTER="imageFilter"
IMAGE_FILTER_CONST="imageFilter_const"

LOOP=5
DEVICE=1
FILTER="fedges.txt"
INPUT="lena.pgm"
OUTPUT="lenaOut.pgm"

reset()
{
    host_sum=0
    device_sum=0
}

run_seq()
{
    reset

    for i in `seq 1 $LOOP`
    do
        echo "Iteraction $i/$LOOP"
        cmd=`./$1 -d $DEVICE -f $FILTER -i $INPUT -o $OUTPUT`
        host_time=`echo $cmd | awk 'NR==1 {printf $4}'`
        device_time=`echo $cmd | awk 'NR==1 {printf $9}'`

        host_sum=`echo $host_time + $host_sum | bc`
        device_sum=`echo $device_time + $device_sum | bc`
    done

    avg[1]=`echo "$host_sum/$LOOP" | bc -l`
    avg[2]=`echo "$device_sum/$LOOP" | bc -l`
}

print_results()
{
    echo
    echo "Host average processing time: "
    echo "$host_sum/$LOOP" | bc -l

    echo
    echo "Device average processing time: "
    echo "$device_sum/$LOOP" | bc -l
}

speedup()
{
    echo "Speedup: "
    echo "$2/$1" | bc -l
}

# Check first if $INPUT file exists
if [ ! -f $INPUT ]
then
    echo "File $INPUT does not exist! Aborting..."
    exit
fi

# Run $IMAGE_FILTER
run_seq $IMAGE_FILTER
IMAGE_FILTER_HOST_AVG=${avg[1]}
IMAGE_FILTER_DEVICE_AVG=${avg[2]}
print_results $IMAGE_FILTER_HOST_AVG $IMAGE_FILTER_DEVICE_AVG

# Run $IMAGE_FILTER_CONST
run_seq $IMAGE_FILTER_CONST
IMAGE_FILTER_CONST_HOST_AVG=${avg[1]}
IMAGE_FILTER_CONST_DEVICE_AVG=${avg[2]}
print_results $IMAGE_FILTER_CONST_HOST_AVG $IMAGE_FILTER_CONST_DEVICE_AVG

# Calculate and output speedup
speedup $IMAGE_FILTER_DEVICE_AVG $IMAGE_FILTER_CONST_DEVICE_AVG

