#!/usr/bin/env bash

#IMAGE_FILTER_REF="imageFilter_v1"
#IMAGE_FILTER_TEST="imageFilter_v2"
#LOOP=1
#DEVICE=0
IMAGE_FILTER_REF="$1"
IMAGE_FILTER_TEST="$2"
LOOP=$3
DEVICE=$4

FILTER="filter.txt"
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
        echo -en "\rIteration $i/$LOOP"
        cmd=`./$1 -d $DEVICE -f $FILTER -i $INPUT -o $OUTPUT`
        host_time=`echo $cmd | awk 'NR==1 {printf $4}'`
        device_time=`echo $cmd | awk 'NR==1 {printf $9}'`

        host_sum=`echo $host_time + $host_sum | bc`
        device_sum=`echo $device_time + $device_sum | bc`
    done

    echo

    avg[1]=`echo "$host_sum/$LOOP" | bc -l`
    avg[2]=`echo "$device_sum/$LOOP" | bc -l`
}

print_results()
{
    echo -e "\nHost average processing time: "
    echo "$host_sum/$LOOP" | bc -l

    echo -e "\nDevice average processing time: "
    echo "$device_sum/$LOOP" | bc -l
}

speedup()
{
    echo -e "\nDate\t\t: `date`"
    echo -e "Iterations\t: $LOOP"
    echo -e "Ref ($IMAGE_FILTER_REF)"
    echo -e "\tRef CPU\t\t: $3"
    echo -e "\tRef GPU\t\t: $1"
    echo -e "Test ($IMAGE_FILTER_TEST)"
    echo -e "\tTest CPU\t: $4"
    echo -e "\tTest GPU\t: $2"
    echo -en "\nSpeedup CPU\t: "
    echo "$3/$4" | bc -l
    echo -en "Speedup GPU\t: "
    echo "$1/$2" | bc -l
}

# Check first if $INPUT file exists
if [ ! -f $INPUT ]
then
    echo "File $INPUT does not exist! Aborting..."
    exit
fi

# Run $IMAGE_FILTER_REF
echo "Running reference: $IMAGE_FILTER_REF"
run_seq $IMAGE_FILTER_REF
IMAGE_FILTER_REF_HOST_AVG=${avg[1]}
IMAGE_FILTER_REF_DEVICE_AVG=${avg[2]}
#print_results $IMAGE_FILTER_REF_HOST_AVG $IMAGE_FILTER_REF_DEVICE_AVG

# Run $IMAGE_FILTER_TEST
echo "Running test: $IMAGE_FILTER_TEST"
run_seq $IMAGE_FILTER_TEST
IMAGE_FILTER_TEST_HOST_AVG=${avg[1]}
IMAGE_FILTER_TEST_DEVICE_AVG=${avg[2]}
#print_results $IMAGE_FILTER_TEST_HOST_AVG $IMAGE_FILTER_TEST_DEVICE_AVG

# Calculate and output speedup
speedup $IMAGE_FILTER_REF_DEVICE_AVG $IMAGE_FILTER_TEST_DEVICE_AVG $IMAGE_FILTER_REF_HOST_AVG $IMAGE_FILTER_TEST_HOST_AVG
echo -e "-------------------------------------------------\n"

