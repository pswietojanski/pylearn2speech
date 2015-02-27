#!/bin/bash

#. /exports/work/inf_hcrc_cstr_students/s1136550/software/path.sh

#set this manually
GPU_LOCK_PATH="$HOME/repos/tnets-trunk/utils/gpu_lock/"

if [ $# == 0 ] ; then
  echo '-----------------------------------------------------------'
  echo "Usage: `basename $0` [OPTIONS] script_name.py arg0 arg1 ..."
  echo ''
  echo 'By default, the script will try to reserve and run the code making use of GPUs or will terminate otherwise. To change this behaviour use one of the following options.'
  echo ''
  echo '--any - run the code using any available device (when the GPUs are not available, the code will run on the CPU only)'
  echo '--cpu - run the code on the CPU only'
  echo '--gpu-id <id> - try to reserve specific GPU board'
  echo '--wait - wait for the GPU'
  echo '--longjob - Runs the script as the long time job (max 28 days) automatically renewing Kerberos tickets for afs access (DICE specific, makes use of preinstalled 'longjob' script)'
  echo '--nohup - Runs the script as nohup'
  echo '--mode - Theano mode. Supported flags: FR (FAST_RUN), FC (FAST_COMPILE), PM (PROFILE_MODE) and PMM (PROFILE_MODE + Memory) (default: FAST_RUN)' 
  echo '--use-sge - Assumes the use of Sun Grid Engine and existance of TASK_ID flag. Modify complilation directory so the compilation processes do not lock each other.'
  echo '--numlib - atlas or mkl or none (default). Sets some sensible default libraries to use. Modification of these may be required.'
  echo '--async - makes the GPU computations asynchronous'
  echo '--cnn-conf - provide a cnn conf to source '
  echo '----------------------------------------------------------'
  
  exit 0
fi

#DEFAULT NUMFLAGS, FEEL FREE TO MODIFY TO YOUR NEEDS
MKLF="-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -lguide -liomp5 -lmkl_mc -lmkl_def"
ATLASF="-lf77blas -latlas"

#defaults
DEVICE='gpu'
DEVICE_ID=-1 #any
CPU_ONLY=0
WAIT_FOR_GPU=0
FORCE_RUN=0
LONG_JOB=0
NOHUP=""
WAIT_TIME=2 #in minutes
MODE=FAST_RUN
USE_SGE=0
TMP_COMPILE=0
NUMLIB=
NUMFLAGS=""
ASYNC=True #will be set allow_gc=$ASYNC. False results in asynchronous mode
CNN_CONF=""
OMP_NUM_THREADS=1
USE_OPENMP=False
#---------

while [ $# -gt 0 ]; do
  case "${1# *}" in  # ${1# *} strips any leading spaces from the arguments
    --any) FORCE_RUN=1; shift ;;
    --cpu) CPU_ONLY=1; shift ;;
    --gpu-id) shift; DEVICE_ID=$1; shift ;;
    --wait) WAIT_FOR_GPU=1; shift ;;
    --longjob) LONG_JOB=1; shift ;;
    --nohup) NOHUP="nohup"; shift;;
    --mode) shift; MODE=$1; shift ;;
    --job-id) shift; JOB_ID=$1; shift ;;
    --use-sge) USE_SGE=1; shift ;;
    --tmp-compile) TMP_COMPILE=1; shift ;;
    --numlib) shift; NUMLIB=$1; shift;;
    --async) ASYNC=False; shift ;;
    --cnn-conf) shift; CNN_CONF=$1; shift;;
    -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
    *)   break ;;   # end of options: interpreted as python script to run
  esac
done

#sleep .$[ ( $RANDOM % 10 ) + 1 ]s

if [ ! -z "$CNN_CONF" ]; then
. ./$CNN_CONF
fi

if [ "$NUMLIB" == "atlas" ]; then
  NUMFLAGS=$ATLASF
elif [ "$NUMLIB" == "mkl" ]; then
  NUMFLAGS=$MKLF
else
  NUMFLAGS=""
fi

if [ $# -lt 1 ]; then
  echo 'Nothing to run!'
  exit 1;
fi

CMD=$@

if [ $CPU_ONLY -eq 1 ] ; then
  #echo 'Running on CPU.'
  DEVICE='cpu'
  OMP_NUM_THREADS=1
fi

# when running on local GPU machines, not managed by the grid,
# obtain GPU locks manually
if [ "$DEVICE" == 'gpu' ]; then
		
  ID=`"$GPU_LOCK_PATH"gpu_lock.py --id`
  while [ $WAIT_FOR_GPU -eq 1 ] && [ $ID -eq -1 ] ; do
    echo "All the GPU boards are currently busy. Will retry in $WAIT_TIME minutes..." 1>&2
    sleep $(($WAIT_TIME*60+($RANDOM % 11)))
    ID=`"$GPU_LOCK_PATH"gpu_lock.py --id`
  done

  if [ $ID -ne -1 ] ; then
    #echo "GPU Board $ID has been reserved."
    DEVICE="$DEVICE$ID";
  elif [ $FORCE_RUN -eq 1 ] ; then
    echo "All the GPUs are currently busy. Running on CPU." 1>&2
    DEVICE='cpu'
  else
    echo 'None of the GPU boards is currently available. Sorry.' 1>&2
    exit 0
  fi
fi

if [ $OMP_NUM_THREADS -gt 1 ]; then
  USE_OPENMP=True
fi

THEANO_FLAGS="device=$DEVICE, openmp=$USE_OPENMP, floatX=float32, force_device=True, print_active_device=False, nvcc.fastmath=True, exception_verbosity=high, mode=$MODE, allow_gc=$ASYNC"
THEANO_FLAGS="$THEANO_FLAGS, blas.ldflags=$NUMFLAGS"

if [ "$MODE" == "PROFILE_MODE" ]; then
  export CUDA_LAUNCH_BLOCKING=1
fi

# we do some special assumptions on the grid - i)compiling in tmp, and ii) spearate compilation dir per task
if [ $USE_SGE -eq 1 ]; then
  BASE_COMPILEDIR='/tmp'

  JOB_HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 2 | head -n 1)
  if [[ -z "$SGE_TASK_ID" || "$SGE_TASK_ID" = "undefined" ]]; then
      COMPILEDIR_FORMAT="theano_sgd_compiledir_${USER}_${JOB_ID}_${JOB_HASH}"
  else
      COMPILEDIR_FORMAT="theano_sgd_compiledir_${USER}_$JOB_ID.$SGE_TASK_ID"
  fi
  
  THEANO_FLAGS="$THEANO_FLAGS, base_compiledir=$BASE_COMPILEDIR, compiledir_format=$COMPILEDIR_FORMAT"
fi

if [ $TMP_COMPILE -eq 1 ] && [ $USE_SGE -eq 0 ]; then

  BASE_COMPILEDIR='/tmp'
  JOB_HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)
  COMPILEDIR_FORMAT="theano_compiledir_${USER}_$JOB_ID"

  THEANO_FLAGS="$THEANO_FLAGS, base_compiledir=$BASE_COMPILEDIR, compiledir_format=$COMPILEDIR_FORMAT"

fi

export THEANO_FLAGS OMP_NUM_THREADS

#echo "Running the command : $CMD"
$CMD



