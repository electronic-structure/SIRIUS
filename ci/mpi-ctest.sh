#!/bin/bash -e


if [[ $SLURM_JOB_PARTITION == "mi300" ]];
then
    export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
fi


exec "$@"
