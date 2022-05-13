#!/bin/bash

filename=$1
nline=$2
proc=$3

if [[ -z ${nline} ]]; then
    nline=5
fi

nbprocmax=$(grep -e "USING DEFAULTS : lunout =" $filename | wc -l)

if [[ -z ${proc} ]] ; then
  for i in $(seq -f "%g" 0 9);
    do
      grep -e " ${i}:" $filename | tail -n $nline
    done
  for i in $(seq -f "%g" 10 $nbprocmax);
    do
      grep -e "${i}:" $filename | tail -n $nline
    done
else
   grep -e "$proc:" $filename | tail -n $nline
fi
