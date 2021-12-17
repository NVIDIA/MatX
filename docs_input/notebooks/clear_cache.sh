#!/bin/bash

for i in $(ls|grep ipynb); do
  jupyter nbconvert --clear-output --inplace $i;
done
