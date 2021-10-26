#!/bin/bash


mkdir temp

printf "temp folder created\n"

for entry in "Notebooks"/*
do
  printf "$entry"
  
  jupyter nbconvert --output-dir="./temp" --to script "$entry"
  printf "\n"

done

pipreqs . --force
printf "requirements.txt is generated!"
rm -rf temp
printf "temp folder removed\n"

