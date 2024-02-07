#!/bin/bash

mkdir zip_file
mkdir images

while IFS= read -r line;do
  echo $line
  wget -c $line -P ./zip_file/
done < URLlist.txt

ls zip_file | grep ".zip" | while read zip_file_name
do
  unzip zip_file/$zip_file_name -d images
done

ls images | grep ".png" | while read img_name
do
  echo $img_name
  convert    images/$img_name    -resize     512x512!    images/$img_name
  break
done