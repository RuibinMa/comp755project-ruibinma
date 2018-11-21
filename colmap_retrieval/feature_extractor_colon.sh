#!/bin/bash

data_dir=$1

rm "$data_dir/database.db"

colmap feature_extractor \
	--database_path "$data_dir/database.db" \
	--image_path "$data_dir/images" \
	--ImageReader.camera_model "PINHOLE" \
	--ImageReader.single_camera "1" \
	--ImageReader.camera_params "374.1583,375.3094,333.3978,273.8418" \
	--SiftExtraction.domain_size_pooling "1"

