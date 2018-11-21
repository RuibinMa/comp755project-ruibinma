#!/bin/bash

data_dir=$1

colmap vocab_tree_retriever \
	--database_path "$data_dir/left.db" \
	--vocab_tree_path "/playpen/vocab_tree/vocab_tree-262144.bin" \
	--num_images_after_verification "140" \
	> "$data_dir/colmap_retriever_result.txt"