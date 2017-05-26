if [ $# -eq 0 ]
then 
  echo "give path to image is missing"
  echo "example: ./process_image.sh imgs/bikes.jpg"
  exit 3
fi

# Run inference to generate captions.
bazel-bin/im2txt/run_inference \
  --checkpoint_path="/home/sean/Desktop/OUIRL-im2txt/pretrained_model/model.ckpt-2000000" \
  --vocab_file="/home/sean/Desktop/OUIRL-im2txt/pretrained_model/word_counts.txt" \
  --input_files="$1"
