# Run inference to generate captions.
bazel-bin/im2txt/run_vid_inference \
  --checkpoint_path="/home/sean/Desktop/OUIRL-im2txt/pretrained_model/model.ckpt-2000000" \
  --vocab_file="/home/sean/Desktop/OUIRL-im2txt/pretrained_model/word_counts.txt" \
  --input_files="$1" \
