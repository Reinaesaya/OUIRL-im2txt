# Build the inference binary.
bazel build -c opt im2txt/run_inference
bazel build -c opt im2txt/run_vid_inference