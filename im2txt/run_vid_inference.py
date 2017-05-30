# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for webcam video using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys


import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

import numpy as np
import cv2

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")

tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of video files.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
	# Build the inference graph.
	g = tf.Graph()
	with g.as_default():
		model = inference_wrapper.InferenceWrapper()
		restore_fn = model.build_graph_from_config(configuration.ModelConfig(),FLAGS.checkpoint_path)
	g.finalize()

	# Create the vocabulary.
	vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

	filenames = []
	for file_pattern in FLAGS.input_files.split(","):
		if tf.gfile.Glob(file_pattern) == []:
			tf.logging.info("Can't find file: "+str(file_pattern))
			sys.exit()
		filenames.extend(tf.gfile.Glob(file_pattern))

	recorded_videos = False
	if len(filenames)>0:
		recorded_videos = True
		tf.logging.info("Running caption generation on %d files matching %s",
		              len(filenames), FLAGS.input_files)
	else:
		tf.logging.info("Running caption generation on webcam video")

	with tf.Session(graph=g) as sess:
		# Load the model from checkpoint.
		restore_fn(sess)

		# Prepare the caption generator. Here we are implicitly using the default
		# beam search parameters. See caption_generator.py for a description of the
		# available beam search parameters.
		generator = caption_generator.CaptionGenerator(model, vocab)


		if not recorded_videos:		# Use camera at default port
			cap = cv2.VideoCapture(0)
			cap.set(cv2.CAP_PROP_FPS, 0.5)
			while(True):
				ret, frame = cap.read()

				frame_str = cv2.imencode('.jpg',frame)[1].tostring()
				print('------')
				captions = generator.beam_search(sess, frame_str)
				for i, caption in enumerate(captions):
					# Ignore begin and end words.
					sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
					sentence = " ".join(sentence)
					print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				cv2.imshow('frame',frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			cap.release()
			cv2.destroyAllWindows()

		else:											# Recorded video
			for filename in filenames:
				print(filename)
				print(type(filename))
				cap = cv2.VideoCapture(filename)
				#cap.set(cv2.CAP_PROP_FPS, 0.5)
				while(cap.isOpened()):
					ret, frame = cap.read()
					print("here2")

					frame_str = cv2.imencode('.jpg',frame)[1].tostring()
					print('------')
					captions = generator.beam_search(sess, frame_str)
					for i, caption in enumerate(captions):
						# Ignore begin and end words.
						sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
						sentence = " ".join(sentence)
						print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

					gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
					cv2.imshow('frame',frame)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break
				cap.release()
				cv2.destroyAllWindows()



if __name__ == "__main__":
	tf.app.run()
