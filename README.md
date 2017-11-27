# Active-Learning-for-Neural-Networks
This re-implemented the "EGL-word" method proposed in AAAI 2017 paper "Active Discriminative Text Representation Learning" in Tensorflow (the original implementation was in Theano). It compared three active learning methods "random", "entropy" and "EGL" on a sentiment analysis dataset.

First run "python process_data.py path/to/pre-trained-embedding" to generate the processed data.

Then run "python active_learning.py --AL_method=active-learning-method", where "active-learning-method" should be one of "random", "entropy" or "EGL".


