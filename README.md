# Naive Bayes classifiers in TensorFlow

<img src="tf_iris.png" style="max-width: 80%; display: block; margin-left: auto; margin-right: auto;" />

A simple [Naive Bayes classifier]() in TensorFlow 1.4. It's a tidy
demonstration of [`tf.distributions`](https://www.tensorflow.org/api_docs/python/tf/distributions) and some unusual tensor operations.

For more information, you can read the [blog post](http://nicolovaligi.com/naive-bayes-tensorflow.html).

## Getting started

Prepare the Python environment:

```
# Create a new virtualenv
mkvirtualenv env
source env/bin/activate
# Install requirements
pip install -r requirements.txt
```

And run the classifier:

```
python tf_iris.py
```
