import platform


use_gpu = platform.system() == "Linux"
third_party_tf = '3rdparty/python:tensorflow'
if use_gpu:
  third_party_tf = '3rdparty/python:tensorflow-gpu'


python_binary(
  name = 'keras-tfds-example',
  dependencies = [
    third_party_tf,
    '3rdparty/python:tensorflow-datasets',
  ],
  source = 'main.py',
)
