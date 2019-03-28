python_binary(
  name = 'keras-tfds-example',
  source = 'main.py',
  dependencies = [
    '3rdparty/python:tensorflow',
    '3rdparty/python:tensorflow-datasets',
  ],
)

python_docker(
  name = 'docker',
  binary = ':keras-tfds-example'
)