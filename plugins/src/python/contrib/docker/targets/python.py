# coding=utf-8
# Copyright 2015 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from os import path

from pants.backend.python.targets.python_binary import PythonBinary
from contrib.docker.targets.base import DockerTargetBase


class DockerPythonTarget(DockerTargetBase):
    """DockerPythonTarget"""

    def __init__(self, **kwargs):
        super(DockerPythonTarget, self).__init__(**kwargs)

    @classmethod
    def binary_target_type(cls):
        return PythonBinary

    @classmethod
    def default_dockerfile(cls):
        """Returns targets default Dockerfile"""
        python_images_dir = path.join(
            path.dirname(__file__), '../images/python')
        dockerfile = path.normpath(path.join(python_images_dir, 'Dockerfile'))
        return dockerfile
