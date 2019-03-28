# coding=utf-8
# Copyright 2016 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).
import os
import shutil

from pants.backend.python.targets.python_binary import PythonBinary
from pants.base.exceptions import TaskError

from contrib.docker.tasks.build_base import BuildTask


class PythonBuildTask(BuildTask):
    """PythonBuildTask"""

    def _check_dependency_type(self, dependency):
        if not isinstance(dependency, PythonBinary):
            raise TaskError(
                'docker_python_image needs a python_binary as a dependency')

    def _prepare_directory(self, tmpdir, target):
        archive_mapping = self.context.products.get('pex_archives').get(target.binary)        
        basedir, paths = list(archive_mapping.items())[0]
        path = paths[0]
        archive_path = os.path.join(basedir, path)
        dockerfile = target.dockerfile
        
        shutil.copy(archive_path, os.path.join(tmpdir, 'app.pex'))
        shutil.copy(dockerfile, os.path.join(tmpdir, 'Dockerfile'))
