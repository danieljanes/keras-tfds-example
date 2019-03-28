"""Docker image task
This task will create docker images from docker files in the different services
"""
import sys
from abc import abstractmethod

import docker

from pants.base.build_environment import get_buildroot
from pants.task.task import Task
from pants.base.workunit import WorkUnit, WorkUnitLabel
from pants.base.exceptions import TaskError
from pants.util.contextutil import temporary_dir

from contrib.docker.targets.base import DockerTargetBase


class BuildTask(Task):
    """BuildTask"""

    @classmethod
    def prepare(cls, _, round_manager):
        """TODO: find out and describe what this does"""
        round_manager.require_data('deployable_archives')

    @classmethod
    def register_options(cls, register):
        """Register task with options

        You can use the below defined options by calling self.get_options().property_name

        Example: self.get_options().push
        """
        default_namespace = get_buildroot().split('/')[-1]

        register('--namespace', default=default_namespace, type=str, help='image namespace for <namespace>/<image_name>')
        
        register('--registry', default=None, type=str, help='image registry')

        register(
            '--push', default=False, type=bool, help='push image to registry')

    @staticmethod
    def _is_target_type(target):
        """Used to check if target is instance of DockerImage"""
        return isinstance(target, DockerTargetBase)

    @abstractmethod
    def _check_dependency_type(self, dependency):
        pass

    @abstractmethod
    def _prepare_directory(self, tmpdir, target):
        """Prepares directory content
        Directory should when the function finishes contain the PEX and Dockerfile
        """

    def _build_image(self, target):
        with temporary_dir(cleanup=False) as tmpdir:
            self._prepare_directory(tmpdir, target)
            self._run_docker_build(tmpdir, target)

    def _run_docker_build(self, tmpdir, target):
        with self.context.new_workunit(
                name='create-image:{}'.format(target.image_name),
                labels=[WorkUnitLabel.TASK]) as workunit:

            tag = '{}/{}'.format(self.get_options().namespace, target.image_name)

            try:
                client = docker.DockerClient(
                    base_url='unix://var/run/docker.sock')


                image, build_log = client.images.build(
                    path=tmpdir, rm=True, tag=tag, buildargs={"BASE_IMAGE": target.base_image})

                for log in build_log:
                    self.context.log.debug(str(log))

            except TypeError as err:
                print("At least one of path nor fileobj has to be specified")
            except docker.errors.BuildError as err:
                print("Error during the build")
                print(err)
            except docker.errors.APIError as err:
                print("APIError")
                print(err)
            except:
                raise

    def execute(self):
        """Execute task"""

        # Filter targets which are instances of DockerImage class
        targets = self.context.targets(self._is_target_type)

        if len(targets) == 0:
            self.context.log.debug("Exiting docker build as there are zero targets")

        # creates a versioned workdir (see pants TaskBase class)
        for target in targets:
            self._build_image(target)
