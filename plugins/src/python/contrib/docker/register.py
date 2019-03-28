"""Register the plugin"""
from pants.goal.task_registrar import TaskRegistrar as task
from pants.build_graph.build_file_aliases import BuildFileAliases

from contrib.docker.targets.python import DockerPythonTarget
from contrib.docker.tasks.build_python import PythonBuildTask

def build_file_aliases():
    """Define name of section in build file"""
    return BuildFileAliases(targets={
        'python_docker': DockerPythonTarget
    })

def register_goals():
    """Regist goal for CLI"""
    task(name='docker-python', action=PythonBuildTask).install('bundle')
