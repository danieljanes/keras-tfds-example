# coding=utf-8
# Copyright 2015 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from ntpath import basename
from twitter.common.collections import maybe_list

from pants.base.build_environment import get_scm
from pants.base.payload import Payload
from pants.base.payload_field import PrimitiveField
from pants.build_graph.target import Target
from pants.base.exceptions import TargetDefinitionException


class DockerTargetBase(Target):
    """DockerTargetBase"""

    def __init__(self,
                 address=None,
                 payload=None,
                 binary=None,
                 image_name=None,
                 image_tags=None,
                 base_image=None,
                 dockerfile=None,
                 **kwargs):
        """
        :param address: The Address that maps to this Target in the BuildGraph.
        :type address: :class:`pants.build_graph.address.Address`
        :param payload: The configuration encapsulated by this target.  Also in charge of most
                        fingerprinting details.
        :type payload: :class:`pants.base.payload.Payload`
        :param string binary: Target spec of the ``jvm_binary`` or the ``python_binary``
      that contains the app main.
        :param image_name: name of docker image
        :type image_name: str
        :param image_tag: tags of the docker image
        :type image_tag: str[]
        : param dockerfile: custom docker file
        : type dockerfile: str
        """
        tags = image_tags or []
        tags.append('c' + get_scm().commit_id[:8])

        payload = payload or Payload()
        payload.add_fields({
            'binary':
            PrimitiveField(binary),
            'image_name':
            PrimitiveField(image_name or basename(address.spec_path)),
            'image_tags':
            PrimitiveField(maybe_list(tags)),
            'base_image':
            PrimitiveField(base_image or "ubuntu:18.04"),
            'dockerfile':
            PrimitiveField(dockerfile)
        })

        super(DockerTargetBase, self).__init__(
            address=address, payload=payload, **kwargs)

    @classmethod
    def binary_target_type(cls):
        raise NotImplementedError(
            'Must implement in subclass (e.g.: `return PythonBinary`)')

    @classmethod
    def default_dockerfile(cls):
        raise NotImplementedError(
            'Must implement in subclass (e.g.: `return PythonBinary`)')

    @classmethod
    def compute_dependency_specs(cls, payload=None):
        binary = payload.as_dict().get('binary')
        if binary:
            yield binary

    @property
    def image_name(self):  # pylint: disable=missing-docstring
        return self.payload.image_name

    @property
    def image_tags(self):  # pylint: disable=missing-docstring
        return self.payload.image_tags

    @property
    def base_image(self):  # pylint: disable=missing-docstring
        return self.payload.base_image

    @property
    def binary(self):
        """Returns the binary this target references."""
        dependencies = self.dependencies

        if len(dependencies) != 1:
            raise TargetDefinitionException(
                self,
                'An app must define exactly one binary dependency, have: {}'.
                format(dependencies))

        binary = dependencies[0]

        if not isinstance(binary, self.binary_target_type()):
            raise TargetDefinitionException(
                self, 'Expected binary dependency to be a {} target, found {}'.
                format(self.binary_target_type(), binary))

        return binary

    @property
    def dockerfile(self):
        """Returns path to Dockerfile or default file path"""
        dockerfile = self.payload.dockerfile or self.default_dockerfile()
        return dockerfile
