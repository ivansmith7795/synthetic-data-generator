"""Metadata module."""

from GAN.metadata import visualization
from GAN.metadata.dataset import Metadata
from GAN.metadata.errors import MetadataError, MetadataNotFittedError
from GAN.metadata.table import Table

__all__ = (
    'Metadata',
    'MetadataError',
    'MetadataNotFittedError',
    'Table',
    'visualization'
)