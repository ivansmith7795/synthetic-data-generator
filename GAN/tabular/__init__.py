"""Models for tabular data."""

from GAN.tabular.copulagan import CopulaGAN
from GAN.tabular.copulas import GaussianCopula
from GAN.tabular.ctgan import CTGAN, TVAE

__all__ = (
    'GaussianCopula',
    'CTGAN',
    'TVAE',
    'CopulaGAN',
)
