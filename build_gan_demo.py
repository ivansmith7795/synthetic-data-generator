from GAN import load_demo


import pandas as pd
import numpy as np
import os

from GAN.gan import GAN
from GAN import Metadata

metadata, tables = load_demo(metadata=True)

print(tables['users'].head())

synthesizer = GAN()
synthesizer.fit(metadata, tables)
