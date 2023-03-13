
import pandas as pd
import numpy as np
import os

from GAN.tabular import CopulaGAN
from GAN.gan import GAN
from GAN import Metadata

metadata = Metadata()

data_df = pd.read_csv("GAN/data/train.csv")

data_df_converted = {}
data_df_converted['findings'] = data_df
print(data_df_converted['findings'].head())

model = CopulaGAN(primary_key='finding_id')
model.fit(data_df)

samples = model.sample(num_rows=200)

print(samples)