
import pandas as pd
import numpy as np
import os

from GAN.gan import GAN
from GAN import Metadata

metadata = Metadata()


def _dtypes64(table):
    for name, column in table.items():
        if column.dtype == np.int32:
            table[name] = column.astype('int64')
        elif column.dtype == np.float32:
            table[name] = column.astype('float64')

    return table

data_df = pd.read_csv("GAN/data/train.csv")
data_df_converted = {}

data_df_converted['findings'] = data_df

print(data_df_converted['findings'].head())

#meta = Metadata(metadata=os.path.join("GAN/data/titanic.csv", 'metadata.json'))


print(data_df.head())
metadata.add_table(
    name='findings',
    data=data_df,
    primary_key='finding_id'
)

print(metadata)
print(metadata.get_table_meta('findings'))

#metadata.visualize()

synthesizer = GAN()
synthesizer.fit(metadata, data_df_converted)

samples = synthesizer.sample(table_name='findings', num_rows=1000)

print(samples)