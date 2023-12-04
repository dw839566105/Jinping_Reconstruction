#!/usr/bin/env python3
'''
Reset index_step in waveform analysis(FSMP)
'''
import pyarrow.parquet as pq

parser.add_argument('-i', '--input', dest='ipt', metavar='filename[*.parquet]', type=str,
                    help='The filename [*Q.parquet] to read')

parser.add_argument('-o', '--output', dest='output', metavar='output[*.h5]', type=str,
                    help='The output filename [*.h5] to save')

f = pq.read_table(filename).to_pandas()
