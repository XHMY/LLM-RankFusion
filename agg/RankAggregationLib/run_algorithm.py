from WT_INDEG import PreferenceRelationsGraph
from Condorcet import Condorcet
from Agglomerative import Agglomerative
from Outranking import Outranking

# input_file = r'D:\Code of RA\RA\dataset\ice-cream.csv'
# output_file = r'D:\Code of RA\RA\results\ice-cream_WT_INDEG_0.5_0.5.csv'
# print('Running WT-INDEG...')
# PreferenceRelationsGraph(input_file, output_file, 0.5, 0.5)

# input_file = r'D:\Code of RA\RA\dataset\ice-cream.csv'
# output_file = r'D:\Code of RA\RA\results\ice-cream_Condorcet.csv'
# print('Running Condorcet...')
# Condorcet(input_file, output_file)


# input_file = r'D:\Code of RA\RA\dataset\ice-cream.csv'
# output_file = r'D:\Code of RA\RA\results\ice-cream_Agglomerative_0.5_0.2.csv'
# print('Running Agglomerative...')
# Agglomerative(input_file, output_file, 0.5, 0.2)

input_file = r'agg/RankAggregation-Lib/datasets/ice-cream/ice-cream.csv'
output_file = r'agg/RankAggregation-Lib/datasets/ice-cream/ice-cream_Outranking_0.15_0.3_0.4_0.2.csv'
print('Running Outranking...')
Outranking(input_file, output_file, 0.15, 0.3, 0.4, 0.2)