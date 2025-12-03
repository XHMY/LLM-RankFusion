import numpy as np
import pandas as pd
import csv


def Map(query_data):

    # Create an empty dictionary to hold the mapping between Item Code and Voter Name
    item_to_int_map = {}
    voter_to_int_map = {}

    # Get the unique Item Code and Voter Name values and create a map indexed to integers
    unique_item_codes = query_data['Item Code'].unique()
    unique_voter_names = query_data['Voter Name'].unique()

    # Establish a reverse mapping from integers to strings
    int_to_item_map = {i: code for i,
                       code in enumerate(unique_item_codes)}
    int_to_voter_map = {
        i: name for i, name in enumerate(unique_voter_names)}

    # Produces a string-to-integer mapping
    item_to_int_map = {v: k for k,
                       v in int_to_item_map.items()}
    voter_to_int_map = {v: k for k,
                        v in int_to_voter_map.items()}

    # Create a two-dimensional Numpy array of Voter Name*Item Code, starting with a value of 0
    num_voters = len(unique_voter_names)
    num_items = len(unique_item_codes)
    input_lists = np.full((num_voters, num_items), np.nan)

    # Filling an array
    for index, row in query_data.iterrows():
        voter_name = row['Voter Name']
        item_code = row['Item Code']
        item_rank = row['Item Rank']

        voter_index = voter_to_int_map[voter_name]
        item_index = item_to_int_map[item_code]

        input_lists[voter_index, item_index] = item_rank

    return int_to_item_map, int_to_voter_map, item_to_int_map, voter_to_int_map, input_lists


def RankToScore(input_file_path, output_file_path):

    df = pd.read_csv(input_file_path, header=None)
    df.columns = ['Query', 'Voter', 'Item', 'Rank', 'Algorithm']
    
    grouped = df.groupby(['Query', 'Voter'])['Rank'].transform('max')

    
    df['Score'] = grouped - df['Rank'] + 1
    result_df = df[['Query', 'Voter', 'Item', 'Score', 'Algorithm']]

    result_df.to_csv(output_file_path, index=False, header=None)


def ScoreToRank(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path, header=None)
    df.columns = ['Query', 'Voter', 'Item', 'Score', 'Algorithm']

    df['Rank'] = df.groupby(['Query', 'Voter'])['Score'].rank(
    method='dense', ascending=False)

    df.to_csv(output_file_path, index=False, columns=['Query', 'Voter', 'Item', 'Rank'], header=None)


