import numpy as np
import pandas as pd
import csv
import random
import time

def PartialToFull(input_list):
    # 扩充为full list的方式是将未排序的项目全部并列放在最后一名
    num_voters = input_list.shape[0]
    list_numofitems = np.zeros(num_voters)

    for k in range(num_voters):
        max_rank = np.nanmax(input_list[k])
        list_numofitems[k] = max_rank
        input_list[k] = np.nan_to_num(input_list[k], nan = max_rank + 1)

    return input_list, list_numofitems

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

def Outranking_matrix(input_list):

    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    
    # 当items数量大于等于3时，计算出items之间的偏好，构建outranking矩阵
    outranking_matrix = np.zeros((num_items, num_items))
    if (num_items >= 3):
        for v in range(num_voters):
            for i in range(num_items):
                for j in range(num_items):
                    if(i == j):
                        outranking_matrix[i,j] = 0
                    else:
                        if(input_list[v,i] < input_list[v,j]):
                            outranking_matrix[i,j] += 1
                        elif(input_list[v,i] == input_list[v,j]):
                            outranking_matrix[i,j] += 0.5
                        else:
                            outranking_matrix[i,j] +=0

        return outranking_matrix

def calculate_max_row_score_index(matrix):

    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    
    max_score_index = np.argmax(row_sums)

    return max_score_index

def calculate_max_row_score(matrix):

    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    max_score_index = np.argmax(row_sums)
    max_score = row_sums[max_score_index]

    return max_score

# # 为了裁剪超越矩阵的计算设计i行i列归0函数
# def set_row_col_to_zero(matrix,i):
#     # 将第 i 行设为 0
#     matrix[i, :] = 0
#     # 将第 i 列设为 0
#     matrix[:, i] = 0

def calculate_rank(input_list):

    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    ranked_list = []
    total_score = []
    outranking_matrix = Outranking_matrix(input_list)
    outranking_matrix_ = outranking_matrix

    original_indices = list(range(outranking_matrix.shape[0]))
    # print(original_indices)
    max_score_index = calculate_max_row_score_index(outranking_matrix)
    # max_score = calculate_max_row_score(outranking_matrix)
    # num_equal_score = max_score_index.size

    while outranking_matrix.shape[0] > 0 and outranking_matrix.shape[1] > 0:
        max_score_index_ = calculate_max_row_score_index(outranking_matrix_)
        num_equal_score = max_score_index_.size
        if(num_equal_score > 1):
            for v in range(num_voters):
                for i in range(num_equal_score):
                    new_input_list = []
                    new_input_list.append(input_list[v,max_score_index_[i]])

            new_matrix = Outranking_matrix(new_input_list)
            max_score_index = calculate_max_row_score_index(new_matrix)

            num_equal_score = max_score_index.size

            if(num_equal_score > 1):
                ranked_list.append(max_score_index[0])
            
                outranking_matrix_[max_score_index[0], :] = 0
                outranking_matrix_[:, max_score_index[0]] = 0

                max_score_index = calculate_max_row_score_index(outranking_matrix)
                outranking_matrix = np.delete(
                    outranking_matrix, max_score_index, axis=0)
                outranking_matrix = np.delete(
                    outranking_matrix, max_score_index, axis=1)
            else:
                ranked_list.append(max_score_index)
                outranking_matrix_[max_score_index, :] = 0
                outranking_matrix_[:, max_score_index] = 0

                max_score_index = calculate_max_row_score_index(outranking_matrix)
                outranking_matrix = np.delete(
                    outranking_matrix, max_score_index, axis=0)
                outranking_matrix = np.delete(
                    outranking_matrix, max_score_index, axis=1)

        else:
            if(outranking_matrix.shape[0] == 2 and outranking_matrix.shape[1] == 2):
                row_sums = np.sum(outranking_matrix_, axis=1)
                indices = np.where(row_sums > 0)[0]
                if(indices.size >= 2):
                    if(row_sums[indices[0]] > row_sums[indices[1]]):
                        ranked_list.append(indices[0])
                        ranked_list.append(indices[1])
                    elif(row_sums[indices[0]] < row_sums[indices[1]]):
                        ranked_list.append(indices[1])
                        ranked_list.append(indices[0])
                    else:
                        ranked_list.append(indices[0])
                        ranked_list.append(indices[1])
                    print("1:", indices)
                elif(indices.size == 1):
                    ranked_list.append(indices[0])
                    for i in range(num_items):
                        if (i in ranked_list):
                            pass
                        else:
                            ranked_list.append(i)
                    print("2:", indices)
                    print("2:", indices[0])
                else:
                    for i in range(num_items):
                        if (i in ranked_list):
                            pass
                        else:
                            ranked_list.append(i)
                    print("3:", indices)
                break

            else:
                print(max_score_index_)
                ranked_list.append(max_score_index_)
                
                max_score_index = calculate_max_row_score_index(outranking_matrix)

                outranking_matrix_[max_score_index_, :] = 0
                outranking_matrix_[:, max_score_index_] = 0
                outranking_matrix = np.delete(
                    outranking_matrix, max_score_index, axis=0)
                outranking_matrix = np.delete(
                    outranking_matrix, max_score_index, axis=1)
            

    return ranked_list

def Mork_HeuristicAgg(input_list):

    ranked_list = calculate_rank(input_list)
    return ranked_list

def Mork_Heuristic(input, output, is_partial_list=True):
    df = pd.read_csv(input, header=None)
    df.columns = ['Query', 'Voter Name', 'Item Code', 'Item Rank']

    unique_queries = df['Query'].unique()
    start_time = time.perf_counter()
    result = []

    for query in unique_queries:
        query_data = df[df['Query'] == query]
        int_to_item_map, int_to_voter_map, item_to_int_map, voter_to_int_map, input_lists = Map(
            query_data)

        if (is_partial_list == True):
            full_input_lists, list_numofitems = PartialToFull(input_lists)
        
        item_ranked = Mork_HeuristicAgg(full_input_lists)
        
        for i in range(len(item_ranked)):
            item_code = int_to_item_map[item_ranked[i]]
            item_rank = i+1
            new_row = [query, item_code, item_rank]
            result.append(new_row)
            
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"程序运行时间：{elapsed_time}秒")
    
    with open(output, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in result:
            writer.writerow(row)