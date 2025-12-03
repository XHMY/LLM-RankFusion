# Reference: Rank Aggregation using Score Rule
# fsw, Tancilon：20240725
# Define the input to the algorithm as a csv file format: Query | Voter name | Item Code | Item Rank
#      - Query does not require consecutive integers starting from 1.
#      - Voter name and Item Code are allowed to be in String format.
# Define the final output of the algorithm as a csv file format： Query | Item Code | Item Rank
#      - Output is the rank information, not the score information
# The smaller the Item Rank, the higher the rank.
import numpy as np
import pandas as pd
import csv
import time
from functools import cmp_to_key

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

def borda(input_list):
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    item_borda_score = np.zeros(num_items)
    item_score = np.zeros((num_voters,num_items))

    for k in range(num_voters):
        for i in range(num_items):
            item_score[k,i] = num_items - input_list[k,i] + 1
            item_borda_score[i] += item_score[k,i]

    return item_borda_score

def eliminate_top(vot, m, rule, tiebreaking):
	tie = 0
	tiebreaking = list(reversed(tiebreaking))
	votes = []
	for v in vot:
		vvv = []
		for c in v:
			vvv.append(c)
		votes.append(vvv)
	not_deleted = list(range(m))
	order = [0] * m
	points = rule(vot, m)

	for i in range(m - 1):
		max_relevant = max([points[i] for i in not_deleted])
		cand_to_be_del = [i for i in not_deleted if points[i] == max_relevant]
		if len(cand_to_be_del) > 1:
			tie = tie + 1
		for t in tiebreaking:
			if t in cand_to_be_del:
				delete = t
				break
		order[i] = delete
		not_deleted.remove(delete)
		for i in range(len(votes)):
			if delete in votes[i]:
				votes[i].remove(delete)
		points = rule(votes, m)
	order[m - 1] = not_deleted[0]
	return order, tie

def eliminate_bottom(vot, m, rule, tiebreaking):
	tie = 0
	votes = []
	for v in vot:
		vvv = []
		for c in v:
			vvv.append(c)
		votes.append(vvv)

	not_deleted = list(range(m))
	order = [0] * m
	points = rule(vot, m)
	print(points)
	for i in range(m - 1):
		min_relevant = min([points[i] for i in not_deleted])
		cand_to_be_del = [i for i in not_deleted if points[i] == min_relevant]
		if len(cand_to_be_del) > 1:
			tie = tie + 1
		for t in tiebreaking:
			if t in cand_to_be_del:
				delete = t
				break
		order[m - i - 1] = delete
		not_deleted.remove(delete)
		for i in range(len(votes)):
			if delete in votes[i]:
				votes[i].remove(delete)
		points = rule(votes, m)
	order[0] = not_deleted[0]
	return order, tie

tie_breaking_order = None
tie = None

def compare(item1, item2):
	if item1[0] > item2[0]:
		return 1
	elif item1[0] < item2[0]:
		return -1
	elif tie_breaking_order.index(item1[1]) < tie_breaking_order.index(item2[1]):
		global tie
		tie = tie + 1
		return 1
	else:
		return -1

def score_ordering(m, points,tiebreaking):
	global tie
	tie = 0
	global tie_breaking_order
	print(points)
	tie_breaking_order = tiebreaking
	inversed_points = [-x for x in points]
	to_be_sorted = list(zip(inversed_points, list(range(m))))
	return [x for _, x in sorted(to_be_sorted, key=cmp_to_key(compare))], tie

def Borda_Score(input, output, is_partial_list=True):
    df = pd.read_csv(input,header=None)
    df.columns = ['Query','Voter Name', 'Item Code', 'Item Rank']
	
    # 获取唯一的Query值
    unique_queries = df['Query'].unique()
    start_time = time.perf_counter()
    # 创建一个空的DataFrame来存储结果
    result = []

    for query in unique_queries:
        query_data = df[df['Query'] == query]
        int_to_item_map, int_to_voter_map, item_to_int_map, voter_to_int_map, input_lists = Map(
            query_data)

        if (is_partial_list == True):
            full_input_lists, list_numofitems = PartialToFull(input_lists)

        # 调用函数，获取排名信息
        rank, tie = score_ordering(full_input_lists.shape[1], borda(full_input_lists), list(np.random.permutation(full_input_lists.shape[1])))
        # 将结果添加到result_df中
        for i in range(len(rank)):
            item_code = int_to_item_map[rank[i]]
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