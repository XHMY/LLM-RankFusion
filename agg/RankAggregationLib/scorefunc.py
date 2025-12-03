import numpy as np
import pandas as pd
import csv
import math

def LinearAgg(input_list):
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    item_score = np.zeros((num_voters,num_items))

    for k in range(num_voters):
        for i in range(num_items):
            item_score[k,i] = num_items - input_list[k,i] + 1

    return item_score

def reciprocalAgg(input_list):
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    item_score = np.zeros((num_voters,num_items))

    for k in range(num_voters):
        for i in range(num_items):
            item_score[k,i] = 1 / input_list[k,i]

    return item_score

def powerAgg(input_list):
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    item_power_score = np.zeros(num_items)
    item_score = np.zeros((num_voters,num_items))
    
    for k in range(num_voters):
        for i in range(num_items):
            item_score[k,i] = math.pow(1.1,num_items-input_list[k,i])
    
    return item_score

def logAgg(input_list):
    num_voters = input_list.shape[0]
    num_items = input_list.shape[1]
    item_power_score = np.zeros(num_items)
    item_score = np.zeros((num_voters,num_items))
    
    for k in range(num_voters):
        for i in range(num_items):
            item_score[k,i] = math.log(input_list[k,i],0.1)
    return item_score
