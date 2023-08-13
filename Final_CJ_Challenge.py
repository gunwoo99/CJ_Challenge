import csv, math, copy
import numpy as np
from tqdm import tqdm
from Final_K_Means_Clustering import CJ_Cluster
from FInal_Basic_Information  import Information
from config import OD_MATRIX_FILE, ORDER_TABLE_FILE, VEHICLE_FILE, TERMINAL_FILE

def possible_combinations_from_orders(batch_orders_in_terminal):
    possible_combinations = [[]]
    for order in tqdm(batch_orders_in_terminal):
        next_combintaions = []
        
        for combination in possible_combinations:
            
            for k, group in enumerate(combination):
                group_cbm = np.sum(np.array(group)[:, 4])
                if group_cbm + order[4] < 55:
                    this_combination = copy.deepcopy(combination)
                    this_combination[k].append(order)
                    next_combintaions.append(this_combination)

            another_combination = copy.deepcopy(combination)
            another_combination.append([order])
            next_combintaions.append(another_combination)
        
        possible_combinations = copy.deepcopy(next_combintaions)
    return possible_combinations

information = Information(od_matrix_file   = OD_MATRIX_FILE   ,
                          order_table_file = ORDER_TABLE_FILE ,
                          vehicle_file     = VEHICLE_FILE     ,
                          terminal_file    = TERMINAL_FILE    ,)

sum = 0
order_num = 0
average_order_cbm = 0
for i, batch_orders in enumerate(information.total_orders):
    for j in range(information.total_terminal_num):
        # if i != 0 or j != 0:
        #     continue
        # for k in range(len(batch_orders[j])):
        #     average_order_cbm += float(batch_orders[j][k][4])
        #     order_num += 1
        #     sum += information.distance_matrix[information.total_index_of_vertex[batch_orders[j][k][8]]][information.total_index_of_vertex[batch_orders[j][k][3]]]
        order_number_in_terminal = len(batch_orders[j])
        boundary = 13
        if order_number_in_terminal <= boundary:
            continue
            # possible_combinations = possible_combinations_from_orders(batch_orders[j])
        else:
            terminal_site = [information.terminal_info[j][1], information.terminal_info[j][2]]
            clustering_model = CJ_Cluster(batch_orders[j], information.distance_matrix, information.total_index_of_vertex, terminal_site)
            print("+++++++++++++++++++++++++++++++++++++", i, j)
            print("batch_num", len(batch_orders[j]))
            print("site num", clustering_model.site_number)
            print("cluster num", clustering_model.cluster_number)
            print("++++++++++++++")
            for k in range(clustering_model.cluster_number):
                print(k+1, len(clustering_model.clustered_orders[k]), clustering_model.assigned_cluster_site_num[k])
                # for l in range(len(clustering_model.clustered_orders[k])):
                #     print(clustering_model.clustered_orders[k][l][0])
# print("all sum ", sum)
# print("average cbm", average_order_cbm/order_num)
# print("order num ", order_num)