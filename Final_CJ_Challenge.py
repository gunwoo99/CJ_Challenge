import csv, math, copy
import numpy as np
from tqdm import tqdm
from Final_K_Means_Clustering import KMeans
from FInal_Basic_Information  import Information

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

information = Information(od_matrix_file   = "File_OD_Matrix.csv",
                          order_table_file = "File_Order.csv"    ,
                          vehicle_file     = "File_Vehicle.csv"  ,
                          terminal_file    = "File_Terminal.csv" ,)

sum = 0
order_num = 0
average_order_cbm = 0
for i, batch_orders in enumerate(information.total_orders):
    for j in range(information.total_terminal_num):
        for k in range(len(batch_orders[j])):
            average_order_cbm += float(batch_orders[j][k][4])
            order_num += 1
            sum += information.distance_matrix[information.total_index_of_vertex[batch_orders[j][k][8]]][information.total_index_of_vertex[batch_orders[j][k][3]]]
        order_number_in_terminal = len(batch_orders[j])
        boundary = 13
        if order_number_in_terminal <= boundary:
            continue
            # possible_combinations = possible_combinations_from_orders(batch_orders[j])
        else:
            kmeans = KMeans(batch_orders[j])
            print("+++++++++++++++++++++++++++++++++++++", i, j)
            print("batch_num", len(batch_orders[j]))
            print("site num", kmeans.site_number)
            print("++++++++++++++")
            for k in range(kmeans.cluster_number):
                print(k+1, len(kmeans.clustered_orders[k]), kmeans.assigend_cluster_site_number[k])
                # for l in range(len(kmeans.clustered_orders[k])):
                #     print(kmeans.clustered_orders[k][l][0])
print("all sum ", sum)
print("average cbm", average_order_cbm/order_num)
print("order num ", order_num)