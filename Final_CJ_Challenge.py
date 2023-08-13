import csv, math, copy
import numpy as np
from tqdm import tqdm
from Final_K_Means_Clustering import CJ_Cluster
from FInal_Basic_Information  import Information
from config import OD_MATRIX_FILE, ORDER_TABLE_FILE, VEHICLE_FILE, TERMINAL_FILE
from collections import deque

def group_by_cost(orders: list, quantity_by_veh_cbm: dict) -> dict:
    # weights = sorted(map(lambda x:float(x[4]), orders), reverse=True)
    weights = sorted(map(lambda x:float(x), orders), reverse=True)
    sorted_veh_cbm = sorted((key for key, _ in filter(lambda item: item[1] != 0, quantity_by_veh_cbm.items())))
    groups = dict()
    dq = deque(weights)

    if weights[0] > sorted_veh_cbm[-1]:
        raise ValueError(f"{weights[0]}는 남은 차량의 최대 적재량 {sorted_veh_cbm[-1]}를 초과했습니다.")

    while dq:
        weight = dq.popleft()
        group = [weight]
        current_weight = weight
        
        for cbm in sorted_veh_cbm:
            if quantity_by_veh_cbm[cbm] > 0 and weight <= cbm:
                max_weight = cbm
                quantity_by_veh_cbm[cbm] -= 1
                break
        else:
            raise ValueError("더이상 배치 가능한 차량이 없습니다.")

        for weight in range(len(dq)):
            weight = dq.popleft()
            if current_weight + weight <= max_weight:
                group.append(weight)
                current_weight += weight
            else:
                dq.append(weight)

        if max_weight not in groups:
            groups[max_weight] = []
        groups[max_weight].append(group)
    
    # HACK : 예제로 받은 파일에서 27*2 ~= 55 이기에 가능(배수 관계), 그래서 일단.. 그냥 상수 넣었음.. 양심상.. 변수로는 뺐음
    MIN_CBM = 27
    MAX_CBM = 55
    if MIN_CBM*2 <= MAX_CBM and MIN_CBM in groups and len(groups[MIN_CBM]) > 2:
        if MAX_CBM not in groups:
            groups[MAX_CBM] = []
        while len(groups[MIN_CBM]) >= 2:
            groups[MAX_CBM].append(groups[MIN_CBM].pop()+groups[MIN_CBM].pop())
    return groups

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
            print("site num", len(clustering_model.orders_site))
            print("cluster num", clustering_model.cluster_number)
            print("++++++++++++++")
            for k in range(clustering_model.cluster_number):
                print(k+1, len(clustering_model.clustered_orders[k]), clustering_model.assigned_cluster_site_num[k])
                # for l in range(len(clustering_model.clustered_orders[k])):
                #     print(clustering_model.clustered_orders[k][l][0])
# print("all sum ", sum)
# print("average cbm", average_order_cbm/order_num)
# print("order num ", order_num)