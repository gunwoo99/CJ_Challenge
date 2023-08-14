import numpy as np
import copy, datetime
from Final_Clustering         import CJ_Cluster
from FInal_Basic_Information  import Information
from Final_Calculator         import Calculator

def possible_combinations_from_orders(order_set):
    possible_combinations = [[]]
    for order in order_set:
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

info = Information(od_matrix_file   = "File_OD_Matrix.csv",
                   order_table_file = "File_Order.csv"    ,
                   vehicle_file     = "File_Vehicle.csv"  ,
                   terminal_file    = "File_Terminal.csv" ,)

real_time_terminal_vehicle   = copy.deepcopy(info.terminal_vehicle)
real_time_total_vehicle_info = copy.deepcopy(info.total_vehicle_info)

for ith_batch, batch_orders in enumerate(info.total_orders):
    for terminal_index in range(info.terminal_num):
        
        order_number_in_terminal = len(batch_orders[terminal_index])
        processed_orders = []
        boundary = 9
        
        failed_batch_orders_in_terminal = []
        # 배치에서 적절한 하차가능시간을 가지고 있는거만
        for i in range(len(batch_orders[terminal_index]) - 1, -1, -1):
            batch_time = datetime.timedelta(hours=(ith_batch % 4) * 6 + 1)
            if batch_time > batch_orders[terminal_index][i][5] or batch_orders[terminal_index][i][5] > batch_time + datetime.timedelta(hours=6):
                failed_batch_orders_in_terminal.append(copy.deepcopy(batch_orders[terminal_index][i]))
                del batch_orders[terminal_index][i]
        
        info.total_orders[ith_batch + 1][terminal_index].append(failed_batch_orders_in_terminal)
        
        if order_number_in_terminal > boundary:
            terminal_site = [info.terminal_info[terminal_index][1], info.terminal_info[terminal_index][2]]
            clustering_model = CJ_Cluster(batch_orders[terminal_index], info.distance_matrix, info.vertex_to_index, terminal_site)
            processed_orders = clustering_model.clustered_orders
        else:
            processed_orders.append(batch_orders[terminal_index])
        
        for cluster in processed_orders:
            possible_combinations = possible_combinations_from_orders(cluster)
            calculator = Calculator(
                possible_combinations = possible_combinations, 
                information           = info,
                batch_datetime        = datetime.datetime(year=2023, month=5, day=1) + datetime.timedelta(hours=ith_batch * 6),
                terminal_index        = terminal_index,
                terminal_vehicle      = real_time_terminal_vehicle,
                vehicle_info          = real_time_total_vehicle_info
            )
            
