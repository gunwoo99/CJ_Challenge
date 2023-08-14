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
                   terminal_file    = "File_Terminal.csv" ,
                   TTL              = 11
                   )

real_time_terminal_vehicle   = copy.deepcopy(info.terminal_vehicle)
real_time_total_vehicle_info = copy.deepcopy(info.total_vehicle_info)

cost = 0

group_result = []
after_result = []

for ith_batch, batch_orders in enumerate(info.total_orders):
    batch_cost = 0
    start_order_num = 0
    fail_order_num1 = 0
    fail_order_num2 = 0
    
    terminal_order_len = []
    for i in range(info.terminal_num):
        terminal_order_len.append([i, len(batch_orders[i])])
    terminal_order_len = sorted(terminal_order_len, key=lambda x : x[1])
    terminal_order_len = np.array(terminal_order_len)[:,0].tolist()
    
    for terminal_index in terminal_order_len:
        order_number_in_terminal = len(batch_orders[terminal_index])
        processed_orders = []
        boundary = 9
        
        start_order_num += len(batch_orders[terminal_index])
        failed_batch_orders_in_terminal = []
        # 배치에서 적절한 하차가능시간을 가지고 있는거만
        for i in range(len(batch_orders[terminal_index]) - 1, -1, -1):
            batch_time = datetime.timedelta(hours=(ith_batch % 4) * 6)
            if ith_batch == 27:
                batch_orders[terminal_index][i][-1] = 0
            if batch_time > batch_orders[terminal_index][i][5] or batch_orders[terminal_index][i][5] > batch_time + datetime.timedelta(hours=7):
                if batch_orders[terminal_index][i][5] <= batch_time - datetime.timedelta(hours=17):
                    continue
                if batch_orders[terminal_index][i][-1] - 1 <= 0:
                    continue
                batch_orders[terminal_index][i][-1] -= 1
                failed_batch_orders_in_terminal.append(copy.deepcopy(batch_orders[terminal_index][i]))
                del batch_orders[terminal_index][i]
        info.total_orders[ith_batch + 1][terminal_index] = failed_batch_orders_in_terminal + info.total_orders[ith_batch + 1][terminal_index]
        if ith_batch > 23:
            for i in range(info.total_orders[terminal_index]):
                info.total_orders[ith_batch][terminal_index][i][-1] = 0
        fail_order_num1 += len(failed_batch_orders_in_terminal)
        
        processed_orders = []
        if order_number_in_terminal > boundary:
            terminal_site = [info.terminal_info[terminal_index][1], info.terminal_info[terminal_index][2]]
            clustering_model = CJ_Cluster(batch_orders[terminal_index], info.distance_matrix, info.vertex_to_index, terminal_site)
            clusters = clustering_model.clustered_orders
            for cluster in clusters:
                    if len(cluster) > 9:
                        for itera in range(int((len(cluster) - 1) / 9)):
                            processed_orders = [cluster[itera*9:(itera+1)*9]] + processed_orders
                        processed_orders = [cluster[int((len(cluster) - 1) / 9) * 9:]] + processed_orders
                    else:
                        processed_orders.append(cluster)
        else:
            processed_orders.append(batch_orders[terminal_index])
        
        if len(processed_orders) > 1:
            processed_orders = sorted(processed_orders, key=lambda x:np.mean(np.array(x)[:, -1]))
            
        
        terminal_cost = 0
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
            
            for i in range(len(calculator.fail_group)):
                calculator.fail_group[i][-1] -= 1
            info.total_orders[ith_batch + 1][terminal_index] = calculator.fail_group + info.total_orders[ith_batch + 1][terminal_index]
            fail_order_num2 += len(calculator.fail_group)
            for best_group in calculator.best_combination:
                if best_group[0] != float("inf"):
                    terminal_cost += best_group[0]
            real_time_terminal_vehicle   = copy.deepcopy(calculator.terminal_vehicle)
            real_time_total_vehicle_info = copy.deepcopy(calculator.vehicle_info)
            
            
        batch_cost += terminal_cost
    if ith_batch % 4 == 0:
        print(f"2023-05-{int(ith_batch/4) + 1}")
    
    cost += batch_cost
    print("Group", ith_batch%4, f"{(ith_batch%4)*6}:00", batch_cost, cost, start_order_num, fail_order_num1, fail_order_num2, start_order_num - fail_order_num1 - fail_order_num2)
