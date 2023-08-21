import numpy as np
import pandas as pd
import copy, datetime
import config
from Clustering         import CJ_Cluster
from Information        import Information
from Calculator         import Calculator

def possible_combinations_from_orders(order_set):
    possible_combinations = [[]]
    for order in order_set:
        next_combintaions = []
        
        for combination in possible_combinations:
            
            for k, group in enumerate(combination):
                group_cbm = np.sum(np.array(group)[:, 4])
                if group_cbm + order[4] <= 55:
                    this_combination = copy.deepcopy(combination)
                    this_combination[k].append(order)
                    next_combintaions.append(this_combination)

            another_combination = copy.deepcopy(combination)
            another_combination.append([order])
            next_combintaions.append(another_combination)
        
        possible_combinations = copy.deepcopy(next_combintaions)
    return possible_combinations

info = Information(od_matrix_file   = config.OD_MATRIX_FILE      ,
                   order_table_file = config.ORDER_TABLE_FILE    ,
                   vehicle_file     = config.VEHICLE_FILE        ,
                   terminal_file    = config.TERMINAL_FILE       ,
                   TTL              = config.TTL
                   )

real_time_terminal_vehicle   = copy.deepcopy(info.terminal_vehicle)
real_time_total_vehicle_info = copy.deepcopy(info.total_vehicle_info)

cost = 0
not_failed_num = 0
group_result   = []
after_result   = []
vehicle_result = [[info.total_vehicle_info[i][0]] + np.zeros(10).tolist() for i in range(len(info.total_vehicle_info))]

start_batch_order_num = []
for i in range(config.TOTAL_DAY*config.BATCH_COUNT_PER_DAY-1):
    batch_num = 0
    for j in range(info.terminal_num):
        batch_num += len(info.total_orders[i][j])
    start_batch_order_num.append(batch_num)
start_batch_order_num += [0, 0, 0, 0]

for ith_batch, batch_orders in enumerate(info.total_orders):
    if ith_batch % config.BATCH_COUNT_PER_DAY == 0:
        print(f"{config.YEAR:04}-{config.MONTH:02}-{int(ith_batch/config.BATCH_COUNT_PER_DAY) + 1:02}")
    batch_cost      = 0
    start_order_num = 0
    fail_order_num1 = 0
    fail_order_num2 = 0
    ith_batch_fail = []
    
    for vertex_idx in range(info.destination_num):
        vertex_best_terminal_value = 99999999
        for nearest_terminal in info.nearest_terminals_from_D[info.index_to_vertex[vertex_idx]]:
            terminal_value = (info.distance_matrix[vertex_idx][info.vertex_to_index[nearest_terminal]] ** 3) / (len(batch_orders[info.terminal_to_index[nearest_terminal]]) + 1)
            if vertex_best_terminal_value > terminal_value:
                vertex_best_terminal_value = terminal_value
                info.nearest_termnial_from_D[info.index_to_vertex[vertex_idx]] = nearest_terminal
    
    terminal_order_len = []
    for i in range(info.terminal_num):
        terminal_order_len.append([i, len(batch_orders[i])])
    terminal_order_len = sorted(terminal_order_len, key=lambda x : x[1])
    terminal_order_len = np.array(terminal_order_len)[:,0].tolist()
    
    for terminal_index in terminal_order_len:
        order_number_in_terminal = len(batch_orders[terminal_index])
        processed_orders = []
        
        start_order_num += len(batch_orders[terminal_index])
        failed_batch_orders_in_terminal = []
        # 배치에서 적절한 하차가능시간을 가지고 있는거만
        for i in range(len(batch_orders[terminal_index]) - 1, -1, -1):
            batch_time = datetime.timedelta(hours=(ith_batch % config.BATCH_COUNT_PER_DAY) * config.BATCH_TIME_HOUR)
            # 마지막 배치일때는 전부 emergency
            order_s = batch_orders[terminal_index][i][5]
            order_e = batch_orders[terminal_index][i][6]
            if ith_batch == config.TOTAL_DAY*config.BATCH_COUNT_PER_DAY-1:
                batch_orders[terminal_index][i][-1] = 0
                continue
            elif batch_time + datetime.timedelta(hours=3) <= order_e <= batch_time + datetime.timedelta(hours=config.BATCH_TIME_HOUR):
                continue
            elif batch_time <= order_s and order_s <= batch_time + datetime.timedelta(hours=4):
                continue
            elif batch_time + datetime.timedelta(days=1, hours=3) <= order_e <= batch_time + datetime.timedelta(days=1, hours=6):
                continue
            batch_orders[terminal_index][i][-1] -= 1
            failed_batch_orders_in_terminal.append(copy.deepcopy(batch_orders[terminal_index][i]))
            ith_batch_fail.append(copy.deepcopy(batch_orders[terminal_index][i]))
            del batch_orders[terminal_index][i]
        # print(len(batch_orders[terminal_index]), len(failed_batch_orders_in_terminal))
        
        if ith_batch < config.TOTAL_DAY*config.BATCH_COUNT_PER_DAY-1:
            info.total_orders[ith_batch + 1][terminal_index] += failed_batch_orders_in_terminal
        fail_order_num1 += len(failed_batch_orders_in_terminal)
        
        if 21 < ith_batch < config.TOTAL_DAY*config.BATCH_COUNT_PER_DAY-1:
            for i in range(len(info.total_orders[ith_batch][terminal_index])):
                info.total_orders[ith_batch][terminal_index][i][-1] = 0
        
        processed_orders = []
        batch_ordersasdfnum = 0
        if len(batch_orders[terminal_index]) > config.CLUSTER_COUNT_BOUNDARY:
            terminal_site    = [info.terminal_info[terminal_index][1], info.terminal_info[terminal_index][2]]
            clustering_model = CJ_Cluster(batch_orders[terminal_index], info.distance_matrix, info.vertex_to_index, terminal_site)
            clusters         = clustering_model.clustered_orders
            for cluster in clusters:
                batch_ordersasdfnum += len(cluster)
                if len(cluster) > config.CLUSTER_COUNT_BOUNDARY:
                    sorted_cluster = sorted(cluster, key=lambda x:x[4])
                    groups = []
                    p = 0
                    while p < len(sorted_cluster)//2:
                        group = []
                        for i in range(p, p+4):
                            if i >= len(sorted_cluster)//2: break
                            group.append(sorted_cluster[i])
                            group.append(sorted_cluster[-(i+1)])
                        groups.append(group)
                        p += 4
                    if len(sorted_cluster)%2==1:
                        groups[-1].append(sorted_cluster[len(sorted_cluster)//2+1])
                    for group in groups:
                        processed_orders.append(group)
                else:
                    processed_orders.append(cluster)
        else:
            processed_orders.append(batch_orders[terminal_index])
        
        if len(processed_orders) > 1:
            sorted_orders = sorted(processed_orders, key=lambda x:np.mean(np.array(x)[:, -1]))
        terminal_cost = 0
        
        for cluster in processed_orders:
            possible_combinations = possible_combinations_from_orders(cluster)
            calculator = Calculator(  
                possible_combinations = possible_combinations, 
                information           = info,
                batch_datetime        = datetime.datetime(year=config.YEAR, month=config.MONTH, day=config.START_DAY) + datetime.timedelta(hours=ith_batch * config.BATCH_TIME_HOUR),
                terminal_index        = terminal_index,
                terminal_vehicle      = real_time_terminal_vehicle,
                vehicle_info          = real_time_total_vehicle_info,
                vehicle_result        = vehicle_result,
                group_result          = group_result,
                after_result          = after_result,
                ith_batch             = ith_batch,
            )
            
            for j in range(len(calculator.fail_group)):
                calculator.fail_group[j][-1] -= 1
            ith_batch_fail += copy.deepcopy(calculator.fail_group)
            if ith_batch < config.TOTAL_DAY*config.BATCH_COUNT_PER_DAY-1:
                info.total_orders[ith_batch + 1][terminal_index] += copy.deepcopy(calculator.fail_group)
                fail_order_num2 += len(calculator.fail_group)
            
            for best_group in calculator.best_combination:
                if best_group[0] != float("inf"):
                    terminal_cost += best_group[0]
            
            real_time_terminal_vehicle   = copy.deepcopy(calculator.terminal_vehicle)
            real_time_total_vehicle_info = copy.deepcopy(calculator.vehicle_info    )
            vehicle_result               = copy.deepcopy(calculator.vehicle_result  )
            group_result                 = copy.deepcopy(calculator.group_result    )
            after_result                 = copy.deepcopy(calculator.after_result    )
        batch_cost += terminal_cost
    # 배차된 그룹 정렬(차량순서, 배송순서) => ith_batch_result       
    vehicle_id_dict = {}
    for i in range(len(group_result)):
        if group_result[i][1] not in vehicle_id_dict:
            vehicle_id_dict[group_result[i][1]] = []
        group_result[i][2] = len(vehicle_id_dict[group_result[i][1]]) + 1
        after_result[i][2] = len(vehicle_id_dict[group_result[i][1]]) + 1
        vehicle_id_dict[group_result[i][1]].append(copy.deepcopy(group_result[i]))
    ith_batch_result =[]
    for key in sorted(vehicle_id_dict.keys(), key=lambda x:int(x[4:])):
        ith_batch_result += vehicle_id_dict[key]
    # 실패한 그룹 ith_batch_result에 추가
    for i in range(len(ith_batch_fail)):
        ith_batch_result.append([ith_batch_fail[i][0], "Null", "Null", "Null", "Null", "Null", "Null", "Null", "No"])
    
    final_orders_table = pd.DataFrame(ith_batch_result, columns=["ORD_NO", "VehicleID", "Sequence", "SiteCode", "ArrivalTime",
                                                                "WaitingTime", "ServiceTime", "DepartureTime", "Delivered"])
    final_orders_table.to_excel(excel_writer=f'final_orders_table_{ith_batch}.xlsx')
    cost += batch_cost
    not_failed_num += start_order_num - fail_order_num1 - fail_order_num2
    print("Group", ith_batch%config.BATCH_COUNT_PER_DAY, f"{(ith_batch%config.BATCH_COUNT_PER_DAY)*config.BATCH_TIME_HOUR:<2}:00", 
          "batch_cost", batch_cost, 
          "total_cost", cost, 
          "new_order", start_batch_order_num[ith_batch], 
          "total_this_order", start_order_num, fail_order_num1, 
          fail_order_num2, start_order_num - fail_order_num1 - fail_order_num2, 
          not_failed_num,
          len(ith_batch_result))

total_fixed_cost = 0
for vehicle in vehicle_result:
    total_fixed_cost += vehicle[9]
final_cost = cost + total_fixed_cost
print("====== FINAL_COST =======", final_cost)

final_veh_table = pd.DataFrame(vehicle_result, columns=["VehicleID", "Count", "Volume", "TravelDistance", "WorkTime", "TravleTime",
                                                        "ServiceTime", "WaitingTime", "TotalCost", "FixedCost", "VariableCost"])
final_veh_table.to_excel(excel_writer='final_veh_table.xlsx')

final_orders_table = pd.DataFrame(after_result, columns=["ORD_NO", "VehicleID", "Sequence", "SiteCode", "ArrivalTime",
                                                         "WaitingTime", "ServiceTime", "DepartureTime", "Delivered"])
final_orders_table.to_excel(excel_writer=f'after_result.xlsx')
