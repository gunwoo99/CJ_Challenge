import numpy as np
import itertools, datetime, copy, math
from Information import Information
import config

class Calculator():
    def __init__(self, possible_combinations, information, batch_datetime, terminal_index, terminal_vehicle, vehicle_info, vehicle_result, group_result, after_result, ith_batch):
        self.info           : Information
        self.info           = information
        self.batch_datetime = batch_datetime
        self.vehicle_result = vehicle_result
        self.group_result   = group_result
        self.after_result   = after_result
        self.ith_batch      = ith_batch
        self.vehicle_info   = vehicle_info
        self._type1(
                possible_combinations = possible_combinations,
                terminal_index        = terminal_index,
                terminal_vehicle      = terminal_vehicle,
                vehicle_info          = vehicle_info,
            )
    
    def _type1(self, possible_combinations, terminal_index, terminal_vehicle, vehicle_info):
        best_combination = []
        best_combination_cost = float("inf")
        for combination in possible_combinations:
            combination_info       = []
            group_selected_vehicle = []
            combination_cost = 0
            for group in combination:
                # 그룹의 최적 시간과 최적 순서 그 최적 값을 보내기 위한 최소 출발 시간을 반환받음
                best_distance, best_permutation, min_departure = self._group_minimum_distance_order(group)
                # 그 그룹을 실을때 크게 손해 보지 않는 차 종류 선택
                min_vehicle, max_vehicle                       = self._group_cbm_range(group)
                
                # 그 그룹의 차 선택하기
                minimum_group_cost = float("inf")
                # 출발 터미널에서 부터 자신을 포함한 가장 가까운 터미널 부터 조사 시작
                if best_distance == float("inf"):
                    average_length = 0
                    for i in range(len(group)):
                        average_length += self.info.distance_matrix[self.info.vertex_to_index[group[i][8]]][self.info.vertex_to_index[group[i][3]]]
                        average_length += self.info.distance_matrix[self.info.vertex_to_index[group[i][3]]][self.info.vertex_to_index[self.info.nearest_termnial_from_D[group[i][3]]]]
                    average_length /= len(group)
                    average_length *= 1 + 0.15*len(group)
                    self.info.distance_matrix
                    minimum_group_cost = average_length * np.sum(np.array(group)[:, 4]) * 100
                    combination_cost += minimum_group_cost
                    combination_info.append([float("inf"), best_permutation, ["", -1, -1, -1, -1, float("inf")], float("inf"), 0])
                else:
                    selected_vehicle   = ["", -1, -1, -1, -1, float("inf"), 0]
                    for terminal in self.info.nearest_terminal_from_O[terminal_index]:
                        
                        # 100km보다 먼 terminal은 조사하지 않음
                        if self.info.distance_matrix[self.info.vertex_to_index[terminal]][self.info.destination_num + terminal_index] > config.BURST_CALL_BOUNDARY and 0 not in np.array(group)[:,-1]:
                            break
                        for i in range(min_vehicle, max_vehicle + 1):
                            break_vehicle = 0
                            # 그 터미널에 특정 type의 vehicle이 존재하지 않으면 다음으로 넘어감
                            if len(terminal_vehicle[self.info.terminal_to_index[terminal]][i]) != 0:
                                # 가지고 있는 vehicle 중에서 min_departure 안에 출발 할 수 있는 가장 가까운 vehicle 선택 
                                best_cost_vehicles = [
                                    [terminal_vehicle[self.info.terminal_to_index[terminal]][i][j], 
                                    vehicle_info[self.info.vehicle_to_index[terminal_vehicle[self.info.terminal_to_index[terminal]][i][j]]][1]] 
                                    for j in range(len(terminal_vehicle[self.info.terminal_to_index[terminal]][i]))
                                ]
                                best_cost_vehicles = sorted(best_cost_vehicles, key=lambda x:x[1])
                                
                                closest_vehicle_index = -1
                                for j in range(len(best_cost_vehicles)):
                                    near_terminal_to_terminal_time = self.info.time_matrix[self.info.vertex_to_index[terminal]][self.info.destination_num + terminal_index]
                                    vehicle_start = best_cost_vehicles[j][1]
                                    if best_cost_vehicles[j][1] < self.batch_datetime:
                                        vehicle_start = self.batch_datetime
                                    if vehicle_start + datetime.timedelta(minutes=near_terminal_to_terminal_time) < min_departure and best_cost_vehicles[j][0] not in group_selected_vehicle: ##
                                        closest_vehicle_index = j
                                # min_departure안에 출발 시킬 수 있는게 없으면 다음 으로 넘어가고 출발 시킬 수 있는게 있으면 그 vehicle 선택
                                
                                if closest_vehicle_index != -1:
                                    break_vehicle = 1
                                    #del terminal_vehicle[self.info.terminal_to_index[terminal]][i][delete_index]
                                    minimum_group_cost  = best_distance + self.info.distance_matrix[self.info.vertex_to_index[terminal]][self.info.destination_num + terminal_index]
                                    minimum_group_cost *= self.info.variable_cost[i]
                                    group_fixed_cost = self.info.fixed_cost[i] * abs(self.vehicle_info[self.info.vehicle_to_index[best_cost_vehicles[closest_vehicle_index][0]]][2] - 1) * (1/(28 - self.ith_batch))
                                    if minimum_group_cost + group_fixed_cost < selected_vehicle[5] + selected_vehicle[6]:
                                        selected_vehicle = [
                                            best_cost_vehicles[closest_vehicle_index][0],
                                            self.info.terminal_to_index[terminal],
                                            terminal_vehicle[self.info.terminal_to_index[terminal]][i].index(best_cost_vehicles[closest_vehicle_index][0]),
                                            i, terminal, minimum_group_cost,
                                            group_fixed_cost,
                                        ]
                            if break_vehicle == 1:
                                max_vehicle = i
                                if vehicle_info[self.info.vehicle_to_index[selected_vehicle[0]]][2] == 1:
                                    max_vehicle -= 1
                                break  
                    
                    if minimum_group_cost == float("inf"):
                        combination_cost += best_distance * np.sum(np.array(group)[:,4]) * 100
                        combination_info.append([float("inf"), best_permutation, selected_vehicle, best_distance])
                    else:
                        combination_cost += selected_vehicle[5] + selected_vehicle[6]
                        group_selected_vehicle.append(selected_vehicle[0])
                        combination_info.append([selected_vehicle[5], best_permutation, selected_vehicle, best_distance + self.info.distance_matrix[self.info.destination_num + selected_vehicle[1]][self.info.destination_num + terminal_index]])
            
            if best_combination_cost > combination_cost:
                best_combination = copy.deepcopy(combination_info)
                best_combination_cost = combination_cost
                
            # 각 그룹이 차 선택까지 끝난후에 이 조합이 최선이었는지 판단
            # 이 combination에서 처리한 variable cost/cbm 이 가장 좋은 것을 반환
            
        # 이 cluster의 최적의 cost를 가진 주문배치 정보 반환
        for i, group in enumerate(best_combination):
            if group[2][0] != '':
                delete_index = terminal_vehicle[self.info.terminal_to_index[group[2][4]]][group[2][3]].index(group[2][0])
                del terminal_vehicle[self.info.terminal_to_index[group[2][4]]][group[2][3]][delete_index]
            best_combination[i] = [group[0], group[1], group[2][0], group[2][1], group[3]]
            
        self.best_combination = best_combination
        self._vehicle_info_update(terminal_vehicle, vehicle_info, terminal_index)
    
    def _group_cbm_range(self, group):
        group_cbm = np.sum(np.array(group)[:, 4])
        
        min_index = -1
        max_index = -1
        for i in range(len(self.info.cbm)):
            if min_index == -1 and group_cbm <= self.info.cbm[i]:
                min_index = i
            if (self.info.variable_cost[i] / group_cbm) < config.CBM_BOUNDARY:
                max_index = i 
        
        if 0 in np.array(group)[:,-1]:
            return min_index, len(self.info.cbm) - 1
        
        return min_index, max_index
    
    def _group_minimum_distance_order(self, group):
        minimum_distance  = float("inf")
        best_permutation  = group
        minimum_departure = datetime.timedelta()
        for permutation in itertools.permutations(group):
            possible_departure_time = self._permutation_time_check(permutation)
            if possible_departure_time == False:
                continue
            permutation_distance = self._permutation_distnace(permutation)
            if permutation_distance < minimum_distance:
                minimum_distance  = permutation_distance
                best_permutation  = permutation
                minimum_departure = possible_departure_time
        return minimum_distance, best_permutation, minimum_departure
    
    def _permutation_time_check(self, permutation):
        waiting_time = datetime.timedelta(0)
        time_flow    = datetime.datetime(year=2000, month=1, day=30) + permutation[-1][6]
        time_flow_s  = copy.deepcopy(time_flow)
        for i in range(len(permutation) - 1, 0, -1):
            time_flow -= datetime.timedelta(
                minutes = self.info.time_matrix
                [self.info.vertex_to_index[permutation[i - 1][3]]]
                [self.info.vertex_to_index[permutation[i    ][3]]]
            )
            
            if permutation[i - 1][3] != permutation[i][3]:
                time_flow -= datetime.timedelta(hours = 1)
            
            destination_time_s = datetime.datetime(time_flow.year, time_flow.month, time_flow.day) + permutation[i - 1][5]
            destination_time_e = datetime.datetime(time_flow.year, time_flow.month, time_flow.day) + permutation[i - 1][6]
            
            if destination_time_s <= time_flow and time_flow <= destination_time_e:
                waiting_time += datetime.timedelta(0)
            elif destination_time_s - datetime.timedelta(days=1) <= time_flow and time_flow <= destination_time_e - datetime.timedelta(days=1):
                waiting_time += datetime.timedelta(0)
            else:
                if time_flow < destination_time_s:
                    waiting_time += time_flow - destination_time_e + datetime.timedelta(days=1)
                    time_flow     = destination_time_e - datetime.timedelta(days=1)
                elif destination_time_e < time_flow:
                    waiting_time += time_flow - destination_time_e 
                    time_flow = destination_time_e
        
        time_flow -= datetime.timedelta(
            minutes = self.info.time_matrix
            [self.info.vertex_to_index[permutation[0][8]]]
            [self.info.vertex_to_index[permutation[0][3]]]
        )        
        
        time_flow_e = copy.deepcopy(time_flow)
        
        possible_departure_time_in_batch = datetime.datetime(
            self.batch_datetime.year, self.batch_datetime.month, self.batch_datetime.day,
            time_flow.hour, time_flow.minute, time_flow.second, time_flow.microsecond,
        )
        
        
        if possible_departure_time_in_batch < self.batch_datetime:
            possible_departure_time_in_batch += datetime.timedelta(days=1)
            waiting_time += possible_departure_time_in_batch - self.batch_datetime
        
        if waiting_time > datetime.timedelta(hours=0) and 0 not in np.array(permutation)[:,-1]:
            if waiting_time > datetime.timedelta(hours=config.WAITING_TIME_BOUNDARY):
                return False
        
        if possible_departure_time_in_batch + (time_flow_s -  time_flow_e) >= datetime.datetime(config.YEAR, config.MONTH, config.END_DAY, 23):
            return False
        
        return possible_departure_time_in_batch
    
    def _permutation_distnace(self, permutation):
        total_distance = 0
        for i in range(len(permutation) - 1):
            total_distance += (
                self.info.distance_matrix
                [self.info.vertex_to_index[permutation[i    ][3]]]
                [self.info.vertex_to_index[permutation[i + 1][3]]]
            )
        total_distance += (
            self.info.distance_matrix
            [self.info.vertex_to_index[permutation[0][8]]]
            [self.info.vertex_to_index[permutation[0][3]]]
        )
        total_distance += (
            self.info.distance_matrix
            [self.info.vertex_to_index[permutation[-1][3]]]
            [self.info.vertex_to_index[self.info.nearest_termnial_from_D[permutation[-1][3]]]]
        )
        return total_distance
    
    def _vehicle_info_update(self, terminal_vehicle, vehicle_info, terminal_index):
        #[[group],[group[cost, permutation, vehicle_name, vehicle_terminal, distance]],[]]
        self.fail_group          = []
        for group in self.best_combination:
            arrive_time          = []
            waiting_time         = []
            departure_time       = []
            group_travle_time    = 0
            vehicle_waiting_time = datetime.timedelta(0)
            
            if group[0] == float("inf"):
                for order in group[1]:
                    self.fail_group.append(order)
                continue
            
            time_flow          = vehicle_info[self.info.vehicle_to_index[group[2]]][1]
            if time_flow < self.batch_datetime:
                time_flow = self.batch_datetime
            time_flow         += datetime.timedelta(minutes=self.info.time_matrix[self.info.destination_num + group[3]][self.info.destination_num + terminal_index])
            group_travle_time += self.info.time_matrix[group[3]][self.info.destination_num + terminal_index]
            veh_arrival_time = copy.deepcopy(time_flow)
            for i in range(len(group[1])):
                if i == 0:
                    time_flow         += datetime.timedelta(minutes=self.info.time_matrix[self.info.destination_num + terminal_index][self.info.vertex_to_index[group[1][i][3]]])
                    group_travle_time += self.info.time_matrix[self.info.destination_num + terminal_index][self.info.vertex_to_index[group[1][i][3]]]
                else:
                    time_flow         += datetime.timedelta(minutes=self.info.time_matrix[self.info.vertex_to_index[group[1][i - 1][3]]][self.info.vertex_to_index[group[1][i][3]]])
                    group_travle_time += self.info.time_matrix[self.info.vertex_to_index[group[1][i - 1][3]]][self.info.vertex_to_index[group[1][i][3]]]
                arrive_time.append(time_flow)
                
                order_time_s = datetime.datetime(time_flow.year, time_flow.month, time_flow.day) + group[1][i][5]
                order_time_e = datetime.datetime(time_flow.year, time_flow.month, time_flow.day) + group[1][i][6]
                
                if order_time_s <= time_flow and time_flow <= order_time_e:
                    if i != 0 and group[1][i - 1][3] == group[1][i][3]:
                        waiting_time.append(waiting_time[-1])
                    else:
                        waiting_time.append("00:00:00")
                elif order_time_s - datetime.timedelta(days=1) <= time_flow and time_flow <= order_time_e - datetime.timedelta(days=1):
                    if i != 0 and group[1][i - 1][3] == group[1][i][3]:
                        waiting_time.append(waiting_time[-1])
                    else:
                        waiting_time.append("00:00:00")
                else:
                    if time_flow < order_time_s:
                        waiting_time.append(str(order_time_s - time_flow))
                        vehicle_waiting_time += order_time_s - time_flow
                        time_flow     = order_time_s
                    elif order_time_e < time_flow:
                        waiting_time.append(str(order_time_s + datetime.timedelta(days=1) - time_flow))
                        vehicle_waiting_time += order_time_s + datetime.timedelta(days=1) - time_flow
                        time_flow = order_time_s + datetime.timedelta(days=1)
                
                if i == len(group[1]) - 1 or group[1][i][3] != group[1][i + 1][3]:
                    time_flow += datetime.timedelta(hours=1)
                    departure_time.append(time_flow)
                else:
                    departure_time.append(time_flow + datetime.timedelta(hours=1))
                
            time_flow         += datetime.timedelta(minutes=self.info.time_matrix[self.info.vertex_to_index[group[1][-1][3]]][self.info.vertex_to_index[self.info.nearest_termnial_from_D[group[1][-1][3]]]])
            group_travle_time += self.info.time_matrix[self.info.vertex_to_index[group[1][-1][3]]][self.info.vertex_to_index[self.info.nearest_termnial_from_D[group[1][-1][3]]]]
            vehicle_info[self.info.vehicle_to_index[group[2]]][1] = time_flow
            vehicle_info[self.info.vehicle_to_index[group[2]]][2] = 1
            terminal_vehicle[self.info.terminal_to_index[self.info.nearest_termnial_from_D[group[1][-1][3]]]][vehicle_info[self.info.vehicle_to_index[group[2]]][3]].append(group[2])
            
            # group[cost, permutation, vehicle_name, vehicle_terminal, distance]
            # 터미널 정보 추가
            self.info.terminal_to_index 
            self.group_result.append(["Null", group[2], 0, self.info.index_to_terminal[group[3]], time_flow.strftime("%Y-%m-%d %H:%M"), 0, 0, time_flow.strftime("%Y-%m-%d %H:%M"), "Null"])
            self.after_result.append(["Null", group[2], 0, self.info.index_to_terminal[group[3]], 
                                        veh_arrival_time.strftime("%Y-%m-%d %H:%M"), # 도착시간
                                        0, # 대기시간 = 출발시간 - 도착시간
                                        0, # 서비스 시간?
                                        veh_arrival_time.strftime("%Y-%m-%d %H:%M"), # 차량이 터미널에서 출발하는 시간
                                        "Null"
                                        ])
            
            for i, order in enumerate(group[1]):
                self.group_result.append([order[0], group[2], 0, order[3], "Null", "Null", "Null", "Null", "No"])
                self.after_result.append([order[0], group[2], 0, order[3], 
                                        arrive_time[i].strftime("%Y-%m-%d %H:%M"),
                                        waiting_time[i].split(".")[0],
                                        int(order[7]), 
                                        departure_time[i].strftime("%Y-%m-%d %H:%M"),
                                        "Yes"
                                        ])
            
            for i, order in enumerate(self.group_result):
                if order[8] == "Yes":
                    continue
                if datetime.datetime.strptime(self.after_result[i][7], "%Y-%m-%d %H:%M") <= self.batch_datetime:
                    self.group_result[i] = self.after_result[i]
            
            
            vehicle_index = self.info.vehicle_to_index[group[2]]
            hour, minute, second = str(vehicle_waiting_time).split(":")
            if len(list(hour))>=3:
                day, hour = hour.split(",")
                self.vehicle_result[vehicle_index][7] += float(24*60*float(day[0]) + 60*float(hour) + float(minute) + float(second)/60)  # 총 대기시간
            else: 
                self.vehicle_result[vehicle_index][7] += float(60*float(hour) + float(minute) + float(second)/60) #총 대기시간
            
            # group[cost, permutation, vehicle_name, vehicle_terminal, distance]
            self.vehicle_result[vehicle_index][1]  += len(group[1])                                                                      # 배달한 주문 수
            self.vehicle_result[vehicle_index][2]  += np.sum(np.array(group[1])[:, 4])                                                   # 총 적재량
            self.vehicle_result[vehicle_index][6]  += float(len(set(map(lambda x:x[3], group[1]))) * config.WORKING_TIME_MINITE)                                 # 총 하역시간
            self.vehicle_result[vehicle_index][9]   = self.info.fixed_cost[self.info.total_vehicle_info[vehicle_index][3]]               # 차량 고정비
            self.vehicle_result[vehicle_index][10] += group[0]                                                                           # 차량 거리 운영비
            self.vehicle_result[vehicle_index][8]   = self.vehicle_result[vehicle_index][9] + self.vehicle_result[vehicle_index][10]     # 총 비용
            self.vehicle_result[vehicle_index][3]  += group[4]                                                                           # 총 주행거리
            self.vehicle_result[vehicle_index][5]  += group_travle_time                                                                  # 총 이동 시간
            self.vehicle_result[vehicle_index][4]   = self.vehicle_result[vehicle_index][5] + self.vehicle_result[vehicle_index][6] + self.vehicle_result[vehicle_index][7] # 총 작업 시간
        
        self.vehicle_info     = copy.deepcopy(vehicle_info)
        self.terminal_vehicle = copy.deepcopy(terminal_vehicle)