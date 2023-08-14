import numpy as np
import itertools, datetime
from FInal_Basic_Information import Information

class Calculator():
    def __init__(self, possible_combinations, information, batch_datetime, terminal_index, terminal_vehicle, vehicle_info):
        self.info           : Information
        self.info           = information
        self.batch_datetime = batch_datetime
        for combination in possible_combinations:
            combination_cost = 0
            best_combintaion = [[permutation, vehicle], [], [],]
            for group in combination:
                # 그룹의 최적 시간과 최적 순서 그 최적 값을 보내기 위한 최소 출발 시간을 반환받음
                best_distance, best_permutation, min_departure = self._group_minimum_distance_order(group)
                # 그 그룹을 실을때 크게 손해 보지 않는 차 종류 선택
                min_vehicle, max_vehicle                       = self._group_cbm_range(group)
                
                # 그 그룹의 차 선택하기
                minimum_group_cost = 0
                selected_vehicle   = ""
                # 출발 터미널에서 부터 자신을 포함한 가장 가까운 터미널 부터 조사 시작
                for terminal in self.info.nearest_terminal_from_O[terminal_index]:
                    # 100km보다 먼 terminal은 조사하지 않음
                    if self.info.distance_matrix[self.info.vertex_to_index[terminal]][self.info.destination_num + terminal_index] > 100:
                        break
                    have_to_break = False
                    for i in range(min_vehicle, max_vehicle + 1):
                        # 그 터미널에 특정 type의 vehicle이 존재하지 않으면 다음으로 넘어감
                        if len(terminal_vehicle[self.info.terminal_to_index[terminal]][i]) != 0:
                            # 가지고 있는 vehicle 중에서 min_departure 안에 출발 할 수 있는 가장 가까운 vehicle 선택 
                            have_to_break = True
                            best_cost_vehicles = [
                                [terminal_vehicle[self.info.terminal_to_index[terminal]][i][j], 
                                 vehicle_info[terminal_vehicle[self.info.terminal_to_index[terminal]][i][j]][1]] 
                                for j in range(len(terminal_vehicle[self.info.terminal_to_index[terminal]][i]))
                            ]
                            best_cost_vehicles = sorted(best_cost_vehicles, key=lambda x:x[1])
                            
                            closest_vehicle_index = -1
                            for j in range(len(best_cost_vehicles)):
                                if closest_vehicle_index == -1 and best_cost_vehicles[j][1] - min_departure > 0:
                                    closest_vehicle_index = j
                            
                            # min_departure안에 출발 시킬 수 있는게 없으면 다음 으로 넘어가고 출발 시킬 수 있는게 있으면 그 vehicle 선택
                            if closest_vehicle_index <= 0:
                                have_to_break = False
                            else:
                                selected_vehicle = best_cost_vehicles[closest_vehicle_index - 1][0]
                                delete_index     = terminal_vehicle[self.info.terminal_to_index[terminal]][i].indexOf(selected_vehicle)
                                del terminal_vehicle[self.info.terminal_to_index[terminal]][i][delete_index]
                                minimum_group_cost  = best_distance + self.info.distance_matrix[self.info.vertex_to_index[terminal]][self.info.destination_num + terminal_index]
                                minimum_group_cost *= self.info.variable_cost
                    if have_to_break == True:
                        break
                
            # 각 그룹이 차 선택까지 끝난후에 이 조합이 최선이었는지 판단
            # 이 combination에서 처리한 variable cost/cbm 이 가장 좋은 것을 반환
        # 이 cluster의 최적의 cost를 가진 주문배치 정보 반환
        self.best_arrange_cluster = 0
    
    def _group_cbm_range(self, group):
        group_cbm = np.sum(np.array(group)[:, 4])
        
        min_index = -1
        max_index = -1
        for i in range(len(self.info.cbm)):
            if min_index == -1 and group_cbm <= self.info.cbm[i]:
                min_index = i
            if (self.info.variable_cost[i] / group_cbm) < 0.04:
                max_index = i 
        
        return min_index, max_index
    
    def _group_minimum_distance_order(self, group):
        minimum_distance  = float("inf")
        best_permutation  = []
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
        time_flow    = datetime.datetime(year=2000, month=1, day=1, hour=1) + permutation[-1][6]
        for i in range(len(permutation) - 1, 0, -1):
            time_flow -= datetime.timedelta(
                hours   = 1,
                minutes = self.info.time_matrix
                [self.info.vertex_to_index[permutation[i - 1][3]]]
                [self.info.vertex_to_index[permutation[i    ][3]]]
            )
            
            if i < len(permutation) - 1 and permutation[i] == permutation[i + 1]:
                time_flow += datetime.timedelta(hours = 1)
            
            destination_time_s = datetime.datetime(time_flow.year, time_flow.month, time_flow.day) + permutation[i - 1][5]
            destination_time_e = datetime.datetime(time_flow.year, time_flow.month, time_flow.day) + permutation[i - 1][6]
            if permutation[i - 1][6].days > 0:
                destination_time_s -= datetime.timedelta(days=1)
                destination_time_e -= datetime.timedelta(days=1)
            if destination_time_s > time_flow:
                return False
            if destination_time_e < time_flow:
                time_flow     = destination_time_e
                waiting_time += time_flow - destination_time_e
        
        if waiting_time > datetime.timedelta(hours=4):
            return False
        
        time_flow -= datetime.timedelta(
            minutes = self.info.time_matrix
            [self.info.vertex_to_index[permutation[0][8]]]
            [self.info.vertex_to_index[permutation[0][3]]]
        )
        
        if time_flow.hour < self.batch_datetime.hour:
            return False
        
        possible_departure_time_in_batch = datetime.datetime(
            self.batch_datetime.year, self.batch_datetime.month, self.batch_datetime.day,
            time_flow.hour, time_flow.minute, time_flow.second, time_flow.microsecond,
        )
        
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
    
    