import csv
import math, copy, itertools, time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from Final_K_Means_Clustering import CJ_Cluster

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def make_terminal_to_idx():
    file_terminals = open('Terminals.csv', 'r', encoding='cp949')
    terminals_info = csv.reader(file_terminals)
    terminal = []
    terminal_idx = {}
    for i, list in enumerate(terminals_info):
        if i != 0:
            terminal.append(list)
            terminal[-1][1] = float(terminal[-1][1])
            terminal[-1][2] = float(terminal[-1][2])
            terminal_idx[list[0]] = i - 1
    file_terminals.close()
    print("complete make terminal to idx")
    return terminal, terminal_idx

def make_veh_info(tti, ttn):
    file_veh = open('veh_table.csv', 'r', encoding='cp949')
    veh_info = csv.reader(file_veh)
    veh = []
    total_veh = []
    veh_idx = {}
    terminal_veh = [[[], [], [], [], []] for _ in range(ttn)]
    result_veh = []
    global kind_of_veh
    now = datetime.datetime(2023, 5, 1, 0, 0, 0)
    for i, list in enumerate(veh_info):
        if i != 0:
            veh_idx[list[0]] = i - 1
            temp = [list[5], now, 0, kind_of_veh[list[4]]]
            result_veh.append([list[0], 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            total_veh.append(temp)
            terminal_veh[tti[list[5]]][kind_of_veh[list[4]]].append(list[0])
            temp2 = [int(list[4]), int(list[6]), float(list[7])]
            if temp2 not in veh:
                veh.append(temp2)
    file_veh.close()
    print("complete make veh info")
    return sorted(veh), terminal_veh, total_veh, veh_idx, result_veh

def make_point_to_idx():  # 모든 vertix를 모아서 vertix마다 index 부여
    pti = {}
    tp = []
    file_terminals = open('Terminals.csv', 'r', encoding='cp949')
    file_orders = open('orders_table.csv', 'r', encoding='cp949')

    orders_info = csv.reader(file_orders)
    terminals_info = csv.reader(file_terminals)

    for i, line in enumerate(orders_info):
        if i != 0:
            if line[3] not in tp:
                tp.append(line[3])

    for i, line in enumerate(terminals_info):
        if i != 0:
            if line[0] not in tp:
                tp.append(line[0])
    tp = sorted(tp, key=lambda x: (x[0], int(x[2:])))

    for i, point in enumerate(tp):
        pti[point] = i
    file_terminals.close()
    file_orders.close()
    print("complete make point to idx")
    return pti, tp

def make_distance_time_matrix(pti, size):
    file_od = open('od_matrix.csv', 'r')
    od_info = csv.reader(file_od)
    distance = [[0 for i in range(size)] for j in range(size)]
    time = [[0 for i in range(size)] for j in range(size)]
    for i, list in enumerate(od_info):
        if i != 0:
            distance[pti[list[0]]][pti[list[1]]] = float(list[2])
            time[pti[list[0]]][pti[list[1]]] = float(list[3])
    file_od.close()
    print("complete make distance time matrix")
    return distance, time

def make_nearest_terminal(tt, ttn, dm, pti):
    nearest = [[] for i in range(ttn)]
    for i in range(ttn):
        temp = []
        for j in range(ttn):
            temp.append([tt[j][0], dm[pti[tt[i][0]]][pti[tt[j][0]]]])
        temp = sorted(temp, key=lambda x: x[1])
        for j in range(ttn):
            nearest[i].append(temp[j][0])
    return nearest

def collecting_order(tti, ttn, tdn, gn, mt):
    file_orders = open('orders_table.csv', 'r', encoding='cp949')
    orders_info = csv.reader(file_orders)
    orders = [[[[] for i in range(gn)] for i in range(tdn)] for i in range(ttn)]
    orders_accurate_cbm = {}
    for i, list in enumerate(orders_info):
        if i != 0:
            list.append(mt)
            orders_accurate_cbm[list[0]] = float(list[4])
            list[4] = math.ceil(float(list[4]))
            list[1] = float(list[1])
            list[2] = float(list[2])
            orders[tti[list[8]]][int(list[9].split('-')[2]) - 1][int(list[10])].append(list)
    file_orders.close()
    print("complete collecting_order")
    return orders, orders_accurate_cbm

def judge_can_delivery(now, target, same_place):
    start = datetime.datetime(now.year, now.month, now.day, int(target[5].split(':')[0]),
                              int(target[5].split(':')[1]))
    end = datetime.datetime(now.year, now.month, now.day, int(target[6].split(':')[0]),
                            int(target[6].split(':')[1]))
    if target[6] == '2:00':
        end += datetime.timedelta(days=1)
    if now <= start: # 대기가 필요할 때
        tmp = str(start - now).split(".")
        return start + datetime.timedelta(hours=1), tmp[0] # 하차 후의 시간  ,대기 시간
    elif now > start and now <= end: #바로 하차가 가능한 경우
        if same_place == 1:
            return now, "00:00:00"
        return now + datetime.timedelta(hours=1), "00:00:00"
    else:
        tmp = str(start + datetime.timedelta(days=1) - now).split(".")
        return start + datetime.timedelta(days=1) + datetime.timedelta(hours=1), tmp[0]

def make_order_info():
    file_orders = open('orders_table.csv', 'r', encoding='cp949')
    orders_info = csv.reader(file_orders)
    order_date = []
    order_group = []
    for i, list in enumerate(orders_info):
        if i != 0:
            if list[9] not in order_date:
                order_date.append(list[9])
            if list[10] not in order_group:
                order_group.append(list[10])
    file_orders.close()
    return sorted(order_date), sorted(order_group)

def cal_cbm_idx_range(G):
    one_group_cbm = 0
    emergen_judge = 0
    mini_idx = -1
    maxi_idx = -1
    global cbm_of_veh,variable_cost
    for group_cbm in G:
        one_group_cbm += group_cbm[4]   #현재 group의 총 cbm 계산
        if group_cbm[-1] < 1:
            emergen_judge = 1   #현재 날짜에서 반드시 처리해야되는 group인지 판단
    for i in range(5):
        if mini_idx == -1 and one_group_cbm <= cbm_of_veh[i]:
            mini_idx = i    #현재 group를 담을 수 있는 가장 작은 트럭 idx
        if (variable_cost[i] / one_group_cbm) < 0.04: # 연비 대비 담은 cbm을 기준으로 가장 큰 트럭 idx 계산
            maxi_idx = i    
    return mini_idx, maxi_idx+1, emergen_judge

def cal_min_arrive_time(per, tm, pti, nd):
    min_time = datetime.datetime(2000, 1, 30, int(per[-1][6].split(':')[0]),
                                 int(per[-1][6].split(':')[1]), 0) # 년, 월, 일은 상관 X
    
    for i in range(len(per) - 1, 0, -1):
        min_time -= datetime.timedelta(minutes=tm[pti[per[i - 1][3]]][pti[per[i][3]]])  # 도착지 -> 도착지 시간
        if per[i-1][3] != per[i][3]:
            min_time -= datetime.timedelta(hours=1)  # 하차시간
        
        temp_time_s = datetime.datetime(min_time.year, min_time.month, min_time.day, int(per[i - 1][5].split(':')[0]),
                                        int(per[i - 1][5].split(':')[1]), 0)  # 출발시각
        temp_time_e = datetime.datetime(min_time.year, min_time.month, min_time.day, int(per[i - 1][6].split(':')[0]),
                                        int(per[i - 1][6].split(':')[1]), 0)  # 도착시각
        if per[i - 1][6] == '2:00':
            temp_time_s -= datetime.timedelta(days=1) # 도착 시간이 새벽 2시일 경우 day + 1
        if temp_time_s > min_time:
            min_time = temp_time_e - datetime.timedelta(days=1)
        if temp_time_e < min_time:
            min_time = temp_time_e
    min_time -= datetime.timedelta(minutes=tm[pti[per[0][8]]][pti[per[0][3]]])
    ans_time = datetime.datetime(nd.year, nd.month, nd.day, min_time.hour, min_time.minute, min_time.second, min_time.microsecond)
    if min_time.hour < nd.hour: # 최소 출발 시간이 현재 시간보다 전일 경우 day + 1
        ans_time += datetime.timedelta(days=1)
    return ans_time

def cal_cost(bruteforce, pti, dm, tm, nt, etv, itv, tvi, neart, tti, nd, db, ld, near_p):
    cost = [float("inf")]
    veh = []
    sequence = []
    arrive_time = []
    loss = []
    veh_variable_cost=[]
    veh_travel_distance=[]
    veh_travle_time=[]
    final_arrival_time = []
    final_wait_time = []
    final_depart_time = []
    fail_idx = []
    global cbm_of_veh, each_veh_info, now_date , idx_to_veh, variable_cost, fixed_cost

    for one_brute in bruteforce:
        one_brute_cost = []# 현재 brute에서 각 group별 cost
        one_brute_veh = []  # 현재 brute에서 각 group이 사용한 차량들
        one_brute_seq = [] # 현재 brute에서 각 group별 order 순서
        one_brute_at = [] # 현재 brute에서 각 group별 차량이 주문을 모두 처리하고 터미널로 돌아온 시간
        one_brute_loss = [] # 현재 brute에서 각 group별 손실
        one_brute_fail = [] # 현재 brute에서 각 group별 fail 처리 유무
        one_brute_variable_cost = []
        one_brute_travel_distance=[]
        one_brute_travle_time = []
        brute_order_arrival_time = []
        brute_order_wait_time = []
        brute_order_depart_time = []
        for ogrp_idx, one_group in enumerate(one_brute):  # 각 가능한 그룹 마다
            mini_cbm_idx, maxi_cbm_idx,emergency_group = cal_cbm_idx_range(one_group)  # 최소 cbm 차량 idx 반환
            if str(nd).split(" ")[0] == ld: #현재가 마지막 날이라면 실패 X
                emergency_group = 1
            if emergency_group == 1:
                maxi_cbm_idx = 5    # 반드시 처리해야되는 그룹이라면 연비 대비 cbm 값에 상관없이 차량 끝까지 탐색
            one_group_cost = float("inf")
            one_group_seq = []
            one_group_at = None
            one_group_veh = [-1, -1]
            one_group_travle_distance= 0
            one_group_travle_time = 0
            one_group_loss = None
            group_fail = -1 # fail 처리 / -1 : fail X / 0 이상의 값(brute에서 fail된 group의 idx번호) : fail
            # 각 주문별 도착시간, 대기시간(차량이 하차지에 도착해서 하차작업을 시작하기 전까지의 시간), 출발 시간
            each_order_arrive_time = []
            each_order_wait_time = []
            each_order_depart_time = []
            for permute in itertools.permutations(one_group):  # 한 그룹에 가능한 모든 주문 순서를 돌음.
                permute_min_arrive_time = cal_min_arrive_time(permute, tm, pti, nd)  # 최소 터미널에서의 출발 시간
                if permute_min_arrive_time == False:
                    continue
                permute_veh = [-1, -1, -1]
                now_car_idx = -1
                now_car_used = -1
                distance = 0
                veh_time = None
                travel_time = 0 #이동 시간
                arrival_time = []
                wait_time = []
                depart_time = []
                permute_loss = [0,0,0] # [new car, large cbm, burst call]
                min_car_cost = float("inf")
                for t in neart:  # 자기 터미널 포함 가까운 터미널들
                    if dm[pti[t]][pti[nt]] > 100 and emergency_group == 0: # 100km 이내의 터미널까지만 차량지원 허용 / 단 반드시 처리해야될 group일 경우 100km 제한 X
                        break
                    for i in range(mini_cbm_idx, maxi_cbm_idx):
                        for nv in etv[tti[t]][i]:
                            if (tvi[itv[nv]][1] <= permute_min_arrive_time) and (
                                    [nv, t,i] not in one_brute_veh):  # 사용가능한 차량이 최소 출발시간보다 작고, 이번 그룹에서 사용안한 차량이라면
                                this_car_cost = 0
                                if tvi[itv[nv]][2] == 0:
                                    this_car_cost += fixed_cost[i]
                                this_car_cost += dm[pti[t]][pti[nt]] * variable_cost[i]
                                this_car_cost += dm[pti[nt]][pti[permute[0][3]]] * variable_cost[i]
                                for k in range(len(one_group) - 1):
                                    this_car_cost += dm[pti[permute[k][3]]][pti[permute[k + 1][3]]] * variable_cost[i] # 도착지 -> 도착지

                                this_car_cost += dm[pti[permute[-1][3]]][pti[near_p[pti[permute[-1][3]]]]] * variable_cost[i]    # 현재 차량을 사용했을 때의 cost
                                if min_car_cost > this_car_cost:     
                                    permute_veh = [nv, t, i] #[차량 이름, 차량 있던 터미널, 차량 cbm idx]
                                    now_car_idx = i
                                    veh_time = tvi[itv[nv]][1]  # 사용가능한 차량의 출발 가능 시간
                                    now_car_used = tvi[itv[nv]][2]
                                    min_car_cost = this_car_cost
                if now_car_idx == -1:   #선택할 수 있는 차량이 하나도 없는 경우
                    continue
                permute_cost = 0
                if now_car_used == 0: # new car loss 발생
                    permute_cost += each_veh_info[now_car_idx][1]
                    permute_loss[0] += each_veh_info[now_car_idx][1]
                permute_cost += dm[pti[permute_veh[1]]][pti[nt]] * each_veh_info[now_car_idx][2] # 타 터미널 -> 내 터미널 burst call loss
                permute_loss[2] = dm[pti[permute_veh[1]]][pti[nt]] * each_veh_info[now_car_idx][2]
                distance += dm[pti[permute_veh[1]]][pti[nt]]
                permute_cost += dm[pti[nt]][pti[permute[0][3]]] * each_veh_info[now_car_idx][2] # 내 터미널 -> 첫번째
                distance += dm[pti[nt]][pti[permute[0][3]]]

                veh_time += datetime.timedelta(minutes=tm[pti[nt]][pti[permute[0][3]]])  # 터미널 ->첫번째 목적지까지의 시간
                travel_time += tm[pti[nt]][pti[permute[0][3]]]
                arrival_time.append(veh_time)
                veh_time, wait = judge_can_delivery(veh_time, permute[0],0)  # 하차가 끝나고 다음 목적지로 출발할 때의 시간
                wait_time.append(wait)
                depart_time.append(veh_time)

                for k in range(len(one_group) - 1):
                    permute_cost += dm[pti[permute[k][3]]][pti[permute[k + 1][3]]] * each_veh_info[now_car_idx][2] # 도착지 -> 도착지
                    distance += dm[pti[permute[k][3]]][pti[permute[k + 1][3]]]
                    veh_time += datetime.timedelta(minutes=tm[pti[permute[k][3]]][pti[permute[k + 1][3]]])
                    travel_time +=tm[pti[permute[k][3]]][pti[permute[k + 1][3]]]
                    arrival_time.append(veh_time)
                    veh_time, wait = judge_can_delivery(veh_time, permute[k + 1], permute[k][3] == permute[k+1][3])
                    wait_time.append(wait)
                    depart_time.append(veh_time)
                permute_cost += dm[pti[permute[-1][3]]][pti[near_p[pti[permute[-1][3]]]]] * each_veh_info[now_car_idx][2]
                distance += dm[pti[permute[-1][3]]][pti[near_p[pti[permute[-1][3]]]]]
                veh_time += datetime.timedelta(minutes=tm[pti[permute[-1][3]]][pti[near_p[pti[permute[-1][3]]]]])
                permute_loss[1] = distance * (variable_cost[now_car_idx] - variable_cost[mini_cbm_idx])
                travel_time += tm[pti[permute[-1][3]]][pti[near_p[pti[permute[-1][3]]]]]
                if one_group_cost > permute_cost:
                    one_group_cost = permute_cost
                    one_group_seq = list(permute)
                    one_group_at = veh_time
                    one_group_veh = permute_veh
                    one_group_loss = copy.deepcopy(permute_loss)
                    one_group_travle_distance = distance
                    one_group_travle_time = travel_time
                    each_order_depart_time = depart_time
                    each_order_wait_time = wait_time
                    each_order_arrive_time = arrival_time
                    if veh_time  > nd + db: # db: date boundary / 차량의 총 이동시간이 date boundary보다 크면 해당 group은 fail처리 / 단 cost는 유지
                        group_fail = ogrp_idx
                    else:
                        group_fail = -1

            if one_group_cost == float("inf"): # 선택할 수 있는 차량이 하나도 없었을 경우
                one_group_cost = 99999  # sum(one_brute_cost) 계산을 위해서 cost = 99999
                one_group_seq = copy.deepcopy(one_group) # fail 처리를 위해 group의 주문들 저장
                group_fail = ogrp_idx   # fail 처리 / group fail == -1일 경우 non-fail
                one_group_loss = [0,0,0]
            one_brute_cost.append(one_group_cost) 
            one_brute_variable_cost.append(one_group_cost)
            one_brute_seq.append(one_group_seq)
            one_brute_at.append(one_group_at)
            one_brute_veh.append(one_group_veh)
            one_brute_loss.append(one_group_loss)
            one_brute_fail.append(group_fail)
            one_brute_travle_time.append(one_group_travle_time)
            one_brute_travel_distance.append(one_group_travle_distance)
            brute_order_arrival_time.append(each_order_arrive_time)
            brute_order_depart_time.append(each_order_depart_time)
            brute_order_wait_time.append(each_order_wait_time)
        if sum(one_brute_cost) < sum(cost):
            cost = copy.deepcopy(one_brute_cost)
            veh_variable_cost= copy.deepcopy(one_brute_variable_cost)
            veh = copy.deepcopy(one_brute_veh)
            sequence = copy.deepcopy(one_brute_seq)
            arrive_time = copy.deepcopy(one_brute_at)
            veh_travel_distance = copy.deepcopy(one_brute_travel_distance)
            veh_travle_time = copy.deepcopy(one_brute_travle_time)
            final_wait_time = copy.deepcopy(brute_order_wait_time)
            final_arrival_time = copy.deepcopy(brute_order_arrival_time)
            final_depart_time = copy.deepcopy(brute_order_depart_time)
            loss = copy.deepcopy(one_brute_loss)
            fail_idx = copy.deepcopy(one_brute_fail)
    return cost, veh, sequence, arrive_time, final_arrival_time, final_wait_time, final_depart_time , veh_variable_cost , veh_travel_distance , veh_travle_time,loss, fail_idx

def make_bruteforce(orders):  # 주어진 order에서 가능한 경우의 수를 모두 만들어서 리턴해주는 함수
    answer = [[]]
    for g in orders:
        temp_answer = []
        for ans in answer:
            for j in range(len(ans)):
                temp_ans = copy.deepcopy(ans)
                s = 0
                for k in ans[j]:
                    s += k[4]
                if s + g[4] <= 55: # cbm 총합이 55를 초과할경우 트럭에 실을 수 없음
                    temp_ans[j].append(g)
                    temp_answer.append(temp_ans)

            if len(ans) < now_order_length:
                temp_ans = copy.deepcopy(ans)
                temp_ans.append([g])
                temp_answer.append(temp_ans)
        answer = copy.deepcopy(temp_answer)
    return answer

def make_nearest_point(tt, tp, dm, pti):
    near_p = []
    for point in tp:
        min_val = -1
        min_t = -1
        for t in tt:
            if min_val == -1 or min_val > dm[pti[point]][pti[t[0]]]:
                min_val = dm[pti[point]][pti[t[0]]]
                min_t = t[0]
        near_p.append(min_t)
    return near_p

start_time = time.time()

#전제 조건
cbm_of_veh = [27, 33, 42, 51, 55]
kind_of_veh = {'27': 0, '33': 1, '42': 2, '51': 3, '55': 4}
variable_cost = [0.8, 1, 1.2, 1.5, 1.8]
fixed_cost = [80, 110, 150, 200, 250]
total_day_num = 7
group_num = 4
max_ttl = 12    # 72시간 내로 처리해야됨 72시간 = 12배치
date_boundary = datetime.timedelta(hours=36)
last_day = '2023-05-07'

point_to_idx, total_point = make_point_to_idx()
total_point_num = len(point_to_idx)
distance_matrix, time_matrix = make_distance_time_matrix(point_to_idx, total_point_num)
total_terminal, terminal_to_idx = make_terminal_to_idx()
total_terminal_num = len(total_terminal) # 터미널 개수
each_veh_info, each_terminal_veh, total_veh_info, idx_to_veh, veh_results = make_veh_info(terminal_to_idx,
                                                                                          total_terminal_num)

order_date, order_group = make_order_info() # order table에서 주문 일자 / group 뽑아내기
order_date.append(last_day) # order table에는 없는 마지막 날짜 추가
nearest_terminal = make_nearest_terminal(total_terminal, total_terminal_num, distance_matrix, point_to_idx) # 자신의 터미널에서 자신 포함 모든 터미널 가까운 순으로 정렬
nearest_point = make_nearest_point(total_terminal, total_point,distance_matrix, point_to_idx)
total_order, orders_accurate_cbm = collecting_order(terminal_to_idx,total_terminal_num, total_day_num, group_num, max_ttl) # 모든 날짜와 그룹의 order들을 가져옴. (모든 터미널)
group_results = []  # 모든 주문 결과를 저장함.
after_results = []
final_veh_table =None
final_orders_table = None

sequence = 1
sum_of_min_cost = 0
fail_order = [[] for i in range(total_terminal_num)]

for date_idx, date in enumerate(order_date):  # 각 날짜마다
    print("\n", date)
    if date_idx == 0:
        tmp = date.split('-')
        now_date = datetime.datetime(int(tmp[0]), int(tmp[1]), int(tmp[2]), 0, 0, 0)  # 년,월,일,시,분,초 
    for group_idx in range(group_num):  # 각 배치마다
        print("Group", group_idx, "Now_date", now_date, sum_of_min_cost)
        
        for veh_vaild_time in total_veh_info:   # 각 차량의 최소 이용시간은 현재 시간보다는 길어야된다
            if veh_vaild_time[1] < now_date:
                veh_vaild_time[1] = copy.deepcopy(now_date)
        group_results = copy.deepcopy(after_results)

        terminal_order_len = []
        for i in range(total_terminal_num):
            terminal_order_len.append([i, len(fail_order[i]) + len(total_order[i][date_idx][group_idx])])
        terminal_order_len = sorted(terminal_order_len, key=lambda x : x[1])    #터미널을 주문량 적은 순서대로 정렬
        
        for termi_idx in range(total_terminal_num):  # 터미널 마다
            now_terminal = terminal_order_len[termi_idx][0]
            #print(total_terminal[now_terminal], len(total_order[now_terminal][date_idx][group_idx]), len(fail_order[now_terminal]))
            now_orders = copy.deepcopy(fail_order[now_terminal] + total_order[now_terminal][date_idx][group_idx]) #현재 터미널의 주문 : 이전에 실패했던 주문 + 새롭게 들어온 주문
            fail_order[now_terminal] = []   #이전에 실패했던 주문 초기화
            now_order_length = len(now_orders)
            boundary = 9    #brute force boundary
            preprocessed_order = []
            if now_order_length > boundary:  # 클러스터링
                terminal_site = [total_terminal[now_terminal][1], total_terminal[now_terminal][2]]
                model = CJ_Cluster(now_orders, distance_matrix, point_to_idx, terminal_site)
                clusters = model.clustered_orders
                for cluster in clusters:
                    if len(cluster) > 9:
                        for itera in range(int((len(cluster) - 1) / 9)):
                            preprocessed_order.append(cluster[itera*9:(itera+1)*9])
                        preprocessed_order.append(cluster[int((len(cluster) - 1) / 9) * 9:])
                    else:
                        preprocessed_order.append(cluster)
            else:
                preprocessed_order.append(now_orders)  # 현재 테이블의 주문들을 넣음 .

            for p_order in preprocessed_order:  # 해당 터미널의 배치 속 주문 들
                bruteforce_set = make_bruteforce(p_order)
                min_cost, min_veh, min_seq, min_arrive_time, arrival_time, waiting_time, departure_time, min_variable_cost, min_travle_distance, min_travle_time, min_loss, min_fail = cal_cost(
                    bruteforce_set, point_to_idx,
                    distance_matrix, time_matrix,
                    total_terminal[now_terminal][0], each_terminal_veh,
                    idx_to_veh, total_veh_info, nearest_terminal[now_terminal],
                    terminal_to_idx, now_date, date_boundary, last_day, nearest_point)
                sum_of_min_cost += sum(min_cost)
                
                #print(min_cost, min_veh, min_seq,min_arrive_time) #총 운영비용, [차량ID, 터미널ID] , 주문들(순서), 도착시, 각각의 len는 모두 같아야됨
                #print(arrival_time, waiting_time, departure_time)
                
                for j, order in enumerate(min_seq):
                    emergency = 1
                    if date != last_day and min_fail[j] != -1:
                        emergency = 0
                        for mso in min_seq[j]:
                            mso[11] -= 1        #주문 TTL -1
                            if mso[11] < 0:
                                emergency = 1   #TTL이 -가 되는 주문이 있으면 지금 만드시 처리해야됨
                        if emergency == 0:
                            sum_of_min_cost -= min_cost[j] # fail 처리 하므로 sum_of_min_cost에서 min cost 제외
                            for mso in min_seq[j]:
                                fail_order[now_terminal].append(mso) # fail처리 후 다음 배치에 넣기
                    
                    if emergency == 1:
                        for l, o in enumerate(order):  # 함께 묶인 group을 모두 넣기 위함.
                            # 주문번호 , 할당된 차량 ID , 할당된 배송 시퀀스 ,터미널ID , 대기시간, 상하/하차 시간 , 출발시각 , 배송완료여부
                            group_results.append(
                                [o[0], min_veh[j][0], sequence, min_veh[j][1], "Null", "Null", "Null", "Null",
                                "No"])  # 차량 배치 후
                            # 주번호 , 할당된 차량 ID , 할당된 배송 시퀀스 ,도착지ID , 대기시간, 상하/하차 시간 , 출발시각 , 배송완료여부
                            after_results.append(
                                [o[0], min_veh[j][0], sequence, o[3], arrival_time[j][l].strftime("%Y-%m-%d %H:%M"),
                                waiting_time[j][l], o[7], departure_time[j][l].strftime("%Y-%m-%d %H:%M"),
                                "Yes"])  # 배송 완료 후
                            sequence += 1
                            veh_results[idx_to_veh[min_veh[j][0]]][2] += orders_accurate_cbm[o[0]]  # 총 적재량
                            hour, minute,second =waiting_time[j][l].split(":")
                            if len(list(hour))>=3:
                                day , hour = hour.split(",")
                                veh_results[idx_to_veh[min_veh[j][0]]][7] += float(
                                    24*60*float(day[0])+60 * float(hour) + float(minute) + float(second) * 0.6)  # 총 대기시간
                            else: veh_results[idx_to_veh[min_veh[j][0]]][7] += float(60*float(hour)+float(minute)+float(second)*0.6) #총 대기시간
                            #print(after_results[-1])


                        veh_results[idx_to_veh[min_veh[j][0]]][1] += len(order)  # 배달한 주문수
                        veh_results[idx_to_veh[min_veh[j][0]]][6] += float(60*float(order[0][7])) #총 하역 시간
                        if total_veh_info[idx_to_veh[min_veh[j][0]]][2] == 0:
                            veh_results[idx_to_veh[min_veh[j][0]]][9] = each_veh_info[min_veh[j][2]][1] # 차량 고정비
                        veh_results[idx_to_veh[min_veh[j][0]]][10] += min_variable_cost[j] #거리 운영비
                        veh_results[idx_to_veh[min_veh[j][0]]][8] = veh_results[idx_to_veh[min_veh[j][0]]][10]+veh_results[idx_to_veh[min_veh[j][0]]][9] # 총 비용
                        veh_results[idx_to_veh[min_veh[j][0]]][3] += min_travle_distance[j] # 주행 거리
                        veh_results[idx_to_veh[min_veh[j][0]]][5] += min_travle_time[j] #이동 시간
                        # 작업시간(대기시간+이동시간+하역시간)
                        veh_results[idx_to_veh[min_veh[j][0]]][4] =\
                            veh_results[idx_to_veh[min_veh[j][0]]][5]+veh_results[idx_to_veh[min_veh[j][0]]][6]+veh_results[idx_to_veh[min_veh[j][0]]][7]
                        each_terminal_veh[terminal_to_idx[min_veh[j][1]]][min_veh[j][2]].remove(min_veh[j][0])  # 없어진 차량은 제거
                        each_terminal_veh[terminal_to_idx[nearest_point[point_to_idx[min_seq[j][-1][3]]]]][min_veh[j][2]].insert(0, min_veh[j][0])  # 유입된 차량 추가
                        total_veh_info[idx_to_veh[min_veh[j][0]]][0] = nearest_point[point_to_idx[min_seq[j][-1][3]]]  #차량이 있는 위치 변경
                        total_veh_info[idx_to_veh[min_veh[j][0]]][1] = min_arrive_time[j]   #차량 이용가능시작시간 변경
                        total_veh_info[idx_to_veh[min_veh[j][0]]][2] = 1  # 이 차량 사용했음을 저장 
        final_orders_table = pd.DataFrame(group_results,
                                    columns=["ORD_NO", "VehicleID", "Sequence", "SiteCode", "ArrivalTime",
                                             "WaitingTime", "ServiceTime", "DepartureTime", "Delivered"])
        #print(final_orders_table.set_index('ORD_NO'))
        now_date += datetime.timedelta(hours=6)
final_veh_table = pd.DataFrame(veh_results,
                            columns=["VehicleID", "Count", "Volume", "TravelDistance", "WorkTime", "TravleTime",
                                     "ServiceTime", "WaitingTime", "TotalCost", "FixedCost",
                                     "VariableCost"])
#print(final_veh_table.set_index('VehicleID'))

print("Total Cost : ",sum_of_min_cost)  # 총 비용 / 현재 3,608,869 / 3,000,000 이하로 줄여야됨
print("Total Time : ",time.time() - start_time) # 전체 돌아가는데 걸리는 시간 / 현재 : 1195s
#final_veh_table.to_excel(excel_writer='final_veh_table.xlsx')
#final_orders_table.to_excel(excel_writer='final_orders_table.xlsx' )