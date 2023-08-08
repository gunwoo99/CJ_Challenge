import numpy as np
import copy

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))



class MinDistanceGrouping():
    def __init__(self, distance_matrix, max_group_size=10):
        '''
        가장 짧은 거리 기준으로 클러스터링.
        주어진 date와 batch에서 클러스터링하고, max_group_size(default:10)이하로 묶이도록 함. 
        '''
        
        self.orders_to_idx = self.__make_orders_to_idx_from_file(date, batch)
        self.total_orders_num = len(self.orders_to_idx)
        distance_matrix, time_matrix = self.__make_distance_time_matrix(self.orders_to_idx, self.total_orders_num) # fully 그래프를 만들어서 반환해줌 (이차원 배열)
        self.cost_matrix = time_matrix
        self.grouping_result = self._grouping(max_group_size) # grouping 결과가 idx로 묶여있음.
    
    def _union(self, groups, group_ids, is_deleted, i, j, size=10):
        if group_ids[i] == -1: # i가 그룹x
            if group_ids[j] == -1: # j가 그룹x => i, j 새그룹
                groups.append([i, j])
                is_deleted.append(False)
                group_ids[i] = group_ids[j] = len(groups) - 1
            else: # j는 그룹
                if len(groups[group_ids[j]]) >= size: # i를 새 그룹으로!
                    groups.append([i])
                    is_deleted.append(False)
                    group_ids[i] = len(groups)-1
                else :
                    groups[group_ids[j]].append(i) # j에 i 추가
                    group_ids[i] = group_ids[j]
        else: # i가 그룹
            if group_ids[j] == -1: # j가 그룹아님
                if len(groups[group_ids[i]]) >= size: # j를 새 그룹으로!
                    groups.append([j])
                    is_deleted.append(False)
                    group_ids[j] = len(groups)-1
                else :
                    groups[group_ids[i]].append(j) # i에 j 추가
                    group_ids[j] = group_ids[i]
            else: # j도 그룹
                if group_ids[i]==group_ids[j]:return # 합칠필요 x
                if len(groups[group_ids[i]])+len(groups[group_ids[j]]) <= size:
                    if len(groups[group_ids[i]])>=len(groups[group_ids[j]]): # i에 j 합치기
                        is_deleted[group_ids[j]] = True # j 삭제
                        for ji in groups[group_ids[j]]:
                            group_ids[ji] = group_ids[i]
                            groups[group_ids[i]].append(ji)
                    else: # j에 i 합치기
                        is_deleted[group_ids[i]] = True # i 삭제
                        for ii in groups[group_ids[i]]:
                            group_ids[ii] = group_ids[j]
                            groups[group_ids[j]].append(ii)

    def _grouping(self, max_group_size):
        '''
        dm matrix를 기반으로 max_group_size 만큼 그룹핑
        '''
        groups = [] # 결과 배열
        is_deleted = [] # for soft delete
        group_ids = [-1]*self.total_orders_num # 어떤 그룹에 속해있는지 나타냄.
        distance_list = []
        # 1. dm을 list로 저장해서 정렬
        for i in range(self.total_orders_num):
            for j in range(self.total_orders_num):
                if i==j: continue
                distance_list.append((self.cost_matrix[i][j], i, j))
        distance_list.sort() # 최단 거리 기준으로 정렬

        # 2. 가장 가까운 것부터 빼서 그룹에 넣어줌.
        for _, i, j in distance_list:
            self._union(groups, group_ids, is_deleted, i, j, max_group_size)
        return list(map(lambda x: x[1], filter(lambda x: not is_deleted[x[0]], enumerate(groups))))


class CJ_Cluster:
    def __init__(self, order_list, distance_matrix, vertex_to_index):
        self.assigend_cluster    = []
        self.order_number        = len(order_list)
        orders_site              = []
        same_site_group          = []
        same_site_number         = {}
        
        # 같은 위치 Order끼리 묶기
        for i in range(self.order_number):
            order = [order_list[i][1], order_list[i][2]]
            if order_list[i][8] not in same_site_number:
                orders_site.append(order)
                same_site_group.append([i])
                same_site_number[order_list[i][8]] = 1
            else:
                index = orders_site.index(order)
                same_site_group[index].append(i)
                same_site_number[order_list[i][8]] += 1
        
        # 같은위치인 order가 X개 이상인 위치들은 cluster로 추가
        # Order_site에서도 빼주기
        plus_clustered_orders = []
        iterate = len(same_site_group) - 1
        for i in range(iterate, -1, -1):
            if len(same_site_group[i]) > 5:
                plus_clustered_orders.append([])
                for j in range(len(same_site_group[i])):
                    plus_clustered_orders[-1].append(order_list[same_site_group[i][j]])
                del same_site_group[i]
                del orders_site[i]
        
        # KMeans cluster 갯수 정하고 KMeans
        self.site_number      = len(orders_site)
        orders_site_ndarray   = np.array(orders_site)
        self.cluster_number   = int((self.site_number + 0) / 15)
        
        # cluster가 2개 이상이면 KMeans 진행 아니면 전체를 하나의 클러스터로 묶어주기 
        if self.cluster_number > 1:
            self.clustered_orders         = [[] for _ in range(self.cluster_number)]
            self.clustered_order_site_set = [[] for _ in range(self.cluster_number)]
            self.fit(orders=orders_site_ndarray)
            
            self.assigend_cluster_site_number = np.zeros(self.cluster_number).tolist()
            for i in range(self.site_number):
                self.assigend_cluster_site_number[self.assigend_cluster[i]] += 1
                self.clustered_order_site_set[self.assigend_cluster[i]].append(order_list[same_site_group[i][0]][8])
                for j in range(len(same_site_group[i])):
                    self.clustered_orders[self.assigend_cluster[i]].append(order_list[same_site_group[i][j]])
            
        else:
            self.clustered_orders = [[]]
            self.clustered_order_site_set = [[]]
            self.assigend_cluster_site_number = [self.site_number]
            for i in range(self.site_number):
                self.clustered_order_site_set[0].append(order_list[same_site_group[i][0]][8])
                for j in range(len(same_site_group[i])):
                    self.clustered_orders[0].append(order_list[same_site_group[i][j]])
        
        for i in range(self.cluster_number):
            
            
            site_indices = [vertex_to_index[self.clustered_order_site_set[i][8]] for i in range(len(self.clustered_order_site_set[i]))]
            self.clustered_orders[i]
            MinDistanceGrouping(distance_matrix, site_indices)
        
        # self.cluster_number = len(self.clustered_orders)
        # for i in range(self.cluster_number-1, -1, -1):
        #     if len(self.clustered_orders[i]) > 9 and self.assigend_cluster_site_number[i] != 1:
        #         sub_kmeans = KMeans(self.clustered_orders[i])
        #         del self.clustered_orders[i]
        #         del self.assigend_cluster_site_number[i]
                
        #         self.clustered_orders = copy.deepcopy(self.clustered_orders) + copy.deepcopy(sub_kmeans.clustered_orders)
        #         self.assigend_cluster_site_number = copy.deepcopy(self.assigend_cluster_site_number) + copy.deepcopy(sub_kmeans.assigend_cluster_site_number)
        # self.cluster_number = len(self.clustered_orders)
        # self.site_number = real_site_num
    
    # KMeans 실행부분
    def fit(self, orders):
        # max_iteration, centroid initialize
        max_iteration = 20
        initialize_centroid_indices = np.random.choice(self.site_number, self.cluster_number, replace=False)
        self.centroids = orders[initialize_centroid_indices]
        
        # max_iteration만큼 K_Means algorithm진행
        for _ in range(max_iteration):
            # order마다 가장 가까운 centroid로 클러스터 배정
            self._assign_order_to_cluster(orders=orders)
            
            # cluster들의 centroid 다시계산
            new_centroids = self._calculate_centroids(orders=orders)
            
            # 이전과 비교해서 차이가 적으면 break
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
    
    def _assign_order_to_cluster(self, orders):
        self.assigend_cluster.clear()
        for order in orders:
            distances = [euclidean_distance(order, centroid) for centroid in self.centroids]
            self.assigend_cluster.append(np.argmin(distances))
    
    def _calculate_centroids(self, orders):
        new_centroids  = np.zeros((self.cluster_number, 2))
        cluster_counts = np.zeros(self.cluster_number)
        for order, cluster in zip(orders, self.assigend_cluster):
            new_centroids[cluster]  += order
            cluster_counts[cluster] += 1
        
        new_centroids = new_centroids / cluster_counts[:, np.newaxis]
        return new_centroids