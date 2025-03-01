import numpy as np
import copy 

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def euclidean_distance(x1, x2):
    return np.sum((x1 - x2) ** 2)

class MinDistanceGrouping():
    def __init__(self, distance_matrix, site_indices, index_site_number, max_group_size=10):
        '''
        가장 짧은 거리 기준으로 클러스터링.
        주어진 date와 batch에서 클러스터링하고, max_group_size(default:10)이하로 묶이도록 함. 
        '''
        self.grouping_result = self._grouping(max_group_size, site_indices, distance_matrix, index_site_number) # grouping 결과가 idx로 묶여있음.
    
    def _union(self, groups, groups_size, group_ids, is_deleted, i, j, index_site_number, size=10):
        if group_ids[i] == -1: # i가 그룹x
            if group_ids[j] == -1: # j가 그룹x => i, j 새그룹
                groups.append([i, j])
                groups_size.append(index_site_number[i] + index_site_number[j])
                is_deleted.append(False)
                group_ids[i] = group_ids[j] = len(groups) - 1
            else: # j는 그룹
                if index_site_number[i] + groups_size[group_ids[j]] > size: # i를 새 그룹으로!
                    groups.append([i])
                    groups_size.append(index_site_number[i])
                    is_deleted.append(False)
                    group_ids[i] = len(groups)-1
                else :
                    groups[group_ids[j]].append(i) # j에 i 추가
                    groups_size[group_ids[j]] += index_site_number[i]
                    group_ids[i] = group_ids[j]
        else: # i가 그룹
            if group_ids[j] == -1: # j가 그룹아님
                if groups_size[group_ids[i]] + index_site_number[j] > size: # j를 새 그룹으로!
                    groups.append([j])
                    groups_size.append(index_site_number[j])
                    is_deleted.append(False)
                    group_ids[j] = len(groups)-1
                else:
                    groups[group_ids[i]].append(j) # i에 j 추가
                    groups_size[group_ids[i]] += index_site_number[j]
                    group_ids[j] = group_ids[i]
            else: # j도 그룹
                if group_ids[i]==group_ids[j]:return # 합칠필요 x
                if groups_size[group_ids[i]] + groups_size[group_ids[j]] <= size:
                    if groups_size[group_ids[i]] >= groups_size[group_ids[j]]: # i에 j 합치기
                        is_deleted[group_ids[j]] = True # j 삭제
                        for ji in groups[group_ids[j]]:
                            group_ids[ji] = group_ids[i]
                            groups[group_ids[i]].append(ji)
                        groups_size[group_ids[i]] += groups_size[group_ids[j]]
                    else: # j에 i 합치기
                        is_deleted[group_ids[i]] = True # i 삭제
                        for ii in groups[group_ids[i]]:
                            group_ids[ii] = group_ids[j]
                            groups[group_ids[j]].append(ii)
                        groups_size[group_ids[j]] += groups_size[group_ids[i]]

    def _grouping(self, max_group_size, site_indices, distance_matrix, index_site_number):
        groups      = [] # 결과 배열
        groups_size = []
        is_deleted  = [] # for soft delete
        group_ids   = [-1] * len(index_site_number) # 어떤 그룹에 속해있는지 나타냄.
        
        # 1. 가까운 순으로 정렬
        processed_distance_list = []
        for i in range(len(site_indices)):
            for j in range(len(site_indices)):
                if i == j:
                    continue
                processed_distance_list.append((distance_matrix[site_indices[i]][site_indices[j]], i, j))
        processed_distance_list.sort()
        
        # 2. 가장 가까운 것부터 빼서 그룹에 넣어줌.
        for _, i, j in processed_distance_list:
            self._union(groups, groups_size, group_ids, is_deleted, i, j, index_site_number, max_group_size)
        return list(map(lambda x: x[1], filter(lambda x: not is_deleted[x[0]], enumerate(groups))))


class CJ_Cluster:
    def __init__(self, order_list, distance_matrix, vertex_to_index, terminal_site):
        self.assigend_cluster     = []
        self.order_number         = len(order_list)
        orders_site               = []
        orders_destination        = []
        same_site_orders_indices  = []
        
        
        # 같은 위치 Order끼리 묶기
        for i in range(self.order_number):
            # order의 위치를 list로 받기
            order = [order_list[i][1], order_list[i][2]]
            if order not in orders_site:
                orders_site.append(order)
                orders_destination.append(order_list[i][3])
                same_site_orders_indices.append([i])
            else:
                index = orders_site.index(order)
                same_site_orders_indices[index].append(i)
        
        # 같은위치인 order가 X개 이상인 위치들은 cluster로 추가
        # Order_site에서도 빼주기
        plus_clustered_orders = []
        site_num = len(orders_site) - 1
        for i in range(site_num, -1, -1):
            if len(same_site_orders_indices[i]) > 5:
                exceeded_site_orders = []
                for j in range(len(same_site_orders_indices[i])):
                    exceeded_site_orders.append(order_list[same_site_orders_indices[i][j]])
                plus_clustered_orders.append(exceeded_site_orders)
                
                del same_site_orders_indices[i]
                del orders_destination[i]
                del orders_site[i]
        
        # KMeans cluster 갯수 정하고 KMeans
        self.site_number      = len(orders_site)
        orders_site_ndarray   = np.array(orders_site)
        self.cluster_number   = int((self.site_number + 10) / 10)
        
        # cluster가 2개 이상이면 KMeans 진행 아니면 전체를 하나의 클러스터로 묶어주기 
        if self.cluster_number >= 2:
            self.clustered_order_site_indices_set = [[] for _ in range(self.cluster_number)]
            self.fit(orders=orders_site_ndarray)
            
            for i in range(self.site_number):
                self.clustered_order_site_indices_set[self.assigend_cluster[i]].append(i)
        # cluster가 1개이면 KMeans를 진행하지 않고 하나의 클러스터라고 생각한다
        else:
            self.cluster_number = 1
            self.clustered_order_site_indices_set = [[]]
            for i in range(self.site_number):
                self.clustered_order_site_indices_set[0].append(i)
        # K Means 종료
        
        # Clustering이 완료된 각각의 cluster를 hierarchical clustering함
        # cluster에 주문 갯수를 10개로 제한 할 수 있도록 함
        self.clustered_orders = []
        self.assigned_cluster_site_num = []
        for i in range(self.cluster_number):
            # 각 site들이 distance matrix에서 몇번째 인지를 list로
            site_indices = [vertex_to_index[orders_destination[self.clustered_order_site_indices_set[i][j]]] for j in range(len(self.clustered_order_site_indices_set[i]))]
            # 각각의 site들이 몇개의 주문을 가지고 있는지 list로
            index_site_number = [len(same_site_orders_indices[self.clustered_order_site_indices_set[i][j]]) for j in range(len(self.clustered_order_site_indices_set[i]))]
            
            hierarchical_clustering = MinDistanceGrouping(distance_matrix=distance_matrix, site_indices=site_indices, index_site_number=index_site_number)
            result = hierarchical_clustering.grouping_result
            
            for j in range(len(result)):
                cluster = []
                cluster_average = np.zeros((2))
                for k in range(len(result[j])):
                    for l in range(len(same_site_orders_indices[self.clustered_order_site_indices_set[i][result[j][k]]])):
                        cluster.append(order_list[same_site_orders_indices[self.clustered_order_site_indices_set[i][result[j][k]]][l]])
                        site = np.array([cluster[-1][1], cluster[-1][2]])
                        cluster_average += site
                cluster_average /= len(cluster)
                cluster_distance_from_terminal = euclidean_distance(np.array(terminal_site), cluster_average)
                cluster = [cluster, cluster_distance_from_terminal, len(result[j])]
                
                # cluster = [[assigned_order_list], [cluster_average_distance_from_terminal], site_num]
                self.clustered_orders.append(cluster)
        
        # 한 착지에 너무 많은 주문이 몰려서 clustering에서 제외됐던 site들은 하나의 클러터로 전체 클러스터링 결과에 합쳐짐
        for i in range(len(plus_clustered_orders)):
            plus_cluster_site = np.array([plus_clustered_orders[i][0][1], plus_clustered_orders[i][0][2]])
            cluster = [plus_clustered_orders[i], euclidean_distance(np.array(terminal_site), plus_cluster_site), 1]            
            self.clustered_orders.append(cluster)
        
        # terminal에서 가까운 cluster부터 반환 하도록 함
        self.clustered_orders          = sorted(self.clustered_orders, key=lambda x:x[1])
        self.assigned_cluster_site_num = list(map(lambda x:x[2], self.clustered_orders))
        self.distance                  = list(map(lambda x:x[1], self.clustered_orders))
        self.clustered_orders          = list(map(lambda x:x[0], self.clustered_orders))
        self.cluster_number            = len(self.clustered_orders)
        
    # KMeans 실행부분
    def fit(self, orders):
        # max_iteration, centroid initialize
        max_iteration = 50
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
            
            if [0, 0] in new_centroids:
                indices = np.where(np.all(new_centroids == [0, 0], axis=1))[0]
                new_centroids = np.delete(new_centroids, indices.tolist(), axis=0)
                self.cluster_number -= len(indices)
            
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
        if 0 in cluster_counts:
            indices = np.where(cluster_counts == 0)[0]
            for index in indices:
                new_centroids[index] = np.zeros(2)
        return new_centroids