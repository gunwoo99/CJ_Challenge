import numpy as np
import csv, copy

class Information:
    def __init__(self, od_matrix_file, order_table_file, terminal_file, vehicle_file):
        self._make_vertex_to_index(terminal_file_name=terminal_file, order_table_file_name=order_table_file)
        self.total_vertex_num = len(self.total_vertex)
        self._make_distance_time_matrix(od_matrix_file_name=od_matrix_file)
        self.total_terminal_num = len(self.total_terminal)
        self._processing_order(order_table_file_name=order_table_file)
        self._make_nearest_terminal_from_D()
    
    def _make_vertex_to_index(self, terminal_file_name, order_table_file_name):
        terminal_file = open(terminal_file_name, 'r',encoding='cp949')
        order_file    = open(order_table_file_name, 'r',encoding='cp949')
        terminal_reader = csv.reader(terminal_file)
        order_reader    = csv.reader(order_file)
        next(terminal_reader)
        next(order_reader)
        
        total_vertex   = []
        total_terminal = []
        terminal_info  = []
        for line in order_reader:
            if line[3] not in total_vertex:
                total_vertex.append(line[3])
        for line in terminal_reader:
            if line[0] not in total_terminal:
                terminal_info.append(line)
                terminal_info[-1][1] = float(terminal_info[-1][1])
                terminal_info[-1][2] = float(terminal_info[-1][2])
                total_terminal.append(line[0])
        
        total_vertex = total_vertex + total_terminal
        total_vertex = sorted(total_vertex, key=lambda x: (x[0], x[2:]))
        
        total_index_of_vertex   = {}
        total_index_of_terminal = {}
        for i, vertex in enumerate(total_vertex):
            total_index_of_vertex[vertex] = i
        for i, terminal in enumerate(total_terminal):
            total_index_of_terminal[terminal] = i
        
        terminal_file.close()
        order_file.close()
        self.terminal_info           = terminal_info
        self.total_vertex            = total_vertex
        self.total_terminal          = total_terminal
        self.total_index_of_vertex   = total_index_of_vertex
        self.total_index_of_terminal = total_index_of_terminal
        print("complete make vertex to index")
    
    def _make_distance_time_matrix(self, od_matrix_file_name):
        od_matrix_file   = open(od_matrix_file_name, 'r')
        od_matrix_reader = csv.reader(od_matrix_file)
        
        distance_matrix = np.zeros((self.total_vertex_num, self.total_vertex_num)).tolist()
        time_matrix     = np.zeros((self.total_vertex_num, self.total_vertex_num)).tolist()
        next(od_matrix_reader)
        for od in od_matrix_reader:
            distance_matrix[self.total_index_of_vertex[od[0]]][self.total_index_of_vertex[od[1]]] = float(od[2])
            time_matrix[self.total_index_of_vertex[od[0]]][self.total_index_of_vertex[od[1]]]     = float(od[3])
        
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        od_matrix_file.close()
        print("complete make distance time matrix")
    
    def _processing_order(self, order_table_file_name):
        order_file   = open(order_table_file_name, 'r',encoding='cp949')
        order_reader = csv.reader(order_file)
        
        total_orders = [[[] for __ in range(self.total_terminal_num)] for _ in range(6 * 4)]
        next(order_reader)
        for order in order_reader:
            order[1] = float(order[1])
            order[2] = float(order[2])
            total_orders[(int(order[9][-2:]) - 1) * 4 + int(order[10])][self.total_index_of_terminal[order[8]]].append(order)
        
        self.total_orders = total_orders
        order_file.close()
        print("complete collecting_order")
    
    def _make_nearest_terminal_from_D(self):
        nearest_terminal_from_D = {}
        total_destination_number = self.total_vertex_num - self.total_terminal_num
        for i in range(total_destination_number):
            row = self.distance_matrix[i].copy()[total_destination_number:]
            nearest_terminal_from_D[self.total_vertex[i]] = self.total_terminal[np.argmin(row)]
        self.nearest_termnial_from_D = nearest_terminal_from_D
