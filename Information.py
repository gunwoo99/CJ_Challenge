import numpy as np
import csv, copy, datetime
import config

class Information:
    def __init__(self, od_matrix_file, order_table_file, terminal_file, vehicle_file, TTL):
        self._make_vertex_to_index(terminal_file_name=terminal_file, order_table_file_name=order_table_file)
        self.vertex_num = len(self.index_to_vertex)
        self._make_distance_time_matrix(od_matrix_file_name=od_matrix_file)
        self.terminal_num = len(self.index_to_terminal)
        self.destination_num = self.vertex_num - self.terminal_num
        self._processing_order(order_table_file_name=order_table_file, TTL=TTL)
        self._make_nearest_terminal()
        self._make_vehicle_information(vehicle_file_name=vehicle_file)
    
    def _make_vertex_to_index(self, terminal_file_name, order_table_file_name):
        terminal_file = open(terminal_file_name,    'r',encoding=config.ENCODING_TYPE)
        order_file    = open(order_table_file_name, 'r',encoding=config.ENCODING_TYPE)
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
        
        total_vertex   = total_vertex + total_terminal
        total_vertex   = sorted(total_vertex,   key=lambda x: (x[0], x[2:]))
        total_terminal = sorted(total_terminal, key=lambda x: (x[0], x[2:]))
        
        total_index_of_vertex   = {}
        total_index_of_terminal = {}
        for i, vertex in enumerate(total_vertex):
            total_index_of_vertex[vertex] = i
        for i, terminal in enumerate(total_terminal):
            total_index_of_terminal[terminal] = i
        
        terminal_file.close()
        order_file.close()
        self.terminal_info     = terminal_info
        self.index_to_vertex   = total_vertex
        self.index_to_terminal = total_terminal
        self.vertex_to_index   = total_index_of_vertex
        self.terminal_to_index = total_index_of_terminal
        print("complete make vertex to index")
    
    def _make_distance_time_matrix(self, od_matrix_file_name):
        od_matrix_file   = open(od_matrix_file_name, 'r')
        od_matrix_reader = csv.reader(od_matrix_file)
        
        distance_matrix = np.zeros((self.vertex_num, self.vertex_num)).tolist()
        time_matrix     = np.zeros((self.vertex_num, self.vertex_num)).tolist()
        next(od_matrix_reader)
        for od in od_matrix_reader:
            distance_matrix[self.vertex_to_index[od[0]]][self.vertex_to_index[od[1]]] = float(od[2])
            time_matrix    [self.vertex_to_index[od[0]]][self.vertex_to_index[od[1]]] = float(od[3])
        
        self.distance_matrix = distance_matrix
        self.time_matrix     = time_matrix
        od_matrix_file.close()
        print("complete make distance time matrix")
    
    def _processing_order(self, order_table_file_name, TTL):
        order_file   = open(order_table_file_name, 'r', encoding=config.ENCODING_TYPE)
        order_reader = csv.reader(order_file) 
        
        total_orders = [[[] for __ in range(self.terminal_num)] for _ in range(config.TOTAL_DAY * config.BATCH_COUNT_PER_DAY)]
        next(order_reader)
        for order in order_reader:
            order[1] = float(order[1])
            order[2] = float(order[2])
            order[4] = float(order[4])
            order[5] = datetime.timedelta(hours=int(order[5].split(":")[0]), minutes=int(order[5].split(":")[1]))
            order[6] = datetime.timedelta(hours=int(order[6].split(":")[0]), minutes=int(order[6].split(":")[1]))
            if (order[6] - order[5]).days == - 1:
                order[6] += datetime.timedelta(days=1)
            order.append(TTL)
            total_orders[(int(order[9][-2:]) - 1) * config.BATCH_COUNT_PER_DAY + int(order[10])][self.terminal_to_index[order[8]]].append(order)
        
        self.total_orders = total_orders
        order_file.close()
        print("complete collecting_order")
    
    def _make_nearest_terminal(self):
        nearest_terminal_from_D = {}
        for i in range(self.destination_num):
            row = copy.deepcopy(self.distance_matrix[i][self.destination_num:])
            nearest_terminal_from_D[self.index_to_vertex[i]] = self.index_to_terminal[np.argmin(row)]
        self.nearest_termnial_from_D = nearest_terminal_from_D
        
        self.nearest_terminal_from_O = []
        for i in range(self.terminal_num):
            terminal = []
            for j in range(self.terminal_num):
                terminal.append([self.index_to_terminal[j], self.distance_matrix[self.destination_num + i][self.destination_num + j]])
            terminal = sorted(terminal, key=lambda x: x[1])
            terminal = [terminal[j][0] for j in range(self.terminal_num)]
            self.nearest_terminal_from_O.append(terminal)
    
    def _make_vehicle_information(self, vehicle_file_name):
        vehicle_file   = open(vehicle_file_name, 'r', encoding=config.ENCODING_TYPE)
        vehicle_reader = csv.reader(vehicle_file)
        
        next(vehicle_reader)
        start_time            = datetime.datetime(config.YEAR, config.MONTH, config.START_DAY)
        self.vehicle_to_index = {}
        self.cbm_to_index     = {}
        self.cbm              = []
        self.fixed_cost       = []
        self.variable_cost    = []
        
        # vehicle 종류 setting
        infos = []
        for vehicle in vehicle_reader:
            if int(vehicle[4]) not in self.cbm:
                self.cbm          .append(int(vehicle[4]))
                self.fixed_cost   .append(int(vehicle[6]))
                self.variable_cost.append(float(vehicle[7]))
        
        self.cbm          .sort()
        self.fixed_cost   .sort()
        self.variable_cost.sort()
        for i in range(len(self.cbm)):
            self.cbm_to_index[self.cbm[i]] = i
        # vehicle info 
        self.total_vehicle_info = []
        self.terminal_vehicle   = [[[] for __ in range(len(self.cbm))] for _ in range(self.terminal_num)]
        
        vehicle_file.close()
        vehicle_file   = open(vehicle_file_name, 'r', encoding=config.ENCODING_TYPE)
        vehicle_reader = csv.reader(vehicle_file)
        next(vehicle_reader)
        for index, vehicle in enumerate(vehicle_reader):
            self.vehicle_to_index[vehicle[0]] = index
            vehicle_info                      = [vehicle[0], start_time, 0, self.cbm_to_index[int(vehicle[4])]]
            self.total_vehicle_info.append(vehicle_info)
            self.terminal_vehicle[self.terminal_to_index[vehicle[5]]][self.cbm_to_index[int(vehicle[4])]].append(vehicle[0])
        
        vehicle_file.close()
        print("complete make vehicle information")