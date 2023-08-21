# file encoding
ENCODING_TYPE    = "cp949"

# file name
OD_MATRIX_FILE   = "od_matrix.csv"
ORDER_TABLE_FILE = "orders_table.csv"
VEHICLE_FILE     = "veh_table.csv"
TERMINAL_FILE    = "terminals.csv"
# OD_MATRIX_FILE   = "File_OD_Matrix.csv"
# ORDER_TABLE_FILE = "File_Order.csv"
# VEHICLE_FILE     = "File_Vehicle.csv"
# TERMINAL_FILE    = "File_Terminal.csv"

# 알고리즘 관련 설정
CLUSTER_COUNT_BOUNDARY = 9
TTL                    = 11
BURST_CALL_BOUNDARY    = 100
WAITING_TIME_BOUNDARY  = 6

# 입력받은 데이터에 관한 설정
### 일자
YEAR                = 2023
MONTH               = 5
START_DAY           = 1
END_DAY             = 7
TOTAL_DAY           = END_DAY-START_DAY+1
### 배치
BATCH_COUNT_PER_DAY = 4
BATCH_TIME_HOUR     = 6
# TOTAL_BATCH         = BATCH_COUNT_PER_DAY*TOTAL_DAY
### 기타
CBM_BOUNDARY        = 0.04
WORKING_TIME_MINITE = 60 # 작업(하차)시간 (분)