# 생성할 레벨의 차원 크기
SIZE_X = 27
SIZE_Y = 10
SIZE_Z = 27

# 패턴 크기
MATRIX_X = 3
MATRIX_Y = 3
MATRIX_Z = 3

EPOCH = 1
EPSILON = 1e-6
HASH_MODULO = 100000007

OFF_X = MATRIX_X // 2
OFF_Y = MATRIX_Y // 2
OFF_Z = MATRIX_Z // 2

# WFC 알고리즘 패턴 필터 값
FILTER_AIR              = -2
FILTER_PLATFORM         = -3
FILTER_ADJACENT_WALL    = -4
FILTER_ADJACENT_FLOOR   = -5

# 파일 입출력 경로
INPUT_DIR = './Input/'
OUTPUT_DIR = './Output/'