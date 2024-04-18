from enum import Enum

# 플레이어 동선 그래프 복잡도 - 도달 가능 위치(정점) 결정 수
COMPLEXITY = 20

# 결정한 정점 주변 다른 정점 생성 방지 구역 크기
POS_MASK_SIZE = 7

# 레벨 입구 생성 Y 좌표
ENTRY_POS = 1

# 플레이어 동선 그래프 상에서 정점 사이 최대 거리
NODE_DIST_MAX = 10

# 두 정점 간의 구조물 생성 시 정점 좌표로 결정되는 직육면체 패딩 범위
CUBOID_PADDING = 2

# 플레이어 동선 그래프 상에서 두 정점의 최대 높이 차
EDGE_MAX_DY = 6

# 플레이어 동선 그래프 상에서 간선의 최대 경사
EDGE_MAX_SLOPE = 0.5

# WFC 알고리즘 휴리스틱 설정
class WFCHeuristic(Enum):
    Entropy = 0
    Scanline = 1