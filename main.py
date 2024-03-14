import anvil, time, random
import wfcModel, mcaFileIO
from constantTable import SIZE_X, SIZE_Y, SIZE_Z, MATRIX_X, MATRIX_Y, MATRIX_Z, INPUT_DIR, OUTPUT_DIR
from heuristicConfig import COMPLEXITY, ENTRY_POS, POS_MASK_SIZE, NODE_DIST_MAX, CUBOID_PADDING
from wfcModel import wfcModel, Node, Pattern

inputWorld = anvil.Region.from_file(INPUT_DIR + 'r.0.0.mca')
inputCache = []

nodeList = []
edgeSet = set()
movementGraph = []

dx_3x3 = [0, 1, 1, 1, 0, -1, -1, -1]
dz_3x3 = [-1, -1, 0, 1, 1, 1, 0, -1]

DIM_X = SIZE_X
DIM_Y = SIZE_Y
DIM_Z = SIZE_Z

BLOCK_REGISTRY = {}
BLOCK_REGISTRY_INV = []
PATTERNS = []

initLevel = None

def mcaInitializer():
    """
        월드 파일을 읽고 블록 레지스트리 등록, 패턴 추출 수행
    """
    global DIM_X, DIM_Y, DIM_Z, inputWorld, inputCache, BLOCK_REGISTRY, BLOCK_REGISTRY_INV, PATTERNS
    timestamp = time.time()
    DIM_Y = mcaFileIO.getMaxY(inputWorld)
    inputCache = [[[-1 for _ in range(DIM_Z)] for _ in range(DIM_Y)] for _ in range(DIM_X)]
    BLOCK_REGISTRY, BLOCK_REGISTRY_INV = mcaFileIO.registerBlocks(DIM_X, DIM_Y, DIM_Z, inputWorld, inputCache, timestamp)
    PATTERNS = mcaFileIO.extractPatterns(DIM_X, DIM_Y, DIM_Z, BLOCK_REGISTRY, inputWorld, inputCache, timestamp)

def levelInitializer():
    global DIM_X, DIM_Y, DIM_Z, initLevel, BLOCK_REGISTRY
    initLevel = [[[-1 for _ in range(DIM_Z)] for _ in range(DIM_Y)] for _ in range(DIM_X)]
    for x in range(DIM_X):
        for y in range(DIM_Y):
            for z in range(DIM_Z):
                if y == 0:
                    if x == 0 or x == DIM_X-1 or z == 0 or z == DIM_Z-1:
                        initLevel[x][y][z] = BLOCK_REGISTRY['bedrock']
                    else:
                        initLevel[x][y][z] = BLOCK_REGISTRY['black_wool']
                elif y >= ENTRY_POS and y < ENTRY_POS + 5:
                    if x >= 12 and x <= 14 and (z == 0 or z == DIM_Z-1):
                        initLevel[x][y][z] = BLOCK_REGISTRY['air']
                    elif x >= 11 and x <= 15 and ((z >= 1 and z <= 2) or (z <= DIM_Z-2 and z >= DIM_Z-3)):
                        initLevel[x][y][z] = BLOCK_REGISTRY['air']
                    elif z >= 12 and z <= 14 and (x == 0 or x == DIM_X-1):
                        initLevel[x][y][z] = BLOCK_REGISTRY['air']
                    elif z >= 11 and z <= 15 and ((x >= 1 and x <= 2) or (x <= DIM_X-2 and x >= DIM_X-3)):
                        initLevel[x][y][z] = BLOCK_REGISTRY['air']
                    elif x == 0 or x == 26 or z == 0 or z == DIM_Z-1:
                        if (x == 0 and z == 0) or (x == 0 and z == DIM_Z-1) or (x == DIM_X-1 and z == 0) or (x == DIM_X-1 and z == DIM_Z-1):
                            initLevel[x][y][z] = BLOCK_REGISTRY['bedrock']
                        else:
                            initLevel[x][y][z] = BLOCK_REGISTRY['gray_wool']
                elif y == DIM_Y-1:
                    if x == 0 or x == DIM_X-1 or z == 0 or z == DIM_Z-1:
                        initLevel[x][y][z] = BLOCK_REGISTRY['bedrock']
                    else:
                        initLevel[x][y][z] = BLOCK_REGISTRY['light_gray_wool']
                else:
                    if x == 0 or x == DIM_X-1 or z == 0 or z == DIM_Z-1:
                        if (x == 0 and z == 0) or (x == 0 and z == DIM_Z-1) or (x == DIM_X-1 and z == 0) or (x == DIM_X-1 and z == DIM_Z-1):
                            initLevel[x][y][z] = BLOCK_REGISTRY['bedrock']
                        else:
                            initLevel[x][y][z] = BLOCK_REGISTRY['gray_wool']

def selectNodes():
    """
        레벨 내 플레이어가 도달 가능한 위치(정점)를 임의로 결정하는 함수
        COMPLEXITY  - 결정할 위치의 수 인자
    """
    global DIM_X, DIM_Y, DIM_Z, initLevel, COMPLEXITY, nodeList
    
    # 플레이어가 도달할 수 있는 위치 배열 초기화
    possiblePos = DIM_X * DIM_Y * DIM_Z
    posMask = [[[0 for _ in range(DIM_Z)] for _ in range(DIM_Y)] for _ in range(DIM_X)]
    for x in range(DIM_X):
        for y in range(DIM_Y):
            for z in range(DIM_Z):
                if initLevel[x][y][z] != -1:
                    posMask[x][y][z] = 1
                    possiblePos -= 1
    
    entryPosition = ((0, ENTRY_POS, DIM_Z // 2), (DIM_X - 1, ENTRY_POS, DIM_Z // 2), (DIM_X // 2, ENTRY_POS, 0), (DIM_X // 2, ENTRY_POS, DIM_Z - 1))
    for x, y, z in entryPosition:
        nodeList.append((x, y, z))
        maskedCount = maskPosition(x, y, z, posMask)
        possiblePos -= maskedCount
    
    for epoch in range(COMPLEXITY):
        # 가능한 위치가 없거나 너무 적을 경우 중지
        if possiblePos <= 0: break
        if (DIM_X * DIM_Y * DIM_Z) // possiblePos >= 20: break
        
        pos_x, pos_y, pos_z = None, None, None
        while True:
            pos_x = random.randint(1, DIM_X-1)
            pos_y = random.randint(1, DIM_Y-3)
            pos_z = random.randint(1, DIM_Z-1)
            if posMask[pos_x][pos_y][pos_z] == 0: break
        nodeList.append((pos_x, pos_y, pos_z))
        maskedCount = maskPosition(pos_x, pos_y, pos_z, posMask)
        possiblePos -= maskedCount

def maskPosition(p, q, r, posMask):
    """
        결정한 정점 주변 정육면체 구역을 마스킹하는 함수
        POS_MASK_SIZE   - 
    """
    global POS_MASK_SIZE
    maskedCount = 0
    pos_offset = POS_MASK_SIZE // 2
    for i in range(POS_MASK_SIZE):
        for j in range(POS_MASK_SIZE):
            for k in range(POS_MASK_SIZE):
                
                x = p - pos_offset + i
                y = q - pos_offset + j
                z = r - pos_offset + k
                if x < 0 or y < 0 or z < 0 or x >= DIM_X or y >= DIM_Y or z >= DIM_Z:
                    continue
                if posMask[x][y][z] == 0:
                    maskedCount += 1
                    posMask[x][y][z] = 1
    return maskedCount

def createMovementGraph(nodeList):
    """
        결정한 정점으로부터 플레이어 동선 그래프를 생성하는 함수
    """
    global edgeSet
    movementGraph = [[] for _ in range(len(nodeList))]
    maxDistSquared = NODE_DIST_MAX * NODE_DIST_MAX
    for node in range(len(nodeList)):
        p, q, r = nodeList[node]
        nearbyNode = []
        
        for idx in range(len(nodeList)):
            i, j, k = nodeList[idx]
            dx = (p - i) * (p - i)
            dy = (q - j) * (q - j)
            dz = (r - k) * (r - k)
            distSquared = dx + dy + dz
            if distSquared == 0: continue
            nearbyNode.append((distSquared, i, j, k, idx))
        
        # 주변 노드를 거리순으로 정렬
        nearbyNode.sort(key=lambda x: x[0])
        
        # 각 노드의 x, z 좌표를 원점으로 하는 좌표평면 상의 각 사분면에서 하나씩만 선택
        edgeQuadrant = [[0, 0], [0, 0]]
        for d, i, j, k, idx in nearbyNode:
            if d > maxDistSquared: break
            dx = i - p
            dy = j - q
            dz = k - r
            quad_x = 1 if dx >= 0 else 0
            quad_z = 1 if dz >= 0 else 0
            if edgeQuadrant[quad_x][quad_z] == 0:
                movementGraph[node].append(idx)
                a = node
                b = idx
                if a > b: a, b = b, a
                edgeSet.add((a, b, d))
                edgeQuadrant[quad_x][quad_z] = 1
    return movementGraph

def evaluateEdges():
    pass

def getCuboid(idx1, idx2):
    """
        레벨 내부의 두 정점으로 결정되는 직육면체 구역을 반환하는 함수
        idx1, idx2  - 두 정점의 번호
    """
    global nodeList
    x1, y1, z1 = nodeList[idx1]
    x2, y2, z2 = nodeList[idx2]
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    if z1 > z2: z1, z2 = z2, z1
    x1 = min(DIM_X - 1, max(0, x1 - CUBOID_PADDING))
    x2 = min(DIM_X - 1, max(0, x2 + CUBOID_PADDING))
    y1 = min(DIM_Y - 1, max(0, y1 - CUBOID_PADDING))
    y2 = min(DIM_Y - 1, max(0, y2 + CUBOID_PADDING))
    z1 = min(DIM_Z - 1, max(0, z1 - CUBOID_PADDING))
    z2 = min(DIM_Z - 1, max(0, z2 + CUBOID_PADDING))
    return x1, 0, z1, x2, y2, z2

def createLevel():
    pass

def applyPatternFilter(x, y, z, mask):
    """
        파동함수의 지정된 좌표에 패턴 필터 적용
        -2  - 공기 블록만 받아들이는 필터
        -3  - 공기 블록이 아닌 것만 받아들이는 필터
    """
    global initLevel
    for i in range(8):
        p = x + dx_3x3[i]
        r = z + dz_3x3[i]
        if p < 0 or r < 0 or p >= DIM_X or r >= DIM_Z: continue
        if initLevel[p][y][r] == -1:
            initLevel[p][y][r] = mask

def constructPath(idx1, idx2):
    """
        두 정점으로 결정된 경로를 잇는 구조물을 생성하는 함수
        idx1, idx2  - 두 정점의 번호
    """
    global initLevel, nodeList
    for idx in (idx1, idx2):
        p, q, r = nodeList[idx]
        applyPatternFilter(p, q, r, -3)
        applyPatternFilter(p, q+1, r, -2)
    x1, y1, z1, x2, y2, z2 = getCuboid(idx1, idx2)
    pathSegment = initLevel[x1:x2+1][y1:y2+1][z1:z2+1]
    generatorModel = wfcModel(x2-x1+1, y2-y1+1, z2-z1+1, pathSegment, PATTERNS, BLOCK_REGISTRY,)
    pathSegment = wfcModel.generate()
    for x in range(x1, x2+1):
        for y in range(y1, y2+1):
            for z in range(z1, z2+1):
                initLevel[x][y][z] = pathSegment[x][y][z]

if __name__ == "__main__":
    mcaInitializer()
    levelInitializer()
    print(BLOCK_REGISTRY)
    selectNodes()
    movementGraph = createMovementGraph(nodeList)
    print(movementGraph)
    
    debugField = [[-1 for _ in range(DIM_Z)] for _ in range(DIM_X)]
    for idx in range(len(nodeList)):
        i, j, k = nodeList[idx]
        debugField[i][k] = idx
    for i in range(DIM_X):
        for k in range(DIM_Z):
            print("{:>2} ".format(debugField[i][k] if debugField[i][k] != -1 else '--'), end='')
        print()
    
    # TODO WFC 알고리즘 클래스 테스트