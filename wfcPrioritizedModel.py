import random, anvil, time
from copy import deepcopy
from constantTable import MATRIX_X, MATRIX_Y, MATRIX_Z, EPOCH

"""
    WFC 우선순위 모델 구현
    - WFC 모델과 기본 메커니즘은 동일
    - 패턴에 우선순위를 부여하고, 각 우선순위별로 패턴 집합을 생성
    - 우선순위 집합 각각의 엔트로피를 따로 계산
    - 노드 붕괴 연산 또한 우선순위 집합 순서대로 시행
"""

INPUT_DIR = './Input/'
OUTPUT_DIR = './Output/'

SIZE_X = 27
SIZE_Y = 0
SIZE_Z = 27

OFF_X = MATRIX_X // 2
OFF_Y = MATRIX_Y // 2
OFF_Z = MATRIX_Z // 2

PATTERNS = None
BLOCK_REGISTRY = None
BLOCK_REGISTRY_INV = None
WFC_FIELD = None
EXCLUDE_BLOCK_ID = None

inputWorld = anvil.Region.from_file(INPUT_DIR + 'r.0.0.mca')
inputCache = []

blockCount = 0
patternCount = 0
updateStack = []
excludeBlocks = ['black_wool', 'gray_wool', 'light_gray_wool', 'bedrock']
coverCheckIndex = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]

class Node:
    def __init__(self):
        self.block_id = -1
        self.patternType = -1
        self.patternIndex = -1
        self.collapsed = 0
        self.isCovered = 0
        
        self.states = []
        for patternType in range(len(PATTERNS)):
            self.states.append([1 for _ in range(len(PATTERNS[patternType]))])
        
        self.entropies = []
        for patternType in range(len(PATTERNS)):
            self.entropies.append(len(PATTERNS[patternType]))
        
        self.pattern_entropy = 0
        
        # 상태 셀 -> 파동 함수에서 주변의 확정된 블록 정보를 저장하는 배열
        self.stateCell = [[[-1 for _ in range(MATRIX_Z)] for _ in range(MATRIX_Y)] for _ in range(MATRIX_X)]
    
    def __call__(self):
        return self
    
    def isContradicted(self):
        entropySum = 0
        for e in self.entropies:
            entropySum += e
        if entropySum == 0 and self.collapsed == 0:
            if self.isCovered != 0: return 0
            #if self.block_id == BLOCK_REGISTRY['air'] or self.block_id == BLOCK_REGISTRY['light_blue_wool'] or self.isCovered != 0: return 0
            else: return 1
        return 0
    
    def getEntropy(self):
        return self.entropies
    
    def resetEntropy(self):
        for i in range(len(self.entropies)):
            self.entropies[i] = 0
    
    def setBlockID(self, block_id, x, y, z):
        """
            파동 함수 초기화 시 현재 노드의 블록 상태를 지정하는 함수.
        """
        if type(block_id) == type("str"):
            block_id = BLOCK_REGISTRY[block_id]
        if type(block_id) == type(0):
            pass
        self.block_id = block_id
        self.collapsed = 1
        self.propagateDelta(self.entropies, x, y, z)
        self.resetEntropy()
    
    def collapse(self):
        """
            매트릭스 상에서 현재 노드의 중첩 상태를 임의의 정해진 상태로 붕괴시키는 함수.
        """
        global patternCount
        possibleStates = []
        self.resetEntropy()
        self.collapsed = 1
        
        for patternType in range(len(PATTERNS)):
            for i in range(len(PATTERNS[patternType])):
                if self.states[patternType][i] != 0: possibleStates.append(i)
            if len(possibleStates) != 0:
                collapsed_id = random.choice(possibleStates)
                self.patternType = patternType
                self.patternIndex = collapsed_id
                self.block_id = PATTERNS[patternType][collapsed_id].getCenter()
                return
    
    def propagate(self, x, y, z):
        """
            주변 노드에 현재 노드의 패턴을 전파하는 함수.
        """
        global updateStack
        for p in range(MATRIX_X):
            for q in range(MATRIX_Y):
                for r in range(MATRIX_Z):
                    i = x - OFF_X + p
                    j = y - OFF_Y + q
                    k = z - OFF_Z + r
                    if i < 0 or j < 0 or k < 0 or i >= SIZE_X or j >= SIZE_Y or k >= SIZE_Z: continue
                    if self.patternType == -1 or self.patternIndex == -1:
                        block_id = -1
                    else:
                        block_id = PATTERNS[self.patternType][self.patternIndex].state[p][q][r]
                    if WFC_FIELD[i][j][k].block_id == -1 and block_id != -1:
                        WFC_FIELD[i][j][k].block_id = block_id
        
        for p in range(MATRIX_X+2):
            for q in range(MATRIX_Y+2):
                for r in range(MATRIX_Z+2):
                    i = x - (OFF_X+1) + p
                    j = y - (OFF_Y+1) + q
                    k = z - (OFF_Z+1) + r
                    if i < 0 or j < 0 or k < 0 or i >= SIZE_X or j >= SIZE_Y or k >= SIZE_Z: continue
                    updateStack.append((i, j, k))
    
    def updateStateCell(self, x, y, z):
        """
            현재 노드의 상태 셀 업데이트 함수
        """
        updateCheck = 0
        for p in range(MATRIX_X):
            for q in range(MATRIX_Y):
                for r in range(MATRIX_Z):
                    i = x - OFF_X + p
                    j = y - OFF_Y + q
                    k = z - OFF_Z + r
                    if i < 0 or j < 0 or k < 0 or i >= SIZE_X or j >= SIZE_Y or k >= SIZE_Z: continue
                    if self.stateCell[p][q][r] != WFC_FIELD[i][j][k].block_id:
                        updateCheck += 1
                        self.stateCell[p][q][r] = WFC_FIELD[i][j][k].block_id
        count = 0
        global coverCheckIndex
        for p, q, r in coverCheckIndex:
            i = x - OFF_X + p
            j = y - OFF_Y + q
            k = z - OFF_Z + r
            if i < 0 or j < 0 or k < 0 or i >= SIZE_X or j >= SIZE_Y or k >= SIZE_Z:
                count += 1
                continue
            if WFC_FIELD[i][j][k].block_id != -1: count += 1
        if count == len(coverCheckIndex): self.isCovered = 1
        return updateCheck
    
    def validatePattern(self, matrix):
        """
            현재 노드의 상태 셀을 기준으로 패턴의 가능 여부 반환 함수
        """
        if self.block_id != -1 and matrix[OFF_X][OFF_Y][OFF_Z] != self.block_id:
            return 0
        for i in range(MATRIX_X):
            for j in range(MATRIX_Y):
                for k in range(MATRIX_Z):
                    # 벽, 바닥, 천장 블록 생성 방지
                    if matrix[i][j][k] in EXCLUDE_BLOCK_ID:
                        if self.stateCell[i][j][k] != matrix[i][j][k]:
                            return 0
                    if self.stateCell[i][j][k] == -1 or matrix[i][j][k] == -1:
                        continue
                    if self.stateCell[i][j][k] != matrix[i][j][k]:
                        return 0
        return 1
    
    def prohibitState(self, x, y, z):
        """
            현재 노드의 중첩 상태에서 불가능한 상태 제거
        """
        if self.collapsed == 1: return
        updateCheck = self.updateStateCell(x, y, z)
        if updateCheck < 1: return
        entropyDelta = [0 for _ in range(len(PATTERNS))]
        for patternType in range(len(PATTERNS)):
            for i in range(len(PATTERNS[patternType])):
                if self.states[patternType][i] != 0:
                    if self.validatePattern(PATTERNS[patternType][i].state) == 0:
                        self.states[patternType][i] = 0
                        entropyDelta[patternType] += 1
        for patternType in range(len(PATTERNS)):
            self.entropies[patternType] -= entropyDelta[patternType]
        self.propagateDelta(entropyDelta, x, y, z)
    
    def propagateDelta(self, delta, x, y, z):
        """
            주변 노드에 패턴 엔트로피 변화량 전파
            delta   - 우선순위 별 엔트로피 변화량 리스트
        """
        entropy = delta[0]
        for p in range(MATRIX_X):
            for q in range(MATRIX_Y):
                for r in range(MATRIX_Z):
                    i = x - OFF_X + p
                    j = y - OFF_Y + q
                    k = z - OFF_Z + r
                    if i < 0 or j < 0 or k < 0 or i >= SIZE_X or j >= SIZE_Y or k >= SIZE_Z: continue
                    WFC_FIELD[i][j][k].pattern_entropy -= entropy

class Pattern:
    def __init__(self, state = [], center_id = -1, count = 0):
        self.state = deepcopy(state)
        self.center_id = center_id
        self.count = count
    
    def __call__(self):
        return self
    
    def getCenter(self):
        return self.center_id
    
    def isEqual(self, matrix):
        cnt = 0
        for i in range(MATRIX_X):
            for j in range(MATRIX_Y):
                for k in range(MATRIX_Z):
                    if self.state[i][j][k] == matrix[i][j][k]:
                        cnt += 1
        return 1 if cnt == MATRIX_X * MATRIX_Y * MATRIX_Z else 0

def getMaxY():
    global inputWorld
    chunk = anvil.Chunk.from_region(inputWorld, 0, 0)
    for i in range(10000):
        if chunk.get_block(0, i, 0).id == 'air':
            return i
    else:
        return 0

def getBlock(x, y, z):
    """
        실제 월드에서의 (x, y, z) 좌표에 위치한 블록 객체를 반환하는 함수.
    """
    global inputWorld, inputCache
    if inputCache[x][y][z] != -1: return inputCache[x][y][z]
    else:
        chunk__x = x // 16
        chunk__z = z // 16
        block__x = x % 16
        block__z = z % 16
        inputCache[x][y][z] = anvil.Chunk.from_region(inputWorld, chunk__x, chunk__z).get_block(block__x, y, block__z)
        return inputCache[x][y][z]

def registerBlocks(timestamp):
    """
        읽어온 월드 파일 내 존재하는 블록을 레지스트리에 등록하는 함수
    """
    global inputWorld
    blockRegistry = {}
    blockRegistryInv = []
    
    for i in range(SIZE_X):
        for j in range(SIZE_Y):
            for k in range(SIZE_Z):
                block_id = getBlock(i, j, k).id
                blockRegistry[block_id] = 1
    q = 0
    for key in blockRegistry:
        blockRegistry[key] = q
        blockRegistryInv.append(key)
        q += 1
    timeElapsed = time.time()
    print("Block Registration completed. (took {}s)".format(round(timeElapsed - timestamp, 6)))
    return blockRegistry, blockRegistryInv

def extractPatterns(blockRegistry, timestamp):
    """
        월드 파일 내 블록 배치 패턴을 추출하는 함수
    """
    patterns = []
    for i in range(SIZE_X):
        for j in range(SIZE_Y):
            for k in range(SIZE_Z):
                propMatrix = [[[0 for _ in range(MATRIX_Z)] for _ in range(MATRIX_Y)] for _ in range(MATRIX_X)]
                check = 0
                for p in range(MATRIX_X):
                    for q in range(MATRIX_Y):
                        for r in range(MATRIX_Z):
                            x = i - OFF_X + p
                            y = j - OFF_Y + q
                            z = k - OFF_Z + r
                            if x < 0 or y < 0 or z < 0 or x >= SIZE_X or y >= SIZE_Y or z >= SIZE_Z:
                                propMatrix[p][q][r] = -1
                                check += 1
                            else:
                                propMatrix[p][q][r] = blockRegistry[getBlock(x, y, z).id]
                center_id = blockRegistry[getBlock(i, j, k).id]
                
                # 범위 밖 공간을 포함하는 패턴 필터링
                if check != 0: continue
                
                # 패턴 중심 블록이 공기 블록일 경우 와일드카드 패턴으로 대체
                air_count = 0
                if center_id == blockRegistry['air']:
                    for p in range(MATRIX_X):
                        for q in range(MATRIX_Y):
                            for r in range(MATRIX_Z):
                                if propMatrix[p][q][r] != blockRegistry['air']:
                                    propMatrix[p][q][r] = -1
                                else: air_count += 1
                
                if center_id == blockRegistry['air'] and air_count != MATRIX_X * MATRIX_Y * MATRIX_Z:
                    pass
                else:
                    registerPattern(propMatrix, center_id, patterns, priority=0)
    
    propMatrix = [[[0 for _ in range(MATRIX_Z)] for _ in range(MATRIX_Y)] for _ in range(MATRIX_X)]
    for p in range(MATRIX_X):
        for q in range(MATRIX_Y):
            for r in range(MATRIX_Z):
                propMatrix[p][q][r] = -1
    propMatrix[OFF_X][OFF_Y][OFF_Z] = blockRegistry['air']
    registerPattern(propMatrix, blockRegistry['air'], patterns, priority=1)
    
    timeElapsed = time.time()
    print("Pattern Extraction completed. (took {}s)".format(round(timeElapsed - timestamp, 6)))
    return patterns

def registerPattern(matrix, center_id, patterns, priority):
    # 회전 패턴 생성(Y축 시계방향)
    rotated90 = deepcopy(matrix)
    rotated180 = deepcopy(matrix)
    rotated270 = deepcopy(matrix)
    for x in range(MATRIX_X):
        for y in range(MATRIX_Y):
            for z in range(MATRIX_Z):
                x1 = z
                z1 = MATRIX_X - x - 1
                rotated90[x1][y][z1] = matrix[x][y][z]
                x2 = z1
                z2 = MATRIX_X - x1 - 1
                rotated180[x2][y][z2] = matrix[x][y][z]
                x3 = z2
                z3 = MATRIX_X - x2 - 1
                rotated270[x3][y][z3] = matrix[x][y][z]
    rotatedSet = [matrix, rotated90, rotated180, rotated270]
    
    while priority >= len(patterns):
        patterns.append([])
    
    # 패턴 중복 제거
    for extractedPattern in rotatedSet:
        for pattern in patterns[priority]:
            if pattern.isEqual(extractedPattern):
                pattern.count += 1
                break
        else:
            patterns[priority].append(Pattern(extractedPattern, center_id, 1))

def writeMCA(wfcMatrix, epoch):
    global blockCount
    outputWorld = anvil.EmptyRegion(0, 0)
    blockClass = []
    barrier = anvil.Block('minecraft', 'barrier')
    for i in range(blockCount):
        stringID = BLOCK_REGISTRY_INV[i]
        blockClass.append(anvil.Block('minecraft', stringID))
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for z in range(SIZE_Z):
                block_id = wfcMatrix[x][y][z].block_id
                if block_id == -1: outputWorld.set_block(barrier, x, y, z)
                else: outputWorld.set_block(blockClass[block_id], x, y, z)
    outputWorld.save(OUTPUT_DIR + '{} r.0.0.mca'.format(epoch+1))

def findLeastEntropy():
    """
        매트릭스 내 엔트로피가 최소인 노드의 좌표를 반환하는 함수.
    """
    coords = []
    for i in range(SIZE_X):
        for j in range(SIZE_Y):
            for k in range(SIZE_Z):
                eSum = 0
                for e in WFC_FIELD[i][j][k].getEntropy():
                    eSum += e
                e0, e1 = WFC_FIELD[i][j][k].getEntropy()
                ep = WFC_FIELD[i][j][k].pattern_entropy
                if eSum > 0 and WFC_FIELD[i][j][k].collapsed == 0 and WFC_FIELD[i][j][k].isContradicted() == 0:
                    coords.append((e0, e1, ep, eSum, i, j, k))
    
    # 정렬 기준: ep, e0, e1
    coords.sort(key=lambda x:(x[0], x[1]))

    for e0, e1, ep, eSum, x, y, z in coords:
        if e0 > 0 and WFC_FIELD[x][y][z].collapsed == 0 and WFC_FIELD[x][y][z].isContradicted() == 0:
            return (x, y, z)
    
    for e0, e1, ep, eSum, x, y, z in coords:
        if eSum > 0 and WFC_FIELD[x][y][z].collapsed == 0 and WFC_FIELD[x][y][z].isContradicted() == 0:
            return (x, y, z)
    
    # 매트릭스의 모든 원소가 붕괴된 상태
    return (-1, -1, -1)

def checkContradiction():
    """
        파동 함수가 모순 상태에 빠졌는지 확인하는 함수
    """
    for i in range(SIZE_X):
        for j in range(SIZE_Y):
            for k in range(SIZE_Z):
                if WFC_FIELD[i][j][k].isContradicted():
                    #print('Contradicted:', i, j, k)
                    return 1
    return 0

def updateNodes():
    """
        중첩 상태 업데이트가 필요한 모든 노드를 업데이트하는 함수
    """
    global updateStack, WFC_FIELD
    while len(updateStack) > 0:
        x, y, z = updateStack.pop()
        WFC_FIELD[x][y][z].prohibitState(x, y, z)

def WFCStep(step = -1, debug = 0, debugfile = None):
    """
        WFC 알고리즘의 한 단계(엔트로피가 최소인 원소를 찾아 붕괴 후 전파)를 수행하는 함수
    """
    updateNodes()
    #cont = checkContradiction()
    x, y, z = findLeastEntropy()
    
    # Debug
    if debug:
        e1, e2 = WFC_FIELD[x][y][z].getEntropy()
        ep = WFC_FIELD[x][y][z].pattern_entropy
        debugfile.write('{0:>3} {1:>3} {2:>3} {3:>5} {4:>5} {5:>5} {6:>5}\n'.format(x, y, z, step, e1, e2, ep))
        
        print('{0:>3} {1:>3} {2:>3} {3:>5} {4:>5} {5:>5} {6:>5}'.format(x, y, z, step, e1, e2, ep))
    
    #if cont == 1:
    #    print("Wave matrix ran into a contradictionary state.")
    #    return 0
    if x == -1:
        print("Wave matrix has collapsed into singular state.")
        return 0
    WFC_FIELD[x][y][z].prohibitState(x, y, z)
    WFC_FIELD[x][y][z].collapse()
    WFC_FIELD[x][y][z].propagate(x, y, z)
    
    if debug:
        st = ''
        for i in range(MATRIX_X):
            for j in range(MATRIX_Y):
                for k in range(MATRIX_Z):
                    # 2차원 배열 -> YZ 평면(Y 위 방향, Z 오른쪽 방향)
                    patternType = WFC_FIELD[x][y][z].patternType
                    patternIndex = WFC_FIELD[x][y][z].patternIndex
                    block_id = PATTERNS[patternType][patternIndex].state[i][MATRIX_Y-j-1][k]
                    if block_id == -1:
                        block_id = 'none'
                    else: block_id = BLOCK_REGISTRY_INV[block_id]
                    st += '{:>16} '.format(block_id)
                st += '\n'
            st += '\n'
        debugfile.write(st)
    
    return 1

def runWFC(epoch, debug = 0):
    """
        WFC 알고리즘을 수행하는 함수
    """
    step = 1
    if debug:
        logfile = open('./Debug/wfcPrioritizedModelLog.txt', 'w+')
    
    while True:
        check = WFCStep(step, debug, logfile)
        step += 1
        if check == 0:
            break
    
    cont = checkContradiction()
    if cont == 1:
        print("Wave matrix ran into a contradictionary state.")
    
    if debug:
        for i in range(SIZE_X):
            for j in range(SIZE_Y):
                for k in range(SIZE_Z):
                    if WFC_FIELD[i][j][k].isContradicted():
                        logfile.write('{0:>3} {1:>3} {2:>3}\n'.format(i, j, k))
        
        logfile.close()
    
    return WFC_FIELD

def entropyInitializer():
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for z in range(SIZE_Z):
                entropy = 0
                for p in range(MATRIX_X):
                    for q in range(MATRIX_Y):
                        for r in range(MATRIX_Z):
                            i = x - OFF_X + p
                            j = y - OFF_Y + q
                            k = z - OFF_Z + r
                            if i < 0 or j < 0 or k < 0 or i >= SIZE_X or j >= SIZE_Y or k >= SIZE_Z: continue
                            entropy += WFC_FIELD[i][j][k].getEntropy()[0]
                WFC_FIELD[x][y][z].pattern_entropy = entropy

def fieldInitializer(timestamp):
    entropyInitializer()
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for z in range(SIZE_Z):
                if y == 0:
                    if x == 0 or x == SIZE_X-1 or z == 0 or z == SIZE_Z-1:
                        WFC_FIELD[x][y][z].setBlockID('bedrock', x, y, z)
                    else:
                        WFC_FIELD[x][y][z].setBlockID('black_wool', x, y, z)
                elif y >= 1 and y <= 5:
                    if x >= 12 and x <= 14 and (z == 0 or z == SIZE_Z-1):
                        WFC_FIELD[x][y][z].setBlockID('air', x, y, z)
                    elif x >= 11 and x <= 15 and ((z >= 1 and z <= 2) or (z <= SIZE_Z-2 and z >= SIZE_Z-3)):
                        WFC_FIELD[x][y][z].setBlockID('air', x, y, z)
                    elif z >= 12 and z <= 14 and (x == 0 or x == SIZE_X-1):
                        WFC_FIELD[x][y][z].setBlockID('air', x, y, z)
                    elif z >= 11 and z <= 15 and ((x >= 1 and x <= 2) or (x <= SIZE_X-2 and x >= SIZE_X-3)):
                        WFC_FIELD[x][y][z].setBlockID('air', x, y, z)
                    elif x == 0 or x == 26 or z == 0 or z == SIZE_Z-1:
                        if (x == 0 and z == 0) or (x == 0 and z == SIZE_Z-1) or (x == SIZE_X-1 and z == 0) or (x == SIZE_X-1 and z == SIZE_Z-1):
                            WFC_FIELD[x][y][z].setBlockID('bedrock', x, y, z)
                        else:
                            WFC_FIELD[x][y][z].setBlockID('gray_wool', x, y, z)
                elif y == SIZE_Y-1:
                    if x == 0 or x == SIZE_X-1 or z == 0 or z == SIZE_Z-1:
                        WFC_FIELD[x][y][z].setBlockID('bedrock', x, y, z)
                    else:
                        WFC_FIELD[x][y][z].setBlockID('light_gray_wool', x, y, z)
                else:
                    if x == 0 or x == SIZE_X-1 or z == 0 or z == SIZE_Z-1:
                        if (x == 0 and z == 0) or (x == 0 and z == SIZE_Z-1) or (x == SIZE_X-1 and z == 0) or (x == SIZE_X-1 and z == SIZE_Z-1):
                            WFC_FIELD[x][y][z].setBlockID('bedrock', x, y, z)
                        else:
                            WFC_FIELD[x][y][z].setBlockID('gray_wool', x, y, z)
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for z in range(SIZE_Z):
                WFC_FIELD[x][y][z].prohibitState(x, y, z)
    
    timeElapsed = time.time()
    print("Wave function initialization completed. (took {}s)".format(round(timeElapsed - timestamp, 6)))

def initialize(timestamp):
    global blockCount, patternCount, inputCache, excludeBlocks
    global SIZE_Y, BLOCK_REGISTRY, BLOCK_REGISTRY_INV, PATTERNS, WFC_FIELD, EXCLUDE_BLOCK_ID
    SIZE_Y = getMaxY()
    inputCache = [[[-1 for _ in range(SIZE_Z)] for _ in range(SIZE_Y)] for _ in range(SIZE_X)]
    
    BLOCK_REGISTRY, BLOCK_REGISTRY_INV = registerBlocks(timestamp)
    blockCount = len(BLOCK_REGISTRY_INV)
    
    PATTERNS = extractPatterns(BLOCK_REGISTRY, timestamp)
    patternCount = len(PATTERNS)
    
    initStates = []
    for patternType in range(len(PATTERNS)):
        temp = []
        for i in range(len(PATTERNS[patternType])):
            temp.append(1)
        initStates.append(deepcopy(temp))
    #print(initStates)
    WFC_FIELD = [[[Node() for _ in range(SIZE_Z)] for _ in range(SIZE_Y)] for _ in range(SIZE_X)]
    
    EXCLUDE_BLOCK_ID = []
    for block in excludeBlocks:
        EXCLUDE_BLOCK_ID.append(BLOCK_REGISTRY[block])

if __name__ == '__main__':
    timestamp = time.time()
    initialize(timestamp)
    print(BLOCK_REGISTRY)
    
    for epoch in range(EPOCH):
        WFC_FIELD = [[[Node() for _ in range(SIZE_Z)] for _ in range(SIZE_Y)] for _ in range(SIZE_X)]
        fieldInitializer(timestamp)
        outputMatrix = runWFC(epoch, debug = 1)
        writeMCA(outputMatrix, epoch)
        timeElapsed = time.time()
        print("Epoch {} finished. (took {}s)".format(epoch+1, round(timeElapsed - timestamp, 6)))
    
    timeElapsed = time.time()
    print("Wave Function Collapse algorithm has been terminated. (took {}s)".format(round(timeElapsed - timestamp, 6)))

"""
    TODO
    패턴 엔트로피 재도입, 히든카드 패턴 도입
"""