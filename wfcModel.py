import random
from copy import deepcopy
from constantTable import MATRIX_X, MATRIX_Y, MATRIX_Z, OFF_X, OFF_Y, OFF_Z
from constantTable import FILTER_AIR, FILTER_PLATFORM, FILTER_ADAJACENT_WALL, FILTER_ADAJACENT_FLOOR

class wfcModel:
    """
        dim_x, dim_y, dim_z - WFC 알고리즘을 실행할 공간의 차원 크기
        
        initWave            - 파동함수의 초기 상태 배열
        
        patterns            - 패턴 리스트
        
        blockRegistry       - 블록 레지스트리
        
        excludeBlocks       - 레벨 내부에 등장할 수 없는 블록 리스트
        
        prioritizedCoords   - WFC 알고리즘 실행 시 우선적으로 붕괴할 노드의 좌표 리스트
    """
    def __init__(self, dim_x:int, dim_y:int, dim_z:int, initWave, patterns, blockRegistry, excludeBlocks, floorAdjacentFilter:set[int], wallAdjacentFilter:set[int], priortizedCoords:list[tuple[int, int, int]] = []):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        
        print(priortizedCoords)
        
        self.patterns = patterns
        self.blockRegistry = blockRegistry
        self.excludeBlocks = excludeBlocks
        self.prioritizedCoords = priortizedCoords
        self.floorAdjacentFilter = floorAdjacentFilter
        self.wallAdjacentFilter = wallAdjacentFilter
        
        self.nodeUpdateStack = []
        self.stateCellUpdateStack = set()

        nodeStates = [1 for _ in range(len(patterns))]
        self.waveFunc = [[[Node(dim_x, dim_y, dim_z, states=nodeStates, model=self) for _ in range(dim_z)] for _ in range(dim_y)] for _ in range(dim_x)]
        self.stateCell = [[[set(i for i in range(len(blockRegistry))) for _ in range(dim_z)] for _ in range(dim_y)] for _ in range(dim_x)]
        self.stateCellTemp = [[[set() for _ in range(dim_z)] for _ in range(dim_y)] for _ in range(dim_x)]
        
        self.entropyInitializer()
        self.waveFunctionInitializer(initWave=initWave)
        
        self.debugOutput('./modelInitDebug/world.txt', 'w+')
        self.debugStateCell('./modelInitDebug/statecell.txt', 'w+')
        self.debugEntropy('./modelInitDebug/entropy.txt', 'w+')
    
    def debugOutput(self, dir, mode = 'w+', coord = None):
        """
            디버그용 - 파동 함수의 현재 상태를 파일로 출력
        """
        debugWorld = open(dir, mode)
        if coord:
            debugWorld.write('{:>2} {:>2} {:>2}\n'.format(coord[0], coord[1], coord[2]))
        for y in range(self.dim_y):
            for x in range(self.dim_x):
                for z in range(self.dim_z):
                    debugWorld.write('{:>3} '.format(self.waveFunc[x][y][z].block_id))
                debugWorld.write('\n')
            debugWorld.write('\n')
        debugWorld.close()
    
    def debugStateCell(self, dir, mode = 'w+', coord = None):
        """
            디버그용 - 블록 상태 셀의 현재 상태를 파일로 출력
        """
        debugWorld = open(dir, mode)
        if coord:
            debugWorld.write('{:>2} {:>2} {:>2}\n'.format(coord[0], coord[1], coord[2]))
        for y in range(self.dim_y):
            for x in range(self.dim_x):
                for z in range(self.dim_z):
                    debugWorld.write('{:>3} '.format(len(self.stateCell[x][y][z])))
                debugWorld.write('\n')
            debugWorld.write('\n')
        debugWorld.close()
    
    def debugEntropy(self, dir, mode = 'w+', coord = None):
        """
            디버그용 - 블록 상태 셀의 현재 상태를 파일로 출력
        """
        debugWorld = open(dir, mode)
        if coord:
            debugWorld.write('{:>2} {:>2} {:>2}\n'.format(coord[0], coord[1], coord[2]))
        for y in range(self.dim_y):
            for x in range(self.dim_x):
                for z in range(self.dim_z):
                    debugWorld.write('{:>3} '.format(self.waveFunc[x][y][z].entropy))
                debugWorld.write('\n')
            debugWorld.write('\n')
        debugWorld.close()
    
    def generate(self):
        """
            WFC 알고리즘 실행 후 생성된 데이터를 반환
            만약 파동 함수가 모순 상태에 빠졌다면 None 반환
        """
        if not self.waveFunc:
            raise ValueError("Wave Function Collapse model is not initialized.")
        
        # 플레이어 동선 그래프의 정점 위치에 해당하는 노드 우선 붕괴 및 전파
        for coord in self.prioritizedCoords:
            x, y, z = coord
            print(x, y, z, "prioritized")
            self.waveFunc[x][y][z].collapse(x, y, z)
            self.waveFunc[x][y][z].propagate(x, y, z, self.nodeUpdateStack)
        
        # WFC 알고리즘 실행
        step = 0
        while True:
            check = self.wfcStep(step)
            step += 1
            if check == 0:
                break
        
        #self.debugOutput('./Debug/debugWorld.txt', 'w+')
        
        cont = self.checkContradiction()
        if cont: return None
        outputFunc = [[[-1 for _ in range(self.dim_z)] for _ in range(self.dim_y)] for _ in range(self.dim_x)]
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                for k in range(self.dim_z):
                    outputFunc[i][j][k] = self.waveFunc[i][j][k].block_id
        return outputFunc
    
    def wfcStep(self, step):
        self.updateNodes()
        self.updateStateCell()
        x, y, z = self.findLeastEntropy()
        print('{:>2} {:>2} {:>2} {:>4}'.format(x, y, z, step))
        if x == -1:
            return 0
        self.waveFunc[x][y][z].prohibitState(x, y, z)
        self.waveFunc[x][y][z].collapse(x, y, z)
        self.waveFunc[x][y][z].propagate(x, y, z, self.nodeUpdateStack)
        
        self.debugOutput('./testOutputs/world{}.txt'.format(step), 'w+', (x, y, z))
        self.debugStateCell('./testStateCell/world{}.txt'.format(step), 'w+', (x, y, z))
        
        return 1
    
    def calculateEntropy(self, x, y, z):
        nodeEntropy, patternEntropy = self.waveFunc[x][y][z].getEntropy()
        distMin = 2147483647
        for coord in self.prioritizedCoords:
            p, q, r = coord
            distSquared = (x - p) * (x - p) + (y - q) * (y - q) + (z - r) * (z - r)
            distMin = min(distMin, distSquared)
        return distMin, nodeEntropy, patternEntropy
    
    def findLeastEntropy(self) -> tuple[int, int, int]:
        """
            매트릭스 내 엔트로피가 최소인 노드의 좌표를 반환하는 함수.
        """
        coords = []
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                for k in range(self.dim_z):
                    dist, nodeEntropy, patternEntropy = self.calculateEntropy(i, j, k)
                    if self.waveFunc[i][j][k].collapsed == 0 and self.waveFunc[i][j][k].isContradicted() == 0:
                        coords.append((dist, nodeEntropy, patternEntropy, i, j, k))
        
        # 정렬 기준: 엔트로피 -> 좌표 -> 패턴 엔트로피
        coords.sort(key=lambda x:(x[1], x[0], x[2]))
        
        for d, e1, e2, x, y, z in coords:
            if self.waveFunc[x][y][z].block_id == FILTER_ADAJACENT_FLOOR or self.waveFunc[x][y][z].block_id == FILTER_ADAJACENT_WALL:
                continue
            if e1 > 0 and e2 > 0 and self.waveFunc[x][y][z].collapsed == 0 and self.waveFunc[x][y][z].isContradicted() == 0:
                print('{:>4} {:>4} {:>4} | '.format(d, e1, e2), end='')
                return (x, y, z)
        # 매트릭스의 모든 원소가 붕괴된 상태
        return (-1, -1, -1)
    
    def updateNodes(self):
        """
            중첩 상태 업데이트가 필요한 모든 노드를 업데이트하는 함수
        """
        while self.nodeUpdateStack:
            x, y, z = self.nodeUpdateStack.pop()
            self.waveFunc[x][y][z].prohibitState(x, y, z)
    
    def updateStateCell(self):
        """
            중첩 상태 업데이트가 필요한 모든 상태 셀을 업데이트하는 함수
        """
        for x, y, z in self.stateCellUpdateStack:
            self.stateCell[x][y][z] &= self.stateCellTemp[x][y][z]
            self.stateCellTemp[x][y][z].clear()
        self.stateCellUpdateStack.clear()
    
    def checkContradiction(self):
        """
            파동 함수가 모순 상태에 빠졌는지 확인하는 함수
        """
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                for k in range(self.dim_z):
                    if self.waveFunc[i][j][k].isContradicted():
                        if len(self.stateCell[i][j][k]) == 1:
                            self.waveFunc[i][j][k].block_id = self.stateCell[i][j][k].pop()
                        else:
                            return 1
        return 0
    
    def entropyInitializer(self):
        """
            파동 함수 내 모든 노드의 패턴 엔트로피 초기화 함수
        """
        for x in range(self.dim_x):
            for y in range(self.dim_y):
                for z in range(self.dim_z):
                    entropy = 0
                    for p in range(MATRIX_X):
                        for q in range(MATRIX_Y):
                            for r in range(MATRIX_Z):
                                i, j, k = x - OFF_X + p, y - OFF_Y + q, z - OFF_Z + r
                                if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue
                                entropy += self.waveFunc[i][j][k].entropy
                    self.waveFunc[x][y][z].pattern_entropy = entropy
    
    def waveFunctionInitializer(self, initWave):
        """
            파동 함수 내 모든 노드의 초기 상태 설정 함수
        """
        # initWave 내 블록 배치에 따라 노드 초기화
        stateCellUpdatePos = []
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                for k in range(self.dim_z):
                    if initWave[i][j][k] > -1:
                        self.waveFunc[i][j][k].setBlockID(initWave[i][j][k], i, j, k)
                        stateCellUpdatePos.append((i, j, k))
                    if initWave[i][j][k] < -1:
                        self.waveFunc[i][j][k].setFilter(initWave[i][j][k], i, j, k)
                        stateCellUpdatePos.append((i, j, k))
        
        # 노드 초기 상태 설정 후 상태 셀 업데이트
        for x, y, z in stateCellUpdatePos:
            self.waveFunc[x][y][z].whitelistStateCell(x, y, z)
        self.updateStateCell()
        
        #self.debugStateCell('./testStateCell/init0.txt', 'w+')

        phase = 0
        # 중첩 상태인 모든 노드에서 불가능한 상태 제거
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                for k in range(self.dim_z):
                    self.waveFunc[i][j][k].prohibitState(i, j, k)
                    #if self.waveFunc[i][j][k].collapsed == 0:
                        #self.debugStateCell('./testStateCell/phase{}.txt'.format(phase), 'w+', (i, j, k))
                    phase += 1
        
        self.updateStateCell()

        #self.debugStateCell('./testStateCell/init1.txt', 'w+')

class Node:
    def __init__(self, dim_x:int, dim_y:int, dim_z:int, states = [], model:wfcModel = None):
        self.states = deepcopy(states)
        self.block_id = -1
        self.pattern_id = -1
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        
        self.entropy = len(self.states)
        self.pattern_entropy = 0
        self.collapsed = 0
        self.isCovered = 0
        
        self.model = model
        
        self.coverCheckIndex = ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0))
        self.patternCount = len(self.model.patterns)
        self.blockCount = len(self.model.blockRegistry)
        
        # 상태 셀 -> 파동 함수에서 주변 노드의 가능한 블록 상태 수를 저장하는 배열
        self.stateCache = [[[self.blockCount for _ in range(MATRIX_Z)] for _ in range(MATRIX_Y)] for _ in range(MATRIX_X)]
    
    def __call__(self):
        return self
    
    def getAbsoluteCoord(self, x, y, z, p, q, r) -> tuple[int, int, int]:
        return x - OFF_X + p, y - OFF_Y + q, z - OFF_Z + r
    
    def isContradicted(self):
        if self.entropy == 0 and self.collapsed == 0:
            if self.isCovered != 0: return 0
            else: return 1
        return 0
    
    def getEntropy(self):
        return self.entropy, self.pattern_entropy
    
    def setBlockID(self, block_id, x, y, z):
        """
            파동 함수 초기화 시 현재 노드의 블록 상태를 지정하는 함수.
        """
        if type(block_id) == type("str"):
            block_id = self.model.blockRegistry[block_id]
        if type(block_id) == type(0):
            pass
        self.block_id = block_id
        self.collapsed = 1
        self.model.stateCell[x][y][z] = {block_id}
        self.propagateDelta(self.entropy, x, y, z)
        self.entropy = 0
    
    def setFilter(self, pattern_id, x, y, z):
        """
            파동 함수 초기화 시 현재 노드에 필터를 적용하는 함수.
            -2  - 공기 블록만 받아들이는 필터
            -3  - 공기 블록이 아닌 것만 받아들이는 필터
        """
        self.block_id = pattern_id
        if pattern_id == FILTER_AIR:
            self.model.stateCell[x][y][z] = {self.model.blockRegistry['air']}
        elif pattern_id == FILTER_PLATFORM:
            self.model.stateCell[x][y][z].discard(self.model.blockRegistry['air'])
        elif pattern_id == FILTER_ADAJACENT_FLOOR:
            self.model.stateCell[x][y][z] = deepcopy(self.model.floorAdjacentFilter)
        elif pattern_id == FILTER_ADAJACENT_WALL:
            self.model.stateCell[x][y][z] = deepcopy(self.model.wallAdjacentFilter)
    
    def whitelistStateCell(self, x, y, z):
        """
        """
        for idx in range(self.patternCount):
            if self.validatePattern(x, y, z, self.model.patterns[idx].state) == 1:
                for p in range(MATRIX_X):
                    for q in range(MATRIX_Y):
                        for r in range(MATRIX_Z):
                            i, j, k = self.getAbsoluteCoord(x, y, z, p, q, r)
                            if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue
                            self.model.stateCellTemp[i][j][k].add(self.model.patterns[idx].state[p][q][r])
                            self.model.stateCellUpdateStack.add((i, j, k))
    
    def collapse(self, x, y, z):
        """
            매트릭스 상에서 현재 노드의 중첩 상태를 임의의 정해진 상태로 붕괴시키는 함수.
        """
        possibleStates = []
        self.propagateDelta(self.entropy, x, y, z)
        self.entropy = 0
        self.collapsed = 1
        
        # 현재 노드의 중첩 상태를 하나의 상태로 붕괴
        for i in range(self.patternCount):
            if self.states[i] != 0: possibleStates.append(i)
        if len(possibleStates) == 0: return
        collapsed_id = random.choice(possibleStates)
        
        # 현재 노드의 상태를 붕괴된 상태로 설정
        self.pattern_id = collapsed_id
        self.block_id = self.model.patterns[collapsed_id].getCenter()
        self.model.stateCell[x][y][z] = {self.block_id}
    
    def propagate(self, x, y, z, updateStack:list):
        """
            주변 노드에 현재 노드의 패턴을 전파하는 함수.
        """
        for p in range(MATRIX_X):
            for q in range(MATRIX_Y):
                for r in range(MATRIX_Z):
                    i, j, k = self.getAbsoluteCoord(x, y, z, p, q, r)
                    if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue
                    
                    block_id = -1
                    if self.pattern_id >= 0:
                        block_id = self.model.patterns[self.pattern_id].state[p][q][r]
                    
                    if self.model.waveFunc[i][j][k].block_id < 0 and block_id >= 0:
                        self.model.waveFunc[i][j][k].block_id = block_id
                        self.model.stateCell[i][j][k] = {block_id}
        
        for p in range(MATRIX_X+2):
            for q in range(MATRIX_Y+2):
                for r in range(MATRIX_Z+2):
                    i = x - (OFF_X+1) + p
                    j = y - (OFF_Y+1) + q
                    k = z - (OFF_Z+1) + r
                    if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue
                    updateStack.append((i, j, k))
    
    def updateStateCache(self, x, y, z):
        """
            현재 노드의 상태 셀 업데이트 함수
        """
        updateCheck = 0
        for p in range(MATRIX_X):
            for q in range(MATRIX_Y):
                for r in range(MATRIX_Z):
                    i, j, k = self.getAbsoluteCoord(x, y, z, p, q, r)
                    if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue
                    if self.stateCache[p][q][r] != len(self.model.stateCell[i][j][k]):
                        updateCheck += 1
                        self.stateCache[p][q][r] = len(self.model.stateCell[i][j][k])
        return updateCheck
    
    def checkCover(self, x, y, z):
        """
            현재 노드가 주변 블록 확정 노드에 의해 완전히 둘러싸인 상태인지 확인하는 함수
        """
        count = 0
        for p, q, r in self.coverCheckIndex:
            i, j, k = self.getAbsoluteCoord(x, y, z, p, q, r)
            if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z:
                count += 1
                continue
            if self.model.waveFunc[i][j][k].block_id >= 0: count += 1
        if count == len(self.coverCheckIndex): self.isCovered = 1
        return count
    
    def validatePattern(self, x, y, z, matrix):
        """
            현재 노드의 상태 셀을 기준으로 패턴의 적합 여부를 반환하는 함수
        """
        if self.block_id >= 0 and matrix[OFF_X][OFF_Y][OFF_Z] != self.block_id:
            return 0

        for p in range(MATRIX_X):
            for q in range(MATRIX_Y):
                for r in range(MATRIX_Z):
                    i, j, k = self.getAbsoluteCoord(x, y, z, p, q, r)
                    if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue
                    
                    current_block = self.model.waveFunc[i][j][k].block_id
                    pattern_block = matrix[p][q][r]
                    
                    # 벽, 바닥, 천장 블록 생성 방지
                    if pattern_block in self.model.excludeBlocks:
                        if current_block != pattern_block:
                            return 0
                    
                    # 현재 좌표에 필터가 적용된 경우
                    if current_block < 0:
                        if pattern_block not in self.model.stateCell[i][j][k]:
                            return 0
                        if current_block == -1:
                            continue
                        if current_block == FILTER_AIR:
                            if pattern_block != self.model.blockRegistry['air'] and pattern_block >= 0:
                                return 0
                        if current_block == FILTER_PLATFORM:
                            if pattern_block == self.model.blockRegistry['air'] or pattern_block < 0:
                                return 0
                        if current_block == FILTER_ADAJACENT_FLOOR:
                            if pattern_block not in self.model.floorAdjacentFilter:
                                return 0
                        if current_block == FILTER_ADAJACENT_WALL:
                            if pattern_block not in self.model.wallAdjacentFilter:
                                return 0
                    
                    # 현재 좌표에 블록 ID가 지정된 경우
                    if current_block >= 0:
                        if pattern_block == -1:
                            continue
                        if pattern_block != current_block:
                            return 0
        
        return 1
    
    def prohibitState(self, x, y, z):
        """
            현재 노드의 중첩 상태에서 불가능한 패턴 제거
        """
        if self.collapsed == 1: return
        
        #f = open('./testAvailablePatterns/{}_{}_{}.txt'.format(x, y, z), 'w+')
        #f.write('{:>2} {:>2} {:>2}\n'.format(x, y, z))
        
        updateCheck = self.updateStateCache(x, y, z)
        self.checkCover(x, y, z)
        if updateCheck < 1: return
        
        entropyDelta = 0
        
        #for p in range(MATRIX_X):
        #    for q in range(MATRIX_Y):
        #        for r in range(MATRIX_Z):
        #            i, j, k = self.getAbsoluteCoord(x, y, z, p, q, r)
        #            #f.write(str(self.model.stateCell[i][j][k]) + '\n')
        
        # 현재 노드의 사용 가능 패턴 검사 및 불가능한 패턴 비활성화
        for idx in range(self.patternCount):
            if self.states[idx] != 0:
                check = self.validatePattern(x, y, z, self.model.patterns[idx].state)
                if check == 0:
                    self.states[idx] = 0
                    entropyDelta += 1
                else:
                    #f.write(str(self.model.patterns[idx].state) + '\n')
                    for p in range(MATRIX_X):
                        for q in range(MATRIX_Y):
                            for r in range(MATRIX_Z):
                                patternBlockID = self.model.patterns[idx].state[p][q][r]
                                if patternBlockID >= 0:
                                    i, j, k = self.getAbsoluteCoord(x, y, z, p, q, r)
                                    if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue
                                    self.model.stateCellTemp[i][j][k].add(patternBlockID)
                                    self.model.stateCellUpdateStack.add((i, j, k))
        
        #f.close()

        self.entropy -= entropyDelta
        self.propagateDelta(entropyDelta, x, y, z)
    
    def propagateDelta(self, delta, x, y, z):
        for p in range(MATRIX_X):
            for q in range(MATRIX_Y):
                for r in range(MATRIX_Z):
                    i, j, k = self.getAbsoluteCoord(x, y, z, p, q, r)
                    if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue
                    self.model.waveFunc[i][j][k].pattern_entropy -= delta

class Pattern:
    def __init__(self, state = [], center_id = -1, count = 0):
        self.state = deepcopy(state)
        self.center_id = center_id
        self.count = count
    
    def __call__(self):
        return self
    
    def getCenter(self):
        return self.center_id

"""
    TODO
    - 
"""