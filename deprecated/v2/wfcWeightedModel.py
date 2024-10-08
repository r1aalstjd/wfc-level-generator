import random, math
from copy import deepcopy
from datetime import datetime
from constantTable import MATRIX_X, MATRIX_Y, MATRIX_Z, OFF_X, OFF_Y, OFF_Z, EPSILON
from constantTable import FILTER_AIR, FILTER_PLATFORM, FILTER_ADJACENT_WALL, FILTER_ADJACENT_FLOOR
from animatedVisualizer import animatedVisualizer
from heuristicConfig import WFCHeuristic

class wfcWeightedModel:
    """
        dim_x, dim_y, dim_z - WFC 알고리즘을 실행할 공간의 차원 크기
        
        initWave            - 파동함수의 초기 상태 배열
        
        patterns            - 패턴 리스트
        
        blockRegistry       - 블록 레지스트리
        
        excludeBlocks       - 레벨 내부에 등장할 수 없는 블록 리스트
        
        prioritizedCoords   - WFC 알고리즘 실행 시 우선적으로 붕괴할 노드의 좌표 리스트
    """
    def __init__(self, dim_x:int, dim_y:int, dim_z:int, initWave, patterns, blockRegistry, excludeBlocks,
                priortizedCoords:list[tuple[int, int, int]] = [], heuristic = WFCHeuristic.Entropy, debug = False):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.heuristic = heuristic
        
        self.patterns = patterns
        self.blockRegistry = blockRegistry
        self.excludeBlocks = excludeBlocks
        self.prioritizedCoords = priortizedCoords
        
        self.debugMode = debug
        self.scanlineLastIndex = -1
        self.scanlineInvLastIndex = self.dim_x * self.dim_y * self.dim_z
        self.scanlineLoopCheck = 0
        
        self.debugFrames = []
        self.nodeUpdateStack = []
        self.waveFunc = [[[Node(dim_x, dim_y, dim_z, states=[1 for _ in range(len(patterns))], model=self) for _ in range(dim_z)] for _ in range(dim_y)] for _ in range(dim_x)]
        self.waveFunctionInitializer(initWave=initWave)
    
    def applyFilters(self, platformFilter:set[int], floorAdjacentFilter:set[int], wallAdjacentFilter:set[int]):
        self.platformFilter = platformFilter
        self.floorAdjacentFilter = floorAdjacentFilter
        self.wallAdjacentFilter = wallAdjacentFilter
    
    def debugWriteFrame(self, x, y, z):
        if not self.debugMode: return
        self.debugFrames.append((deepcopy(self.waveFunc), x, y, z))
    
    def debugVisualization(self):
        pass
    
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
        
        if self.debugMode:
            dt = datetime.now().strftime(r'%Y%m%d-%H%M%S')
            animatedVisualizer(self.debugFrames, list(self.blockRegistry.keys()), 
                            './Debug/Visualization/', 'wfc-{}.gif'.format(dt), interval=200, elev=22.5, azim=-135, roll=0)
        cont = self.checkContradiction()
        if cont:
            print("Contradiction detected.")
        #if cont: return None

        outputFunc = [[[-1 for _ in range(self.dim_z)] for _ in range(self.dim_y)] for _ in range(self.dim_x)]
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                for k in range(self.dim_z):
                    outputFunc[i][j][k] = self.waveFunc[i][j][k].block_id
        return outputFunc
    
    def wfcStep(self, step):
        self.updateNodes()
        x, y, z = -1, -1, -1
        
        if self.heuristic == WFCHeuristic.Scanline:
            x, y, z = self.findNextWithScanline()
        else:
            x, y, z = self.findLeastEntropy()
        
        if self.debugMode:
            print('{:>2} {:>2} {:>2} {:>4}'.format(x, y, z, step))
        if x == -1:
            return 0
        self.waveFunc[x][y][z].prohibitState(x, y, z)
        self.waveFunc[x][y][z].collapse(x, y, z)
        self.waveFunc[x][y][z].propagate(x, y, z, self.nodeUpdateStack)
        
        if self.debugMode:
            self.debugWriteFrame(x, y, z)
        
        return 1
    
    @DeprecationWarning
    def calculateEntropy(self, x:int, y:int, z:int):
        nodeEntropy, weightSum = self.waveFunc[x][y][z].getEntropy()
        return nodeEntropy, weightSum
    
    def checkAvailableNode(self, x:int, y:int, z:int):
        if self.waveFunc[x][y][z].collapsed == 1 or self.waveFunc[x][y][z].isContradicted() == 1:
            return 0
        # 벽, 바닥 인접 패턴 및 빈 공간에 해당하는 노드 붕괴 방지
        if self.waveFunc[x][y][z].block_id == -1 or self.waveFunc[x][y][z].block_id == self.blockRegistry['air']:
            return 0
        if self.waveFunc[x][y][z].block_id == FILTER_ADJACENT_FLOOR or self.waveFunc[x][y][z].block_id == FILTER_ADJACENT_WALL:
            return 0
        return 1
    
    def findLeastEntropy(self) -> tuple[int, int, int]:
        """
            파동함수 내 엔트로피가 최소인 노드의 좌표를 반환하는 함수.
        """
        coords = []
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                for k in range(self.dim_z):
                    nodeEntropy, weightSum = self.waveFunc[i][j][k].getEntropy()
                    if self.checkAvailableNode(i, j, k) == 1:
                        coords.append((nodeEntropy, weightSum, i, j, k))
        
        # 엔트로피 기준으로 정렬
        coords.sort(key=lambda x:(x[0]))
        
        for e, wsum, x, y, z in coords:
            if wsum > 0:
                if self.debugMode: print('{:>10} | '.format(round(e, 6)), end='')
                return (x, y, z)
        
        # 매트릭스의 모든 원소가 붕괴된 상태
        return (-1, -1, -1)
    
    def findNextWithScanline(self) -> tuple[int, int, int]:
        """
            Scanline 휴리스틱으로 다음 노드를 선택해 좌표를 반환하는 함수.
            
            Scanline 순서: (0, dim_y-1, 0) -> (X+, Y-, Z+) -> (dim_x-1, 0, dim_z-1)
        """
        # 정방향 Scanline
        index = self.scanlineLastIndex + 1
        if self.scanlineLoopCheck == 0:
            for idx in range(index, self.dim_x * self.dim_y * self.dim_z):
                y = idx // (self.dim_x * self.dim_z)
                x = (idx % (self.dim_x * self.dim_z)) // self.dim_z
                z = idx % self.dim_z
                yInv = self.dim_y - y - 1
                if self.checkAvailableNode(x, yInv, z) == 1:
                    self.scanlineLastIndex = idx
                    return (x, yInv, z)
            else:
                self.scanlineLoopCheck = 1
        
        # 역방향 Scanline
        indexInv = self.scanlineInvLastIndex - 1
        for idx in range(indexInv, -1, -1):
            y = idx // (self.dim_x * self.dim_z)
            x = (idx % (self.dim_x * self.dim_z)) // self.dim_z
            z = idx % self.dim_z
            yInv = self.dim_y - y - 1
            if self.checkAvailableNode(x, yInv, z) == 1:
                self.scanlineInvLastIndex = idx
                return (x, yInv, z)
        return (-1, -1, -1)
    
    def updateNodes(self):
        """
            중첩 상태 업데이트가 필요한 모든 노드를 업데이트하는 함수
        """
        while self.nodeUpdateStack:
            x, y, z = self.nodeUpdateStack.pop()
            self.waveFunc[x][y][z].prohibitState(x, y, z)
    
    def checkContradiction(self):
        """
            파동 함수가 모순 상태에 빠졌는지 확인하는 함수
        """
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                for k in range(self.dim_z):
                    if self.waveFunc[i][j][k].isContradicted():
                        #if len(self.stateCell[i][j][k]) == 1:
                        #    self.waveFunc[i][j][k].block_id = self.stateCell[i][j][k].pop()
                        return 1
        return 0
    
    def waveFunctionInitializer(self, initWave):
        """
            파동 함수 내 모든 노드의 초기 상태 설정 함수
        """
        # initWave 내 블록 배치에 따라 노드 초기화
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                for k in range(self.dim_z):
                    if initWave[i][j][k] > -1:
                        self.waveFunc[i][j][k].setBlockID(initWave[i][j][k], i, j, k)
                    if initWave[i][j][k] <= -1:
                        self.waveFunc[i][j][k].setFilter(initWave[i][j][k], i, j, k)

        phase = 0
        # 중첩 상태인 모든 노드에서 불가능한 상태 제거
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                for k in range(self.dim_z):
                    self.waveFunc[i][j][k].prohibitState(i, j, k)
                    #if self.waveFunc[i][j][k].collapsed == 0:
                        #self.debugStateCell('./testStateCell/phase{}.txt'.format(phase), 'w+', (i, j, k))
                    phase += 1
        
        #self.updateStateCell()

        #self.debugStateCell('./testStateCell/init1.txt', 'w+')

class Node:
    def __init__(self, dim_x:int, dim_y:int, dim_z:int, states = [], model:wfcWeightedModel = None):
        self.states = deepcopy(states)
        self.block_id = -1
        self.pattern_id = -1
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.model = model
        self.patternCount = len(self.model.patterns)
        self.blockCount = len(self.model.blockRegistry)
        
        self.entropy = 0
        self.weightSum = 0
        self.updateEntropy()
        self.collapsed = 0
        self.isCovered = 0
        
        self.coverCheckIndex = ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0))
    
    def __call__(self):
        return self
    
    def getAbsoluteCoord(self, x, y, z, p, q, r) -> tuple[int, int, int]:
        return x - OFF_X + p, y - OFF_Y + q, z - OFF_Z + r
    
    def isContradicted(self):
        if self.weightSum == 0 and self.collapsed == 0:
            if self.isCovered != 0: return 0
            else: return 1
        return 0
    
    def getEntropy(self):
        return self.entropy, self.weightSum
    
    def updateEntropy(self):
        """
            엔트로피 값을 업데이트하는 함수 (상태 배열 업데이트 필요)
            
            Entropy = -\sigma{p(x)log(p(x))}
            p(x)        = w / sum
            log(p(x))   = log(w/sum)    = log(w) - log(sum)
            ->  w / sum * (log(sum) - log(w))
            ->  w / sum * log(sum) - w / sum * log(w)
            ->  w * log(sum) / sum - w * log(w) / sum
        """
        weightSum = 0
        weightSumLog = 0
        entropy = 0
        for i in range(self.patternCount):
            if self.states[i] != 0:
                w = self.model.patterns[i].count
                weightSum += w
                weightSumLog += w * math.log2(w)
        if weightSum > 0:
            entropy = math.log2(weightSum) - weightSumLog / weightSum
        self.weightSum = weightSum
        self.entropy = entropy
        return self.entropy
    
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
        self.entropy = 0
        self.weightSum = 0
    
    def setFilter(self, pattern_id, x, y, z):
        """
            파동 함수 초기화 시 현재 노드에 필터를 적용하는 함수.
            FILTER_AIR              - 공기 블록만 받아들이는 필터
            FILTER_PLATFORM         - 플랫폼 구성 블록만 받아들이는 필터
            FILTER_ADJACENT_FLOOR   - 바닥 블록과 인접한 블록만 받아들이는 필터
            FILTER_ADJACENT_WALL    - 벽 블록과 인접한 블록만 받아들이는 필터
        """
        self.block_id = pattern_id
    
    @DeprecationWarning
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
                            #self.model.stateCellTemp[i][j][k].add(self.model.patterns[idx].state[p][q][r])
                            #self.model.stateCellUpdateStack.add((i, j, k))
    
    def collapse(self, x, y, z):
        """
            매트릭스 상에서 현재 노드의 중첩 상태를 임의의 정해진 상태로 붕괴시키는 함수.
        """
        if self.collapsed == 1: return
        possiblePatterns = []
        patternWeights = []
        self.entropy = 0
        self.weightSum = 0
        self.collapsed = 1
        
        # 현재 노드의 중첩 상태를 하나의 상태로 붕괴
        for i in range(self.patternCount):
            if self.states[i] != 0:
                possiblePatterns.append(i)
                patternWeights.append(self.model.patterns[i].count)
        if len(possiblePatterns) == 0: return
        
        collapsed_id = random.choices(possiblePatterns, weights=patternWeights)[0]
        
        # 현재 노드의 상태를 붕괴된 상태로 설정
        self.pattern_id = collapsed_id
        self.block_id = self.model.patterns[collapsed_id].getCenter()
    
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
                        if current_block == -1:
                            continue
                        if current_block == FILTER_AIR:
                            if pattern_block != self.model.blockRegistry['air'] and pattern_block >= 0:
                                return 0
                        if current_block == FILTER_PLATFORM:
                            if pattern_block not in self.model.platformFilter:
                                return 0
                        if current_block == FILTER_ADJACENT_FLOOR:
                            if pattern_block not in self.model.floorAdjacentFilter:
                                return 0
                        if current_block == FILTER_ADJACENT_WALL:
                            if pattern_block not in self.model.wallAdjacentFilter:
                                return 0
                    
                    # 현재 좌표에 블록 ID가 지정된 경우
                    if current_block >= 0:
                        #if pattern_block == -1:
                        #    continue
                        if pattern_block != current_block:
                            return 0
        
        return 1
    
    def prohibitState(self, x, y, z):
        """
            현재 노드의 중첩 상태에서 불가능한 패턴 제거
        """
        if self.collapsed == 1: return
        
        self.checkCover(x, y, z)
        
        # 현재 노드의 사용 가능 패턴 검사 및 불가능한 패턴 비활성화
        for idx in range(self.patternCount):
            if self.states[idx] != 0:
                check = self.validatePattern(x, y, z, self.model.patterns[idx].state)
                if check == 0:
                    self.states[idx] = 0
                else:
                    for p in range(MATRIX_X):
                        for q in range(MATRIX_Y):
                            for r in range(MATRIX_Z):
                                patternBlockID = self.model.patterns[idx].state[p][q][r]
                                if patternBlockID >= 0:
                                    i, j, k = self.getAbsoluteCoord(x, y, z, p, q, r)
                                    if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue

        self.updateEntropy()

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
    - 패턴 해시 테이블을 활용하는 모순 상태 체크 함수
    - 상태 셀로 인한 모순 상태 해결
"""