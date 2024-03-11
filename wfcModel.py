import random
from copy import deepcopy
from constantTable import MATRIX_X, MATRIX_Y, MATRIX_Z, OFF_X, OFF_Y, OFF_Z

class wfcModel:
    """
        dim_x, dim_y, dim_z - WFC 알고리즘을 실행할 공간의 차원 크기
        initWave            - 파동함수의 초기 상태 배열
        patterns            - 패턴 리스트
        blockRegistry       - 블록 레지스트리
        excludeBlocks       - 레벨 내부에 등장할 수 없는 블록 리스트
    """
    def __init__(self, dim_x, dim_y, dim_z, initWave, patterns, blockRegistry, excludeBlocks):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        
        self.updateStack = []
        self.waveFunc = [[[Node(dim_x, dim_y, dim_z, patterns=patterns, blockRegistry=blockRegistry, excludeBlocks=excludeBlocks, updateStack=self.updateStack) for _ in range(dim_z)] for _ in range(dim_y)] for _ in range(dim_x)]
        
        self.entropyInitializer()
        self.waveFunctionInitializer(initWave=initWave)
    
    def generate(self):
        """
            WFC 알고리즘 실행 후 생성된 데이터를 반환
            만약 파동 함수가 모순 상태에 빠졌다면 None 반환
        """
        if not self.waveFunc:
            raise ValueError("Wave Function Collapse model is not initialized.")
        step = 0
        while True:
            check = self.wfcStep(step)
            step += 1
            if check == 0:
                break
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
        x, y, z = self.findLeastEntropy()
        if x == -1:
            return 0
        self.waveFunc[x][y][z].prohibitState(x, y, z)
        self.waveFunc[x][y][z].collapse(x, y, z)
        self.waveFunc[x][y][z].propagate(x, y, z)
        return 1
    
    def findLeastEntropy(self):
        """
            매트릭스 내 엔트로피가 최소인 노드의 좌표를 반환하는 함수.
        """
        coords = []
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                for k in range(self.dim_z):
                    nodeEntropy, patternEntropy = self.waveFunc[i][j][k].getEntropy()
                    if self.waveFunc[i][j][k].collapsed == 0 and self.waveFunc[i][j][k].isContradicted() == 0:
                        coords.append((nodeEntropy, patternEntropy, i, j, k))
        # 정렬 기준: nodeEntropy, patternEntropy
        coords.sort(key=lambda x:(x[0], x[1]))
        for e1, e2, x, y, z in coords:
            if e1 > 0 and e2 > 0 and self.waveFunc[x][y][z].collapsed == 0 and self.waveFunc[x][y][z].isContradicted() == 0:
                return (x, y, z)
        # 매트릭스의 모든 원소가 붕괴된 상태
        return (-1, -1, -1)
    
    def updateNodes(self):
        """
            중첩 상태 업데이트가 필요한 모든 노드를 업데이트하는 함수
        """
        while self.updateStack:
            x, y, z = self.updateStack.pop()
            self.waveFunc[x][y][z].prohibitState(x, y, z)
    
    def checkContradiction(self):
        """
            파동 함수가 모순 상태에 빠졌는지 확인하는 함수
        """
        for i in range(self.dim_x):
            for j in range(self.dim_x):
                for k in range(self.dim_z):
                    if self.waveFunc[i][j][k].isContradicted():
                        return 1
        return 0
    
    def entropyInitializer(self):
        for x in range(self.dim_x):
            for y in range(self.dim_y):
                for z in range(self.dim_z):
                    entropy = 0
                    for p in range(MATRIX_X):
                        for q in range(MATRIX_Y):
                            for r in range(MATRIX_Z):
                                i = x - OFF_X + p
                                j = y - OFF_Y + q
                                k = z - OFF_Z + r
                                if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue
                                entropy += self.waveFunc[i][j][k].entropy
                    self.waveFunc[x][y][z].pattern_entropy = entropy
    
    def waveFunctionInitializer(self, initWave):
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                for k in range(self.dim_z):
                    self.waveFunc[i][j][k].waveFunc = self.waveFunc
                    if initWave[i][j][k] != -1:
                        self.waveFunc[i][j][k].setBlockID(initWave[i][j][k], i, j, k)
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                for k in range(self.dim_z):
                    self.waveFunc[i][j][k].prohibitState(i, j, k)

class Node:
    def __init__(self, dim_x:int, dim_y:int, dim_z:int, states = [], patterns = [], blockRegistry = {}, excludeBlocks = [], updateStack = None):
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
        # 상태 셀 -> 파동 함수에서 주변의 확정된 블록 정보를 저장하는 배열
        self.stateCell = [[[-1 for _ in range(MATRIX_Z)] for _ in range(MATRIX_Y)] for _ in range(MATRIX_X)]
        
        self.coverCheckIndex = ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0))
        self.waveFunc = []
        self.blockRegistry = blockRegistry
        self.excludeBlocks = excludeBlocks
        self.patterns = patterns
        self.updateStack = updateStack
        self.patternCount = len(patterns)
    
    def __call__(self):
        return self
    
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
            block_id = self.blockRegistry[block_id]
        if type(block_id) == type(0):
            pass
        self.block_id = block_id
        self.collapsed = 1
        self.propagateDelta(self.entropy, x, y, z)
        self.entropy = 0
    
    def collapse(self, x, y, z):
        """
            매트릭스 상에서 현재 노드의 중첩 상태를 임의의 정해진 상태로 붕괴시키는 함수.
        """
        possibleStates = []
        self.propagateDelta(self.entropy, x, y, z)
        self.entropy = 0
        self.collapsed = 1
        
        for i in range(self.patternCount):
            if self.states[i] != 0: possibleStates.append(i)
        if len(possibleStates) == 0: return
        collapsed_id = random.choice(possibleStates)
        self.pattern_id = collapsed_id
        self.block_id = self.patterns[collapsed_id].getCenter()
    
    def propagate(self, x, y, z, updateStack):
        """
            주변 노드에 현재 노드의 패턴을 전파하는 함수.
        """
        for p in range(MATRIX_X):
            for q in range(MATRIX_Y):
                for r in range(MATRIX_Z):
                    i = x - OFF_X + p
                    j = y - OFF_Y + q
                    k = z - OFF_Z + r
                    if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue
                    if self.pattern_id == 1:
                        block_id = -1
                    else:
                        block_id = self.patterns[self.pattern_id].state[p][q][r]
                    
                    if self.waveFunc[i][j][k].block_id == -1 and block_id != -1:
                        self.waveFunc[i][j][k].block_id = block_id
        
        for p in range(MATRIX_X+2):
            for q in range(MATRIX_Y+2):
                for r in range(MATRIX_Z+2):
                    i = x - (OFF_X+1) + p
                    j = y - (OFF_Y+1) + q
                    k = z - (OFF_Z+1) + r
                    if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue
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
                    if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue
                    if self.stateCell[p][q][r] != self.waveFunc[i][j][k].block_id:
                        updateCheck += 1
                        self.stateCell[p][q][r] = self.waveFunc[i][j][k].block_id
        count = 0
        for p, q, r in self.coverCheckIndex:
            i = x - OFF_X + p
            j = y - OFF_Y + q
            k = z - OFF_Z + r
            if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z:
                count += 1
                continue
            if self.waveFunc[i][j][k].block_id != -1: count += 1
        if count == len(self.coverCheckIndex): self.isCovered = 1
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
                    if matrix[i][j][k] in self.excludeBlocks:
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
        entropyDelta = 0
        for i in range(self.patternCount):
            if self.states[i] != 0:
                if self.validatePattern(self.patterns[i].state) == 0:
                    self.states[i] = 0
                    entropyDelta += 1
        self.entropy -= entropyDelta
        self.propagateDelta(entropyDelta, x, y, z)
    
    def propagateDelta(self, delta, x, y, z):
        for p in range(MATRIX_X):
            for q in range(MATRIX_Y):
                for r in range(MATRIX_Z):
                    i = x - OFF_X + p
                    j = y - OFF_Y + q
                    k = z - OFF_Z + r
                    if i < 0 or j < 0 or k < 0 or i >= self.dim_x or j >= self.dim_y or k >= self.dim_z: continue
                    self.waveFunc[i][j][k].pattern_entropy -= delta

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