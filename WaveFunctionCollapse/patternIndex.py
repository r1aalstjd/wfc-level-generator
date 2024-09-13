from constantTable import HASH_MODULO, NEUMANN_NEIGHBOR_3D
from copy import deepcopy
import numpy as np

class PatternIndex:
    def __init__(self):
        pass
    
    def __call__(self, patternShape:tuple[int, int, int], inputData:np.ndarray) -> "PatternIndex":
        self.patternList = self.extractPatterns(patternShape, inputData)
        self.index = self.createPatternIndex(self.patternList)
        return self
    
    def createPatternIndex(self, patternList:list["Pattern"]) -> dict[int, dict[tuple[int, int, int], set[int]]]:
        """
            두 패턴이 겹치는지 O(1) 시간에 확인하는 인덱스를 생성하는 함수
            
            index[pattern][offset] = offset만큼 이동시켰을 때 pattern에 해당하는 패턴과 겹치는 패턴 집합
        """
        index = dict()
        
        for idx in range(len(patternList)):
            index[idx] = dict()
            for coord in NEUMANN_NEIGHBOR_3D:
                index[idx][coord] = {p for p in range(len(patternList)) if self.checkOverlapping(idx, p, coord)}
        
        return index
    
    def canOverlap(self, basePatternID:int, patternID:int, offset:tuple[int, int, int]) -> bool:
        """
            patternID에 해당하는 패턴을 offset만큼 이동시켰을 때 basePatternID와 겹칠 수 있는지 확인하는 함수
            
            basePatternID: 기준 패턴
            patternID: offset만큼 이동시켰을 때 basePatternID와 겹치는지 확인할 대상
            offset: 패턴을 이동시킬 방향 벡터
        """
        return patternID in self.index[basePatternID][offset]
    
    def checkOverlapping(self, pattern1:int, pattern2:int, offset:tuple[int, int, int]) -> bool:
        """
            패턴 2를 offset만큼 이동시켰을 때 패턴 1과 겹치는지 (실제로) 확인하는 함수
        """
        basePattern = self.patternList[pattern1]
        checkPattern = self.patternList[pattern2]
        pattern1MinPos = (0, 0, 0)
        pattern1MaxPos = (self.patternX - 1, self.patternY - 1, self.patternZ - 1)
        pattern2MinPos = offset
        pattern2MaxPos = (self.patternX + offset[0] - 1, self.patternY + offset[1] - 1, self.patternZ + offset[2] - 1)
        if not self.checkCuboidOverlap(pattern1MinPos, pattern1MaxPos, pattern2MinPos, pattern2MaxPos):
            return False
        for i in range(self.patternX):
            for j in range(self.patternY):
                for k in range(self.patternZ):
                    p, q, r = i + offset[0], j + offset[1], k + offset[2]
                    if p < 0 or p >= self.patternX or q < 0 or q >= self.patternY or r < 0 or r >= self.patternZ:
                        continue
                    if basePattern.state[p][q][r] != checkPattern.state[i][j][k]:
                        return False
        return True
    
    def checkCuboidOverlap(self, minPosA:tuple[int, int, int], maxPosA:tuple[int, int, int],
                            minPosB:tuple[int, int, int], maxPosB:tuple[int, int, int]) -> bool:
        overlapX = (minPosA[0] < maxPosB[0] and minPosB[0] < maxPosA[0])
        overlapY = (minPosA[1] < maxPosB[1] and minPosB[1] < maxPosA[1]) or (maxPosA[1] == maxPosB[1] and minPosB[1] == minPosA[1])
        overlapZ = (minPosA[2] < maxPosB[2] and minPosB[2] < maxPosA[2])
        return overlapX and overlapY and overlapZ
    
    def getAbsoluteCoord(self, x:int, y:int, z:int, i:int, j:int, k:int) -> tuple[int, int, int]:
        return x + i - self.patternX // 2, y + j - self.patternY // 2, z + k - self.patternZ // 2
    
    def extractPatterns(self, patternShape:tuple[int, int, int], inputData:np.ndarray) -> list["Pattern"]:
        """
            주어진 3차원 데이터로부터 패턴을 추출하는 함수
        """
        if len(inputData.shape) != 3 or len(patternShape) != 3:
            raise ValueError("Dimension of data must be 3.")
        
        self.patternX, self.patternY, self.patternZ = patternShape
        self.dimX, self.dimY, self.dimZ = inputData.shape
        self.inputData = inputData
        self.hashPower = np.unique(inputData).shape[0] + 1
        
        patternList = []
        patternHashTable = dict()
        
        def getPattern(x, y, z) -> np.ndarray | None:
            pattern = np.zeros((self.patternX, self.patternY, self.patternZ), dtype=int)
            for i in range(self.patternX):
                for j in range(self.patternY):
                    for k in range(self.patternZ):
                        p, q, r = self.getAbsoluteCoord(x, y, z, i, j, k)
                        if p < 0 or p >= self.dimX or q < 0 or q >= self.dimY or r < 0 or r >= self.dimZ:
                            return None
                        pattern[i][j][k] = self.inputData[p][q][r]
            return pattern
        
        def getRotatedPatterns(pattern:np.ndarray) -> list[np.ndarray]:
            """
                주어진 패턴을 Y축 회전 및 반전한 패턴 리스트를 반환하는 함수
            """
            result = []
            for i in range(4):
                result.append(np.rot90(pattern, i, (0, 2)))
            pattern = np.flip(pattern, 1)
            for i in range(4):
                result.append(np.rot90(pattern, i, (0, 2)))
            return result
        
        def registerPattern(pattern:np.ndarray, patternList:list["Pattern"], patternHashTable:dict[int, int]):
            hashValue = self.patternHash(pattern)
            if hashValue in patternHashTable:
                patternList[patternHashTable[hashValue]].weight += 1
                return
            patternList.append(Pattern(pattern, pattern[self.patternX // 2][self.patternY // 2][self.patternZ // 2], 1))
            patternHashTable[hashValue] = len(patternList) - 1
        
        for i in range(self.dimX):
            for j in range(self.dimY):
                for k in range(self.dimZ):
                    pattern = getPattern(i, j, k)
                    if pattern is None: continue
                    for p in getRotatedPatterns(pattern):
                        registerPattern(p, patternList, patternHashTable)
        return patternList
    
    def patternHash(self, pattern:np.ndarray) -> int:
        hashValue = 0
        power = 1
        for i in range(self.patternX):
            for j in range(self.patternY):
                for k in range(self.patternZ):
                    hashValue = (hashValue + power * int(pattern[i][j][k])) % HASH_MODULO
                    power *= self.hashPower % HASH_MODULO
        return hashValue
    
    def getPatternList(self) -> list["Pattern"]:
        return self.patternList

class Pattern:
    def __init__(self, array = [], centerValue = -1, weight = 0):
        self.state = deepcopy(array)
        self.centerValue = centerValue
        self.weight = weight
    
    def __call__(self):
        return self
    
    def getCenter(self):
        return self.centerValue