import numpy as np
import random, math, time
from constantTable import NEUMANN_NEIGHBOR_3D
from datetime import datetime
from heuristicConfig import WFCHeuristic
from pyvistaVisualizer import pyvistaVisualizer
from patternIndex import PatternIndex, Pattern

def printExecutionTime(func):
    def decorated(*args, **kwargs):
        timestamp = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {round(time.time() - timestamp, 6):.06f}s")
        return result
    return decorated

class WaveFunctionCollapse3D():
    def __init__(self, inputData:np.ndarray, dataShape:tuple[int, int, int], patternShape:tuple[int, int, int] = (3, 3, 3)):
        """
            inputData: 유사한 데이터를 생성할 초기 3차원 데이터
            
            dataShape: 생성할 데이터 형태(3차원)
            
            patternShape: 패턴의 형태(3차원)
        """
        if len(dataShape) != 3 or len(patternShape) != 3:
            raise ValueError("Data shape and pattern shape must be 3-dimensional.")
        self.dimX, self.dimY, self.dimZ = dataShape
        self.patternX, self.patternY, self.patternZ = patternShape
        self.inputData = inputData
        self.initialWave = np.array([[[-1 for _ in range(self.dimZ)] for _ in range(self.dimY)] for _ in range(self.dimX)], dtype=int)
        self.updateStack = []
        
        self.patternIndex, self.patternList = self.getPatternIndex(patternShape, inputData)

        """
            compatibiePatternsCount[i][j][k][l][(p, q, r)]
            -> 좌표 (i, j, k)에서 l번째 패턴에 대해 (p, q, r) 방향으로 인접할 수 있는 패턴의 개수
        """
        self.compatiblePatternsCount = [[[
            [{coord:len(self.patternIndex.index[l][coord]) for coord in NEUMANN_NEIGHBOR_3D} for l in range(len(self.patternList))]
        for k in range(self.dimZ)] for j in range(self.dimY)] for i in range(self.dimX)]
    
    def run(self, heuristic = WFCHeuristic.Entropy, debug:bool = False) -> np.ndarray:
        self.heuristic = heuristic
        self.debug = debug
        
        if self.debug:
            self.debugFrames = []
            self.debugCursorPos = []
        
        # Entropy 휴리스틱 전용
        self.unobservedNodes = {(i, j, k) for i in range(self.dimX) for j in range(self.dimY) for k in range(self.dimZ)}
        
        # Scanline 휴리스틱 전용
        self.scanlineLastIndex = -1
        self.scanlineInvLastIndex = self.dimX * self.dimY * self.dimZ
        self.scanlineLoopCheck = 0
        
        self.waveFunctionInitializer()
        
        count = 0
        
        while True:
            check = self.WFCStep()
            count += 1
            #print(count)
            if check == False:
                break
        
        if self.debug:
            dt = datetime.now().strftime(r'%Y%m%d-%H%M%S')
            debugFrameArray = np.array(self.debugFrames)
            debugCursorPosArray = np.array(self.debugCursorPos)
            visualizer = pyvistaVisualizer(colorMode=1)
            visualizer.renderAnimWithCursor(debugFrameArray, debugCursorPosArray, colorSeed=0, directory="./Debug/Visualization/", filename=f"debug-{dt}.mp4", fps=30)
        
        self.outputData = np.array([[[self.waveFunction[i][j][k].value for k in range(self.dimZ)] for j in range(self.dimY)] for i in range(self.dimX)], dtype=int)
        checkContradiction = any([not self.waveFunction[i][j][k].possibleStates for i in range(self.dimX) for j in range(self.dimY) for k in range(self.dimZ)])
        if checkContradiction:
            print("Contradiction detected.")
        return self.outputData

    def WFCStep(self) -> bool:
        x, y, z = self.findNextNode()
        
        if x == -1 or y == -1 or z == -1:
            return False
        else:
            self.unobservedNodes.discard((x, y, z))
            self.observe(x, y, z)
            check = self.propagate()
        
        if self.debug:
            self.debugWriteFrame(cursorPos=(x, y, z))
        return check
    
    def observe(self, x:int, y:int, z:int) -> None:
        """
            노드 관측을 수행하는 함수
        """
        collapsedPattern = self.waveFunction[x][y][z].collapse()
        #print(f"Node ({x:>3}, {y:>3}, {z:>3}) -> Pattern {collapsedPattern}")
        for state in [i for i in self.waveFunction[x][y][z].possibleStates]:
            if state == collapsedPattern: continue
            self.removeState(x, y, z, state)
    
    def propagate(self) -> bool:
        """
            관측으로 인해 업데이트된 정보를 전파하는 함수
        """
        while self.updateStack:
            x, y, z, removedPattern = self.updateStack.pop()
            for i, j, k in NEUMANN_NEIGHBOR_3D:
                p, q, r = x + i, y + j, z + k
                if p < 0 or q < 0 or r < 0 or p >= self.dimX or q >= self.dimY or r >= self.dimZ: continue
                """
                    removedPattern에 해당되는 패턴이 현재 노드에서 삭제됨
                    ->  현재 노드와 인접한 노드에서 removedPattern과 인접할 수 있었던 모든 패턴들에 대해
                        현재 노드 방향으로 인접할 수 있는 패턴의 개수가 1개씩 감소함
                    
                    ->  만약 이 수가 0이 되었을 경우, 인접 노드의 중첩 상태에 속한 패턴 중
                        removedPattern 때문에 인접할 수 있었던 패턴은 더 이상 현재 노드와 인접할 수 없게 됨
                    
                    ->  이 경우 해당 패턴이 존재할 수 없으므로, 인접 노드에서 해당 패턴 삭제
                """
                for compatiblePattern in self.patternIndex.index[removedPattern][(i, j, k)]:
                    reversedCoord = (-i, -j, -k)
                    self.compatiblePatternsCount[p][q][r][compatiblePattern][reversedCoord] -= 1
                    if self.compatiblePatternsCount[p][q][r][compatiblePattern][reversedCoord] == 0:
                        self.removeState(p, q, r, compatiblePattern)
            if not self.waveFunction[x][y][z].possibleStates:
                print(f"Contradiction detected at ({x}, {y}, {z})")
                return False
        return True
    
    def removeState(self, x:int, y:int, z:int, patternID:int) -> None:
        """
            노드의 중첩 상태에서 patternID에 해당하는 패턴을 제거하는 함수
        """
        for coord in NEUMANN_NEIGHBOR_3D:
            self.compatiblePatternsCount[x][y][z][patternID][coord] = 0
        self.updateStack.append((x, y, z, patternID))
        self.waveFunction[x][y][z].removeState(patternID)
    
    @printExecutionTime
    def getPatternIndex(self, patternShape:tuple[int, int, int], inputData:np.ndarray) -> tuple[PatternIndex, list["Pattern"]]:
        patternIndex = PatternIndex()(patternShape, inputData)
        return patternIndex, patternIndex.getPatternList()
    
    def setInitialWave(self, array:np.ndarray) -> None:
        """
            파동함수 초기 상태를 설정하는 함수
        """
        self.initialWave = np.array([[[array[i][j][k] for k in range(self.dimZ)] for j in range(self.dimY)] for i in range(self.dimX)], dtype=int)
    
    @printExecutionTime
    def waveFunctionInitializer(self) -> None:
        """
            파동함수 초기화 및 초기 상태를 반영하는 함수
        """
        self.waveFunction = np.array([[[Node(self.patternList) for _ in range(self.dimZ)] for _ in range(self.dimY)] for _ in range(self.dimX)], dtype=Node)
        
        # 초기 상태가 지정된 좌표의 노드 값 초기화
        for i in range(self.dimX):
            for j in range(self.dimY):
                for k in range(self.dimZ):
                    if self.initialWave[i][j][k] > -1:
                        self.waveFunction[i][j][k].value = self.initialWave[i][j][k]
                        for state in [s for s in self.waveFunction[i][j][k].possibleStates]:
                            if self.patternList[state].getCenter() != self.initialWave[i][j][k]:
                                self.removeState(i, j, k, state)
        
        # 초기화 이후 모든 노드의 중첩 상태 업데이트
        self.propagate()
    
    def isAvailableNode(self, node:"Node") -> bool:
        return node.collapsed == 0 and node.isContradicted() == False and node.weightSum > 0
    
    def findNextNode(self) -> tuple[int, int, int]:
        def minEntropy():
            nodes = []
            for i, j, k in self.unobservedNodes:
                if self.isAvailableNode(self.waveFunction[i][j][k]):
                    entropy, wsum = self.waveFunction[i][j][k].entropy, self.waveFunction[i][j][k].weightSum
                    nodes.append((entropy, wsum, i, j, k))
            nodes.sort(key=lambda x: x[0])
            for entropy, wsum, x, y, z in nodes:
                if wsum > 0: return x, y, z
            return -1, -1, -1
        
        def scanline():
            # 정방향 Scanline
            index = self.scanlineLastIndex + 1
            if self.scanlineLoopCheck == 0:
                for idx in range(index, self.dimX * self.dimY * self.dimZ):
                    y = idx // (self.dimX * self.dimZ)
                    x = (idx % (self.dimX * self.dimZ)) // self.dimZ
                    z = idx % self.dimZ
                    yInv = self.dimY - y - 1
                    if self.waveFunction[x][y][z].collapsed == 0 and self.waveFunction[x][y][z].isContradicted() == False:
                        self.scanlineLastIndex = idx
                        return (x, yInv, z)
                else:
                    self.scanlineLoopCheck = 1

            # 역방향 Scanline
            indexInv = self.scanlineInvLastIndex - 1
            for idx in range(indexInv, -1, -1):
                y = idx // (self.dimX * self.dimZ)
                x = (idx % (self.dimX * self.dimZ)) // self.dimZ
                z = idx % self.dimZ
                yInv = self.dimY - y - 1
                if self.waveFunction[x][y][z].collapsed == 0 and self.waveFunction[x][y][z].isContradicted() == False:
                    self.scanlineInvLastIndex = idx
                    return (x, yInv, z)
            return (-1, -1, -1)

        if self.heuristic == WFCHeuristic.Scanline:
            return scanline()
        
        return minEntropy()
    
    def debugWriteFrame(self, cursorPos:tuple[int, int, int]) -> None:
        if not self.debug: return
        frame = np.array([[[self.waveFunction[i][j][k].value for k in range(self.dimZ)] for j in range(self.dimY)] for i in range(self.dimZ)], dtype=int)
        cursor = np.array(cursorPos, dtype=int)
        self.debugFrames.append(frame)
        self.debugCursorPos.append(cursor)
    
    def savePatternImages(self, patternList:list[Pattern]) -> None:
        visualizer = pyvistaVisualizer(colorMode=1)
        for i, pattern in enumerate(patternList):
            visualizer.renderImage(pattern.state, [], colorSeed=0, directory="./Debug/Visualization/Patterns/", filename=f"{i}-{pattern.weight}.png")

class Node:
    def __init__(self, patternList:list["Pattern"]):
        self.patternList = patternList
        self.possibleStates = {i for i in range(len(patternList))}
        
        self.value = -1
        self.patternID = -1
        self.collapsed = 0
        self.initializeEntropy()
    
    def isContradicted(self) -> bool:
        return self.weightSum == 0 and self.collapsed == 0
    
    def initializeEntropy(self) -> None:
        weightSum = 0
        weightSumLog = 0
        entropy = 0
        for i in self.possibleStates:
            w = self.patternList[i].weight
            weightSum += w
            weightSumLog += w * math.log2(w)
        if weightSum > 0:
            entropy = math.log2(weightSum) - weightSumLog / weightSum
        else:
            entropy = -1
        self.weightSum = weightSum
        self.weightLogSum = weightSumLog
        self.entropy = entropy
    
    def updateEntropy(self, removedPattern:int) -> None:
        """
            노드에서 패턴이 제거될 때 엔트로피를 업데이트하는 함수
        """
        self.weightSum -= self.patternList[removedPattern].weight
        self.weightLogSum -= self.patternList[removedPattern].weight * math.log2(self.patternList[removedPattern].weight)
        if self.weightSum > 0:
            self.entropy = math.log2(self.weightSum) - self.weightLogSum / self.weightSum
        else:
            self.entropy = -1
    
    def collapse(self) -> int:
        """
            현재 노드의 중첩 상태를 하나로 붕괴시키는 함수
        """
        if self.collapsed == 1: return

        # 현재 노드의 중첩 상태를 하나의 상태로 붕괴
        possiblePatternID = list(self.possibleStates)
        patternWeights = [self.patternList[i].weight for i in possiblePatternID]
        
        if len(possiblePatternID) == 0: return
        collapsedID = int(random.choices(possiblePatternID, weights=patternWeights)[0])
        
        # 현재 노드의 상태를 붕괴된 상태로 설정
        self.patternID = collapsedID
        self.collapsed = 1
        self.value = self.patternList[collapsedID].centerValue
        return collapsedID
    
    def removeState(self, patternID:int) -> None:
        """
            현재 노드의 중첩 상태에서 특정 패턴을 제거하는 함수.
            
            patternID: 제거할 패턴의 ID
        """
        if patternID in self.possibleStates:
            self.possibleStates.discard(patternID)
            self.updateEntropy(patternID)
    
    def revertObservation(self) -> None:
        """
            붕괴된 노드의 상태를 되돌리는 함수
        """
        self.possibleStates = {i for i in range(len(self.patternList))}
        self.collapsed = 0
        self.patternID = -1
        self.value = -1
        self.initializeEntropy()