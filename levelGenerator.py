import anvil, time, random, numpy as np, os, sys
import mcaFileIO
from copy import deepcopy
from constantTable import SIZE_X, SIZE_Y, SIZE_Z, DX_3X3, DZ_3X3, MATRIX_X, MATRIX_Y, MATRIX_Z, INPUT_DIR, OUTPUT_DIR, ENTRY_COORDS, NEUMANN_NEIGHBOR_3D
from WaveFunctionCollapse.WaveFunctionCollapse3D import WaveFunctionCollapse3D
from pyvistaVisualizer import pyvistaVisualizer

class levelGenerator():
    def __init__(self, world):
        beginTime = time.time()
        self.dimX = SIZE_X
        self.dimY = mcaFileIO.getMaxY(world)
        self.dimZ = SIZE_Z
        
        self.world = world
        
        self.inputCache = [[[-1 for _ in range(self.dimZ)] for _ in range(self.dimY)] for _ in range(self.dimX)]
        self.chunkCache = [[None for _ in range(SIZE_Z // 16 + 1)] for _ in range(SIZE_X // 16 + 1)]
        
        self.blockRegistry, self.blockRegistryInv = mcaFileIO.registerBlocks(self.dimX, self.dimY, self.dimZ, world, self.inputCache, self.chunkCache)
        print("Block Registration completed. (took {}s)".format(round(time.time() - beginTime, 6)))
        
        self.inputLevel = self.worldToArray()
        self.platformMask = self.getPlatformMask()
    
    def generateLevel(self) -> np.ndarray:
        initialState = np.array([[[-1 for _ in range(self.dimZ)] for _ in range(self.dimY)] for _ in range(self.dimX)], dtype=int)
        for x in range(self.dimX):
            for y in range(self.dimY):
                for z in range(self.dimZ):
                    if x == 0 or x == self.dimX - 1 or y == 0 or y == self.dimY - 1 or z == 0 or z == self.dimZ - 1:
                        initialState[x][y][z] = 0
        
        beginTime = time.time()
        wfcModel = WaveFunctionCollapse3D(self.platformMask, (self.dimX, self.dimY, self.dimZ), (3, 3, 3))
        wfcModel.setInitialWave(initialState)
        result = wfcModel.run(debug=False)
        print(f"Level Generation completed. (took {round(time.time() - beginTime, 6)}s)")
        
        visualizer = pyvistaVisualizer(colorMode=1)
        visualizer.renderPlot(result, [], colorMode=1, colorSeed=0)
    
    def worldToArray(self) -> np.ndarray:
        """
            월드를 3차원 배열로 변환하는 함수
        """
        inputLevel = np.zeros((self.dimX, self.dimY, self.dimZ), dtype=int)
        for x in range(self.dimX):
            for y in range(self.dimY):
                for z in range(self.dimZ):
                    inputLevel[x][y][z] = self.blockRegistry[mcaFileIO.getBlock(x, y, z, self.world, self.inputCache, self.chunkCache).id]
        return inputLevel
    
    def getAirMask(self) -> np.ndarray:
        """
            레벨 입구와 연결된 모든 공기 블록의 좌표 마스크를 반환
        """
        levelAirMask = np.zeros((self.dimX, self.dimY, self.dimZ), dtype=bool)
        airCheckVisit = np.zeros((self.dimX, self.dimY, self.dimZ), dtype=bool)
        
        px, py, pz = ENTRY_COORDS[0]
        stack = [(px, py, pz)]
        while stack:
            x, y, z = stack.pop()
            levelAirMask[x][y][z] = True
            airCheckVisit[x][y][z] = True
            for i in NEUMANN_NEIGHBOR_3D:
                p, q, r = x + i[0], y + i[1], z + i[2]
                if p < 0 or p >= self.dimX or q < 0 or q >= self.dimY or r < 0 or r >= self.dimZ:
                    continue
                if airCheckVisit[p][q][r] == False and self.inputLevel[p][q][r] == 0:
                    stack.append((p, q, r))
        return levelAirMask
    
    def getPlatformMask(self) -> np.ndarray:
        """
            입력 월드에서 플레이어가 올라설 수 있는 블록 좌표 마스크를 추출하는 함수.
        """
        platformMask = np.zeros((self.dimX, self.dimY, self.dimZ), dtype=int)        
        levelAirMask = self.getAirMask()
        
        def canStandOn(x, y, z) -> bool:
            """
                (x, y, z) 좌표에 블록이 있고, 그 위에 최소 3칸의 빈 공간이 있을 때 True를 반환
            """
            if y == 0 or self.inputLevel[x][y][z] <= 0:
                return False
            cnt = 0
            for i in range(1, 4):
                if y + i >= self.dimY:
                    return False
                if levelAirMask[x][y + i][z] == True:
                    cnt += 1
            return cnt >= 3
        
        def isBridge(x, y, z) -> bool:
            """
                (x, y, z) 좌표가 다리인지 판별하는 함수
            """
            if y == 0 or self.inputLevel[x][y][z] <= 0:
                return False
            pos = y
            while pos > 0:
                if levelAirMask[x][pos][z] == True:
                    return True
                pos -= 1
            return False
        
        for x in range(self.dimX):
            for y in range(self.dimY):
                for z in range(self.dimZ):
                    if canStandOn(x, y, z) == True:
                        if isBridge(x, y, z) == False:
                            platformMask[x][y][z] = 3
                        else:
                            platformMask[x][y][z] = 1
        return platformMask

if __name__ == "__main__":
    inputWorld = anvil.Region.from_file(INPUT_DIR + 'r.0.0.mca')
    gen = levelGenerator(inputWorld)
    gen.generateLevel()

"""
    def bfs(x, y, z, idx):
            #플랫폼 연결 요소를 탐색하는 함수
        stack = [(x, y, z)]
        while stack:
            x, y, z = stack.pop()
            visitCheck[x][y][z] = True
            platformMask[x][y][z] = idx
            for i in range(8):
                p, q, r = x + DX_3X3[i], y, z + DZ_3X3[i]
                if p < 0 or p >= self.dimX or q < 0 or q >= self.dimY or r < 0 or r >= self.dimZ:
                    continue
                if visitCheck[p][q][r] == False and platformCheck[p][q][r] == True:
                    stack.append((p, q, r))
"""