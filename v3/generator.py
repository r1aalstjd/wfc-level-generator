import random, os, sys, json
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from Structures.Platform import Platform
from pyvistaVisualizer import pyvistaVisualizer

tempConfig = {
    "dim_x": 27,
    "dim_y": 15,
    "dim_z": 27,
    "floorRange": 3,
    "floorDistance": 4,
    "doorLevel": 0,
    "platformSize": 3
}

class Generator():
    def __init__(self) -> None:
        self.dim_x = tempConfig["dim_x"]
        self.dim_y = tempConfig["dim_y"]
        self.dim_z = tempConfig["dim_z"]
        self.structures = dict()
        self.structureCount = 0
        self.floors = []
    
    def setDim(self, x:int, y:int, z:int) -> None:
        self.dim_x = x
        self.dim_y = y
        self.dim_z = z
    
    def run(self, config:dict, seed=None) -> None:
        if seed != None:
            self.random = random.Random(seed)
        else:
            self.random = random.Random()
        self.config = config
        self.blockID = json.load(open('blockID.json', 'r'))
        self.level = self.initLevel()
        self.structureMap = [[[0 for _ in range(self.dim_z)] for _ in range(self.dim_y)] for _ in range(self.dim_x)]
        
        # 층 구분 생성
        self.floors.append((self.config["doorLevel"], self.config["doorLevel"]))
        floorIndex = self.config["doorLevel"]
        while floorIndex < self.dim_y - (self.config["floorDistance"] + self.config["floorRange"] + 3):
            floorIndex += self.config["floorDistance"] + self.config["floorRange"]
            self.floors.append((floorIndex - self.config["floorRange"], floorIndex))
        
        for i in range(len(self.floors)):
            if i == 0: continue
            self.generateFloor(i)
        return self.level
    
    def initLevel(self) -> list[list[list[int]]]:
        return [[[-1 for _ in range(self.dim_z)] for _ in range(self.dim_y)] for _ in range(self.dim_x)]
    
    def generateFloor(self, floorIndex:int) -> None:
        """
            주어진 번호의 층에 구조물을 배치하는 함수
        """
        floorBottom, floorTop = self.floors[floorIndex]
        platformPadding = self.config["platformSize"] // 2
        posList = []
        
        # 첫 번째 플랫폼을 생성할 basePos 결정
        for y in range(floorBottom, floorTop):
            for x in range(platformPadding, self.dim_x - platformPadding):
                for z in range(platformPadding, self.dim_z - platformPadding):
                    if self.level[x][y][z] == -1:
                        posList.append((x, y, z))
        basePos = posList[self.random.randint(0, len(posList) - 1)]
        basePos1 = (basePos[0] - platformPadding, basePos[1], basePos[2] - platformPadding)
        basePos2 = (basePos[0] + platformPadding, basePos[1], basePos[2] + platformPadding)
        self.addPlatform(basePos1, basePos2)
    
    def placeNextStructure(self, structure:"Structure", direction:int) -> None:
    
    def addPlatform(self, pos1:tuple[int], pos2:tuple[int]) -> None:
        """
            플랫폼 구조물 생성 후 구조물 딕셔너리에 추가
        """
        self.structureCount += 1
        platform = Platform(self.structureCount, pos1, pos2)
        self.structures[self.structureCount] = platform
        for x in range(min(pos1[0], pos2[0]), max(pos1[0], pos2[0]) + 1):
            for y in range(min(pos1[1], pos2[1]), max(pos1[1], pos2[1]) + 1):
                for z in range(min(pos1[2], pos2[2]), max(pos1[2], pos2[2]) + 1):
                    if x < 0 or x >= self.dim_x or y < 0 or y >= self.dim_y or z < 0 or z >= self.dim_z:
                        continue
                    self.level[x][y][z] = self.blockID["PLATFORM"]
                    self.structureMap[x][y][z] = self.structureCount

if __name__ == "__main__":
    generator = Generator()
    level = generator.run(tempConfig, seed=0)
    pyvistaVisualizer(colorMode=1).renderPlot(np.array(level))