import random, os
from Structures import Platform

tempConfig = {
    "dim_x": 27,
    "dim_y": 15,
    "dim_z": 27,
    "floorRange": 3,
    "floorDistance": 4,
    "doorLevel": 0,
}

class Generator():
    def __init__(self) -> None:
        self.dim_x = tempConfig["dim_x"]
        self.dim_y = tempConfig["dim_y"]
        self.dim_z = tempConfig["dim_z"]
        self.structures = dict()
        self.floors = []
    
    def setDim(self, x:int, y:int, z:int) -> None:
        self.dim_x = x
        self.dim_y = y
        self.dim_z = z
    
    def run(self) -> None:
        self.level = self.initLevel()
        self.floors.append((tempConfig["doorLevel"], tempConfig["doorLevel"]))
        floorIndex = tempConfig["doorLevel"]
        while floorIndex < self.dim_y - 3:
            floorIndex += tempConfig["floorDistance"]
            self.floors.append((floorIndex, floorIndex + tempConfig["floorRange"]))
            floorIndex += tempConfig["floorRange"]
    
    def initLevel(self) -> list[list[list[int]]]:
        return [[[0 for _ in range(self.dim_z)] for _ in range(self.dim_y)] for _ in range(self.dim_x)]
    
    def generateFloor(self, floorIndex:int) -> None:
        floorBottom, floorTop = self.floors[floorIndex]
        
