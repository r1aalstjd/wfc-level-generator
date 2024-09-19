import pyvista as pv
import numpy as np
import random, json

class pyvistaVisualizer():
    def __init__(self, colorMode:int = 0):
        self.array = None
        self.colorMode = colorMode
        self.blockColorData = json.load(open('blockColorData.json', 'r'))
        self.randomColorData = dict()
    
    def renderPlot(self, array:np.ndarray, registryInv:list=[], colorMode:int = 1, colorSeed:int = 0) -> None:
        """
            주어진 3차원 데이터를 인터랙티브 시각화하는 함수
        """
        self.colorMode = colorMode
        np.random.seed(colorSeed)
        self.plot = self.createPlot(array, registryInv, headless=False)
        self.plot.show()
    
    def renderImage(self, array:np.ndarray, registryInv:list=[], colorMode:int=1, colorSeed:int = 0, directory:str = "./Debug/Visualization/", filename:str = "output.png", ) -> None:
        self.colorMode = colorMode
        np.random.seed(colorSeed)
        self.plot = self.createPlot(array, registryInv, headless=True)
        self.plot.screenshot(directory + filename)
    
    def getColor(self, value:int) -> tuple[float, float, float, float]:
        """
            colorMode 0: 주어진 블록 레지스트리 값에 해당하는 색상 반환
            colorMode 1: 랜덤 색상 팔레트를 생성해 반환
            value == -2: 커서 위치 표시용 색상
        """
        if value == -2:
            return (1.0, 0.0, 0.0, 0.5)
        if self.colorMode == 1:
            if value not in self.randomColorData:
                self.randomColorData[value] = (random.random(), random.random(), random.random(), 1.0)
            return self.randomColorData[value]
        else:
            return self.blockColorData[self.registryInv[value]]
    
    def createPlot(self, array:np.ndarray, registryInv:list, headless:bool = True) -> None:
        arrayShape = array.shape
        if len(arrayShape) != 3:
            raise ValueError("Invalid array shape detected.")
        
        self.array = array
        self.dimX, self.dimY, self.dimZ = self.array.shape

        self.array = array
        self.registryInv = registryInv
        
        self.plot = pv.Plotter(off_screen=headless)
        self.plot.camera_position = [(-(self.dimX), self.dimY*2, -(self.dimZ)), (self.dimX/2, self.dimY/2, self.dimZ/2), (0, 1, 0)]
        self.plot.set_background('#ffffff')
        self.addAxes(self.plot)
        
        for x in range(self.dimX):
            for y in range(self.dimY):
                for z in range(self.dimZ):
                    value = self.array[x][y][z]
                    if value <= 0: continue
                    color = self.getColor(value)
                    self.addCube(self.plot, [x, y, z], color)
        
        return self.plot
    
    def renderAnim(self, array:np.ndarray, registryInv:list=[], colorMode:int=1, colorSeed:int = 0, directory:str="./Debug/Visualization/", filename:str="output.mp4", fps:int=30, headless:bool=True) -> None:
        self.colorMode = colorMode
        np.random.seed(colorSeed)
        arrayShape = array.shape
        if len(arrayShape) != 4:
            raise ValueError("Invalid array shape detected.")
        
        self.array = array
        self.registryInv = registryInv
        self.dimT, self.dimX, self.dimY, self.dimZ = self.array.shape
        
        cubeMeshArray = [[[None for _ in range(self.dimZ)] for _ in range(self.dimY)] for _ in range(self.dimX)]
        differenceArray = np.zeros((self.dimT, self.dimX, self.dimY, self.dimZ), dtype=np.int8)
        for t in range(self.dimT):
            for x in range(self.dimX):
                for y in range(self.dimY):
                    for z in range(self.dimZ):
                        if t == 0:
                            differenceArray[t][x][y][z] = self.array[t][x][y][z]
                        else:
                            if self.array[t][x][y][z] != self.array[t-1][x][y][z]:
                                differenceArray[t][x][y][z] = self.array[t][x][y][z]
        
        self.plot = pv.Plotter(window_size=([640, 480]), off_screen=headless)
        self.plot.camera_position = [(-(self.dimX), self.dimY*2, -(self.dimZ)), (self.dimX//2, self.dimY//2, self.dimZ//2), (0, 1, 0)]
        self.plot.set_background('#ffffff')
        self.addAxes(self.plot)
        title = self.plot.add_text(f"Time Step: {0:>05}", position='lower_edge', font_size=10, color='black', font='times')
        self.plot.open_movie(directory + filename, fps)
        
        # t = 0일 때 Mesh 생성
        for x in range(self.dimX):
            for y in range(self.dimY):
                for z in range(self.dimZ):
                    value = self.array[0][x][y][z]
                    if value <= 0: continue
                    color = self.getColor(value)
                    cubeMeshArray[x][y][z] = self.addCube(self.plot, (x, y, z), color)
        
        # t >= 1일 때, Mesh 중 변화가 있는 부분만 업데이트
        for t in range(1, self.dimT):
            self.plot.remove_actor(title)
            title = self.plot.add_text(f"Time Step: {t:>05}", position='lower_edge', font_size=10, color='black', font='times')
            for x in range(self.dimX):
                for y in range(self.dimY):
                    for z in range(self.dimZ):
                        if differenceArray[t][x][y][z] > 0:
                            value = self.array[t][x][y][z]
                            if value <= 0: continue
                            color = self.getColor(value)
                            if cubeMeshArray[x][y][z] != None:
                                self.plot.remove_actor(cubeMeshArray[x][y][z])
                            cubeMeshArray[x][y][z] = self.addCube(self.plot, (x, y, z), color)
            self.plot.write_frame()
        
        self.plot.close()
    
    def renderAnimWithCursor(self, array:np.ndarray, cursorArray:np.ndarray, registryInv:list=[],
                            colorMode:int=1, colorSeed:int = 0,
                            directory:str="./Debug/Visualization/", filename:str="output.mp4",
                            fps:int=30, headless:bool=True) -> None:
        """
            [[3DArray, (cursorX, cursorY, cursorZ)], ...] 형태의 데이터 시각화 함수
        """
        self.colorMode = colorMode
        np.random.seed(colorSeed)
        arrayShape = array.shape
        if len(arrayShape) != 4:
            raise ValueError("Invalid array shape detected.")
        
        self.array = array
        self.registryInv = registryInv
        self.dimT, self.dimX, self.dimY, self.dimZ = self.array.shape
        
        cubeMeshArray = [[[None for _ in range(self.dimZ)] for _ in range(self.dimY)] for _ in range(self.dimX)]
        differenceArray = np.array([[[[-1 for _ in range(self.dimZ)] for _ in range(self.dimY)] for _ in range(self.dimX)] for _ in range(self.dimT)], dtype=int)
        for t in range(self.dimT):
            for x in range(self.dimX):
                for y in range(self.dimY):
                    for z in range(self.dimZ):
                        if t == 0:
                            differenceArray[t][x][y][z] = self.array[t][x][y][z]
                        else:
                            if self.array[t][x][y][z] != self.array[t-1][x][y][z]:
                                differenceArray[t][x][y][z] = self.array[t][x][y][z]
        
        self.plot = pv.Plotter(window_size=([640, 480]), off_screen=headless)
        self.plot.camera_position = [(-(self.dimX), self.dimY*2, -(self.dimZ)), (self.dimX//2, self.dimY//2, self.dimZ//2), (0, 1, 0)]
        self.plot.set_background('#ffffff')
        self.addAxes(self.plot)
        title = self.plot.add_text(f"Time Step: {0:>05}", position='lower_edge', font_size=10, color='black', font='times')
        cursorActor = self.addCube(self.plot, cursorArray[0], self.getColor(-2))
        self.plot.open_movie(directory + filename, fps)
        
        # Mesh 중 변화가 있는 부분만 업데이트
        for t in range(self.dimT):
            self.plot.remove_actor(title)
            title = self.plot.add_text(f"Time Step: {t:>05}", position='lower_edge', font_size=10, color='black', font='times')
            for x in range(self.dimX):
                for y in range(self.dimY):
                    for z in range(self.dimZ):
                        if differenceArray[t][x][y][z] != -1:
                            value = differenceArray[t][x][y][z]
                            if value == 0: continue
                            color = self.getColor(value)
                            if cubeMeshArray[x][y][z] != None:
                                self.plot.remove_actor(cubeMeshArray[x][y][z])
                            cubeMeshArray[x][y][z] = self.addCube(self.plot, (x, y, z), color)
            self.plot.remove_actor(cursorActor)
            cursorActor = self.addCube(self.plot, cursorArray[t], self.getColor(-2))
            self.plot.write_frame()
        
        self.plot.close()
    
    def addCube(self, plot, coord:tuple[int, int, int], color:tuple[float, float, float, float]) -> pv.PolyData:
        x, y, z = coord
        cubeShape = pv.Box(bounds=(x, x + 1, y, y + 1, z, z + 1))
        return plot.add_mesh(cubeShape, color=color[0:3], opacity=color[3], show_edges=True)
    
    def addAxes(self, plot) -> None:
        self.plot.add_axes()
        xAxis = pv.Line((0, 0, 0), (self.dimX, 0, 0))
        yAxis = pv.Line((0, 0, 0), (0, self.dimY, 0))
        zAxis = pv.Line((0, 0, 0), (0, 0, self.dimZ))
        plot.add_mesh(xAxis, color='#FF0000', opacity=1.0)
        plot.add_mesh(yAxis, color='#00FF00', opacity=1.0)
        plot.add_mesh(zAxis, color='#0000FF', opacity=1.0)