import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

class animatedVisualizer:
    """
        frames:             시각화 배열 정보
        
        blockRegistryInv:   블록 ID에 대응하는 블록 이름
        
        interval:           각 프레임 간 시간 간격 (ms)
        
        rotH:               시각화 회전 각도 (수평)
        
        rotV:               시각화 회전 각도 (수직)
        
        dir:                저장할 디렉토리
        
        filename:           저장할 파일 이름
    """
    def __init__(self, frameData:list[tuple[list[list[list[int]]], int, int, int]], blockRegistryInv:list[str], dir, filename='animated.gif', 
                interval = 50, elev = 0, azim = 0, roll = 0):
        self.frameCount = len(frameData)
        self.dim_x = len(frameData[0][0])
        self.dim_y = len(frameData[0][0][0])
        self.dim_z = len(frameData[0][0][0][0])
        self.graphDim = max(self.dim_x, self.dim_y, self.dim_z)
        self.elev = elev
        self.azim = azim
        self.roll = roll
        
        self.frameData = frameData
        self.blockRegistryInv = blockRegistryInv
        self.blockCount = len(blockRegistryInv)
        
        self.cubeVertices = np.array([
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        
        # 각 꼭짓점으로 결정되는 면
        self.cubeFaces = [
            [0, 1, 2, 3],  # 앞면
            [4, 5, 6, 7],  # 뒷면
            [0, 4, 7, 3],  # 오른쪽 면
            [1, 5, 6, 2],  # 왼쪽 면
            [0, 1, 5, 4],  # 윗면
            [3, 2, 6, 7]   # 아랫면
        ]
        
        self.blockColors = {
            'air':              (0, 0, 0, 0),           # 투명
            'bedrock':          (0, 0, 0, 0.5),         # 검은색
            
            'black_wool':       (0, 0, 0, 1),           # 검은색
            'gray_wool':        (0.5, 0.5, 0.5, 1),     # 회색
            'light_gray_wool':  (0.25, 0.25, 0.25, 1),  # 연회색
            'white_wool':       (1, 1, 1, 1),           # 흰색
            
            'orange_wool':      (1, 0.5, 0, 1),         # 주황색
            'yellow_wool':      (1, 1, 0, 1),           # 노란색
            
            'light_blue_wool':  (0, 0.75, 1, 1),        # 하늘색
            'green_wool':       (0, 1, 0, 1),           # 초록색
            'lime_wool':        (0.5, 1, 0, 1)          # 연두색
        }
        
        self.filterColors = {
            -1: (0, 0, 0, 0),  # 투명
            -2: (0, 0, 0, 1),  # 검은색
            -3: (1, 0, 0, 0.1),  # 빨간색
            -4: (0.5, 0.5, 0.5, 0.1),  # 초록색
            -5: (1, 1, 1, 0.1)   # 파란색
        }

        # 프레임 설정
        # https://matplotlib.org/3.6.0/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D.view_init
        fig = plt.figure(facecolor='#cccccc')
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=self.elev, azim=self.azim, roll=self.roll, vertical_axis='y')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        anim = FuncAnimation(fig, self.drawFrame, frames=self.frameCount, interval=interval)
        anim.save(dir + filename, writer='pillow')
        plt.show()
    
    def drawFrame(self, frame:int):
        self.ax.clear()
        self.ax.patch.set_facecolor('#cccccc')
        self.ax.set_axis_off()
        self.ax.set_aspect('equal')
        self.ax.set_xlim([0, self.graphDim])
        self.ax.set_ylim([0, self.graphDim])
        self.ax.set_zlim([0, self.graphDim])
        
        #self.ax.view_init(elev=self.rotV, azim=self.rotH)
        
        frameData = np.array(self.frameData[frame][0])
        posX = self.frameData[frame][1]
        posY = self.frameData[frame][2]
        posZ = self.frameData[frame][3]
        blockPos = np.empty((frameData.shape), dtype=object)
        blockColors = np.empty((self.dim_x, self.dim_y, self.dim_z, 4), dtype=object)
        
        for block_id in range(self.blockCount):
            blockPosArray = np.where(frameData == block_id)
            if self.blockRegistryInv[block_id] == 'air': continue
            blockPos[blockPosArray] = True
            blockColors[blockPosArray] = self.getBlockColor(block_id)
        
        blockPos[posX, posY, posZ] = True
        blockColors[posX, posY, posZ] = (1, 0, 0, 0.5)
        
        self.ax.voxels(blockPos, facecolors=blockColors, edgecolor='k')
        
        self.ax.plot([0, self.graphDim], [0, 0], [0, 0], color='red')    # X축
        self.ax.plot([0, 0], [0, self.graphDim], [0, 0], color='green')  # Y축
        self.ax.plot([0, 0], [0, 0], [0, self.graphDim], color='blue')   # Z축
        self.ax.text(s='({0}, {1}, {2}) / Step {3}'.format(posX, posY, posZ, frame-1), x=-0.5, y=-0.5, z=-0.5, color='black', fontsize=10, ha='left', va='top')
    
    @DeprecationWarning
    def placeCube(self, ax:plt.Axes, coord:list[int], blockColor):
        vertices = self.cubeVertices + coord

        # 정육면체 각 면 생성
        for i in range(len(self.cubeFaces)):
            face = [vertices[j] for j in self.cubeFaces[i]]
            ax.add_collection3d(Poly3DCollection([face], color=blockColor))
    
    def getBlockColor(self, block_id:int):
        if block_id >= 0:
            return self.blockColors[self.blockRegistryInv[block_id]]
        else:
            return (0, 0, 0, 0)
            #if block_id in self.filterColors:
            #    return self.filterColors[block_id]
            #else:
            #    return (0, 0, 0, 0)

if __name__ == '__main__':
    testData = [
        ([[[0, 0, 0, 0, 1]]],0, 0, 0),
        ([[[0, 0, 0, 1, 1]]],0, 0, 0),
        ([[[0, 0, 1, 1, 1]]],0, 0, 0),
        ([[[0, 1, 1, 1, 1]]],0, 0, 0),
        ([[[1, 1, 1, 1, 1]]],0, 0, 0),
    ]
    animatedVisualizer(testData, ['black_wool', 'bedrock'], './Debug/Visualization/', 'animator_test.gif', 1000, 22.5, -135, 0)