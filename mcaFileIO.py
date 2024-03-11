import anvil, time
from copy import deepcopy
from constantTable import SIZE_X, SIZE_Y, SIZE_Z, MATRIX_X, MATRIX_Y, MATRIX_Z, OFF_X, OFF_Y, OFF_Z, INPUT_DIR, OUTPUT_DIR
from wfcModel import Node, Pattern

#inputWorld = anvil.Region.from_file(INPUT_DIR + 'r.0.0.mca')
#inputCache = []

def getMaxY(inputWorld):
    for i in range(1000):
        if anvil.Chunk.from_region(inputWorld, 0, 0).get_block(0, i, 0).id == 'air':
            return i
    else:
        return 0

def getBlock(x, y, z, inputWorld, inputCache):
    """
        실제 월드에서의 (x, y, z) 좌표에 위치한 블록 객체를 반환하는 함수.
    """
    if inputCache[x][y][z] != -1: return inputCache[x][y][z]
    else:
        chunk__x = x // 16
        chunk__z = z // 16
        block__x = x % 16
        block__z = z % 16
        inputCache[x][y][z] = anvil.Chunk.from_region(inputWorld, chunk__x, chunk__z).get_block(block__x, y, block__z)
        return inputCache[x][y][z]

def registerBlocks(dim_x, dim_y, dim_z, inputWorld, inputCache, timestamp):
    """
        읽어온 월드 파일 내 존재하는 블록을 레지스트리에 등록하는 함수
    """
    blockRegistry = {}
    blockRegistryInv = []
    
    for i in range(dim_x):
        for j in range(dim_y):
            for k in range(dim_z):
                block_id = getBlock(i, j, k, inputWorld, inputCache).id
                blockRegistry[block_id] = 1
    q = 0
    for key in blockRegistry:
        blockRegistry[key] = q
        blockRegistryInv.append(key)
        q += 1
    timeElapsed = time.time()
    print("Block Registration completed. (took {}s)".format(round(timeElapsed - timestamp, 6)))
    return blockRegistry, blockRegistryInv

def extractPatterns(dim_x, dim_y, dim_z, blockRegistry, inputWorld, inputCache, timestamp):
    """
        월드 파일 내 블록 배치 패턴을 추출하는 함수
    """
    patterns = []
    for i in range(dim_x):
        for j in range(dim_y):
            for k in range(dim_z):
                propMatrix = [[[0 for _ in range(MATRIX_Z)] for _ in range(MATRIX_Y)] for _ in range(MATRIX_X)]
                check = 0
                for p in range(MATRIX_X):
                    for q in range(MATRIX_Y):
                        for r in range(MATRIX_Z):
                            x = i - OFF_X + p
                            y = j - OFF_Y + q
                            z = k - OFF_Z + r
                            if x < 0 or y < 0 or z < 0 or x >= SIZE_X or y >= SIZE_Y or z >= SIZE_Z:
                                propMatrix[p][q][r] = -1
                                check += 1
                            else:
                                propMatrix[p][q][r] = blockRegistry[getBlock(x, y, z, inputWorld, inputCache).id]
                center_id = blockRegistry[getBlock(i, j, k, inputWorld, inputCache).id]
                
                # 범위 밖 공간을 포함하는 패턴 필터링
                if check != 0: continue
                
                # 패턴 중심 블록이 공기 블록일 경우 와일드카드 패턴으로 대체
                if center_id == blockRegistry['air']:
                    for p in range(MATRIX_X):
                        for q in range(MATRIX_Y):
                            for r in range(MATRIX_Z):
                                if propMatrix[p][q][r] != blockRegistry['air']:
                                    propMatrix[p][q][r] = -1
                registerPattern(propMatrix, center_id, patterns)
                #if center_id != blockRegistry['air']:
                #    registerPattern(propMatrix, center_id, patterns)
    
    #propMatrix = [[[0 for _ in range(MATRIX_Z)] for _ in range(MATRIX_Y)] for _ in range(MATRIX_X)]
    #for p in range(MATRIX_X):
    #    for q in range(MATRIX_Y):
    #        for r in range(MATRIX_Z):
    #            propMatrix[p][q][r] = -1
    #propMatrix[OFF_X][OFF_Y][OFF_Z] = blockRegistry['air']
    #registerPattern(propMatrix, blockRegistry['air'], patterns)
    #propMatrix = [[[0 for _ in range(MATRIX_Z)] for _ in range(MATRIX_Y)] for _ in range(MATRIX_X)]
    #for p in range(MATRIX_X):
    #    for q in range(MATRIX_Y):
    #        for r in range(MATRIX_Z):
    #            propMatrix[p][q][r] = blockRegistry['air']
    #registerPattern(propMatrix, blockRegistry['air'], patterns)
    
    timeElapsed = time.time()
    print("Pattern Extraction completed. (took {}s)".format(round(timeElapsed - timestamp, 6)))
    return patterns

def registerPattern(matrix, center_id, patterns):
    # 회전 패턴 생성(Y축 시계방향)
    rotated90 = deepcopy(matrix)
    rotated180 = deepcopy(matrix)
    rotated270 = deepcopy(matrix)
    for x in range(MATRIX_X):
        for y in range(MATRIX_Y):
            for z in range(MATRIX_Z):
                x1 = z
                z1 = MATRIX_X - x - 1
                rotated90[x1][y][z1] = matrix[x][y][z]
                x2 = z1
                z2 = MATRIX_X - x1 - 1
                rotated180[x2][y][z2] = matrix[x][y][z]
                x3 = z2
                z3 = MATRIX_X - x2 - 1
                rotated270[x3][y][z3] = matrix[x][y][z]
    rotatedSet = [matrix, rotated90, rotated180, rotated270]
    
    # 패턴 중복 제거
    for extractedPattern in rotatedSet:
        for pattern in patterns:
            if pattern.isEqual(extractedPattern):
                pattern.count += 1
                break
        else:
            patterns.append(Pattern(extractedPattern, center_id, 1))

def writeMCA(wfcMatrix: list[list[list[Node]]], blockRegistryInv, epoch = 0):
    blockCount = len(blockRegistryInv)
    outputWorld = anvil.EmptyRegion(0, 0)
    blockClass = []
    barrier = anvil.Block('minecraft', 'barrier')
    for i in range(blockCount):
        stringID = blockRegistryInv[i]
        blockClass.append(anvil.Block('minecraft', stringID))
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            for z in range(SIZE_Z):
                block_id = wfcMatrix[x][y][z].block_id
                if block_id == -1: outputWorld.set_block(barrier, x, y, z)
                else: outputWorld.set_block(blockClass[block_id], x, y, z)
    if epoch == 0:
        outputWorld.save(OUTPUT_DIR + 'r.0.0.mca')
    else:
        outputWorld.save(OUTPUT_DIR + '{} r.0.0.mca'.format(epoch+1))

if __name__ == "__main__":
    pass