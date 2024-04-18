import anvil, time
from copy import deepcopy
from constantTable import SIZE_X, SIZE_Y, SIZE_Z, MATRIX_X, MATRIX_Y, MATRIX_Z, OFF_X, OFF_Y, OFF_Z, INPUT_DIR, OUTPUT_DIR, HASH_MODULO
from wfcModel import Pattern

def getMaxY(inputWorld):
    for i in range(1000):
        if anvil.Chunk.from_region(inputWorld, 0, 0).get_block(0, i, 0).id == 'air':
            return i
    else:
        return 0

def getBlock(x, y, z, inputWorld, inputCache, chunkCache) -> anvil.Block:
    """
        실제 월드에서의 (x, y, z) 좌표에 위치한 블록 객체를 반환하는 함수.
    """
    if inputCache[x][y][z] != -1: return inputCache[x][y][z]
    else:
        chunk__x = x // 16
        chunk__z = z // 16
        block__x = x % 16
        block__z = z % 16
        if chunkCache[chunk__x][chunk__z] == None:
            chunkCache[chunk__x][chunk__z] = anvil.Chunk.from_region(inputWorld, chunk__x, chunk__z)
        inputCache[x][y][z] = chunkCache[chunk__x][chunk__z].get_block(block__x, y, block__z)
        return inputCache[x][y][z]

def extractFilterWhitelist(dim_x, dim_y, dim_z, inputWorld, inputCache, chunkCache, blockRegistry, airBlockID, floorBlockID, wallBlockID) -> tuple[set, set, set]: 
    """
        필터링을 위한 블록 목록을 추출하는 함수
    """
    platformFilter = set()
    floorFilter = set()
    wallFilter = set()
    for i in range(dim_x):
        for j in range(dim_y):
            for k in range(dim_z):
                block_string_id = getBlock(i, j, k, inputWorld, inputCache, chunkCache).id
                block_id = blockRegistry[block_string_id]
                if block_id == airBlockID:
                    platformFilter.add(blockRegistry[getBlock(i, j-1, k, inputWorld, inputCache, chunkCache).id])
                if block_id == floorBlockID:
                    floorFilter.add(blockRegistry[getBlock(i, j+1, k, inputWorld, inputCache, chunkCache).id])
                if block_id == wallBlockID:
                    if i == 0:
                        wallFilter.add(blockRegistry[getBlock(i+1, j, k, inputWorld, inputCache, chunkCache).id])
                    if i == dim_x - 1:
                        wallFilter.add(blockRegistry[getBlock(i-1, j, k, inputWorld, inputCache, chunkCache).id])
                    if k == 0:
                        wallFilter.add(blockRegistry[getBlock(i, j, k+1, inputWorld, inputCache, chunkCache).id])
                    if k == dim_z - 1:
                        wallFilter.add(blockRegistry[getBlock(i, j, k-1, inputWorld, inputCache, chunkCache).id])
    platformFilter.discard(airBlockID)
    platformFilter.discard(floorBlockID)
    floorFilter.discard(floorBlockID)
    wallFilter.discard(wallBlockID)
    return platformFilter, floorFilter, wallFilter

def registerBlocks(dim_x, dim_y, dim_z, inputWorld, inputCache, chunkCache, timestamp):
    """
        읽어온 월드 파일 내 존재하는 블록을 레지스트리에 등록하는 함수
    """
    blockRegistry = {}
    blockRegistryInv = []
    
    for i in range(dim_x):
        for j in range(dim_y):
            for k in range(dim_z):
                block_id = getBlock(i, j, k, inputWorld, inputCache, chunkCache).id
                blockRegistry[block_id] = 1
    q = 0
    for key in blockRegistry:
        blockRegistry[key] = q
        blockRegistryInv.append(key)
        q += 1
    timeElapsed = time.time()
    print("Block Registration completed. (took {}s)".format(round(timeElapsed - timestamp, 6)))
    return blockRegistry, blockRegistryInv

def extractPatterns(dim_x, dim_y, dim_z, blockRegistry, excludeBlocks, inputWorld, inputCache, chunkCache, timestamp):
    """
        월드 파일 내 블록 배치 패턴을 추출하는 함수
    """
    patterns = []
    patternHashTable = dict()
    for i in range(dim_x):
        for j in range(dim_y):
            for k in range(dim_z):
                propMatrix = [[[0 for _ in range(MATRIX_Z)] for _ in range(MATRIX_Y)] for _ in range(MATRIX_X)]
                check = 0
                usedBlocks = set()
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
                                block_id = blockRegistry[getBlock(x, y, z, inputWorld, inputCache, chunkCache).id]
                                if block_id == blockRegistry['air']:
                                    block_id = -1
                                propMatrix[p][q][r] = block_id
                                usedBlocks.add(propMatrix[p][q][r])
                center_id = blockRegistry[getBlock(i, j, k, inputWorld, inputCache, chunkCache).id]
                
                # 범위 밖 공간을 포함하는 패턴 등록 제외
                if check != 0: continue
                if len(usedBlocks & excludeBlocks) > 0: continue
                
                # 패턴 중심 블록이 공기 블록일 경우 등록 제외
                if center_id == blockRegistry['air']:
                    continue
                
                # 패턴 중심 블록이 공기 블록일 경우 와일드카드 패턴으로 대체
                #if center_id == blockRegistry['air']:
                #    for p in range(MATRIX_X):
                #        for q in range(MATRIX_Y):
                #            for r in range(MATRIX_Z):
                #                if propMatrix[p][q][r] != blockRegistry['air']:
                #                    propMatrix[p][q][r] = -1
                
                # 패턴 내 벽, 바닥 구성 블록이 등장할 경우 공기 블록으로 대체한 패턴을 같이 등록
                #if len(usedBlocks & excludeBlocks) > 0:
                #    extraPattern = deepcopy(propMatrix)
                #    for p in range(MATRIX_X):
                #        for q in range(MATRIX_Y):
                #            for r in range(MATRIX_Z):
                #                if extraPattern[p][q][r] not in excludeBlocks:
                #                    extraPattern[p][q][r] = blockRegistry['air']
                #    registerPattern(extraPattern, center_id, patterns, patternHashTable)
                
                registerPattern(propMatrix, center_id, patterns, patternHashTable)
    
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

def registerPattern(matrix, center_id, patterns, patternHashTable):
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
    
    # 패턴 등록 및 중복 제거
    for extractedPattern in rotatedSet:
        hashValue = patternHash(extractedPattern)
        if hashValue in patternHashTable:
            patterns[patternHashTable[hashValue]].count += 1
        else:
            patternHashTable[hashValue] = len(patterns)
            patterns.append(Pattern(extractedPattern, center_id, 1))

def patternHash(pattern):
    """
        패턴 해시값을 반환하는 함수
    """
    hashValue = 0
    for x in range(MATRIX_X):
        for y in range(MATRIX_Y):
            for z in range(MATRIX_Z):
                hashValue = (hashValue * 31 + pattern[x][y][z]) % HASH_MODULO
    return hashValue

def writeMCA(dim_x, dim_y, dim_z, matrix:list[list[list]], blockRegistryInv, epoch = 0):
    """
        완성된 3차원 월드 배열을 MCA 파일로 변환해 내보내는 함수
    """
    blockCount = len(blockRegistryInv)
    outputWorld = anvil.EmptyRegion(0, 0)
    blockClass = []
    barrier = anvil.Block('minecraft', 'barrier')
    for i in range(blockCount):
        stringID = blockRegistryInv[i]
        blockClass.append(anvil.Block('minecraft', stringID))
    for x in range(dim_x):
        for y in range(dim_y):
            for z in range(dim_z):
                block_id = matrix[x][y][z]
                if block_id < 0:
                    outputWorld.set_block(barrier, x, y, z)
                else:
                    outputWorld.set_block(blockClass[block_id], x, y, z)
    if epoch == 0:
        outputWorld.save(OUTPUT_DIR + 'r.0.0.mca')
    else:
        outputWorld.save(OUTPUT_DIR + '{} r.0.0.mca'.format(epoch))

if __name__ == "__main__":
    pass