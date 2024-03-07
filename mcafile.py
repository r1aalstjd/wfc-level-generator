import anvil
from copy import deepcopy
from constantTable import MATRIX_X, MATRIX_Y, MATRIX_Z

INPUT_DIR = './Input/'
inputWorld = anvil.Region.from_file(INPUT_DIR + 'r.0.0.mca')
inputCache = []

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
    if inputCache[x][y][z] != 0: return inputCache[x][y][z]
    else:
        chunk__x = x // 16
        chunk__z = z // 16
        block__x = x % 16
        block__z = z % 16
        inputCache[x][y][z] = anvil.Chunk.from_region(inputWorld, chunk__x, chunk__z).get_block(block__x, y, block__z)
        return inputCache[x][y][z]

def registerBlocks(inputWorld):
    """
        읽어온 월드 파일 내 존재하는 블록을 레지스트리에 등록하는 함수
    """
    from wfcModelTest import SIZE_X, SIZE_Y, SIZE_Z
    blockRegistry = {}
    blockRegistryInv = []
    
    for i in range(SIZE_X):
        for j in range(SIZE_Y):
            for k in range(SIZE_Z):
                block_id = getBlock(i, j, k).id
                blockRegistry[block_id] = 1
    q = 0
    for key in blockRegistry:
        blockRegistry[key] = q
        blockRegistryInv.append(key)
        q += 1
    print("Block Registration completed.")
    return blockRegistry, blockRegistryInv

def extractPatterns(blockRegistry, patterns):
    """
        월드 파일 내 블록 배치 패턴을 추출하는 함수
    """
    from wfcModelTest import SIZE_X, SIZE_Y, SIZE_Z
    off_x = MATRIX_X // 2
    off_y = MATRIX_Y // 2
    off_z = MATRIX_Z // 2
    for i in range(SIZE_X):
        for j in range(SIZE_Y):
            for k in range(SIZE_Z):
                propMatrix = [[[0 for _ in range(MATRIX_Z)] for _ in range(MATRIX_Y)] for _ in range(MATRIX_X)]
                x = i - off_x
                y = j - off_y
                z = k - off_z
                for p in range(MATRIX_X):
                    for q in range(MATRIX_Y):
                        for r in range(MATRIX_Z):
                            x = i - off_x + p
                            y = j - off_y + q
                            z = k - off_z + r
                            if x < 0 or y < 0 or z < 0 or x >= SIZE_X or y >= SIZE_Y or z >= SIZE_Z:
                                propMatrix[p][q][r] = -1
                            else:
                                propMatrix[p][q][r] = blockRegistry[getBlock(x, y, z).id]
                center_id = blockRegistry[getBlock(i, j, k).id]
                
                # 패턴 중복 제거
                for matrix in patterns[center_id]:
                    duplication_check = 0
                    for p in range(MATRIX_X):
                        for q in range(MATRIX_Y):
                            for r in range(MATRIX_Z):
                                if matrix[p][q][r] == propMatrix[p][q][r]:
                                    duplication_check += 1
                    if duplication_check == MATRIX_X * MATRIX_Y * MATRIX_Z:
                        break
                else:
                    patterns[center_id].append(propMatrix)
    print("Pattern Extraction completed.")
    return patterns

if __name__ == "__main__":
    pass