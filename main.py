import anvil, time
import wfcModel, mcaFileIO
from constantTable import SIZE_X, SIZE_Y, SIZE_Z, MATRIX_X, MATRIX_Y, MATRIX_Z, INPUT_DIR, OUTPUT_DIR
from wfcModel import wfcModel, Node, Pattern

complexity = 1

inputWorld = anvil.Region.from_file(INPUT_DIR + 'r.0.0.mca')
inputCache = []

dim_x = SIZE_X
dim_y = SIZE_Y
dim_z = SIZE_Z

BLOCK_REGISTRY = {}
BLOCK_REGISTRY_INV = []
PATTERNS = []

initLevel = None

def mcaInitializer():
    global dim_x, dim_y, dim_z, inputWorld, inputCache, BLOCK_REGISTRY, BLOCK_REGISTRY_INV, PATTERNS
    timestamp = time.time()
    dim_y = mcaFileIO.getMaxY(inputWorld)
    inputCache = [[[-1 for _ in range(dim_z)] for _ in range(dim_y)] for _ in range(dim_x)]
    BLOCK_REGISTRY, BLOCK_REGISTRY_INV = mcaFileIO.registerBlocks(dim_x, dim_y, dim_z, inputWorld, inputCache, timestamp)
    PATTERNS = mcaFileIO.extractPatterns(dim_x, dim_y, dim_z, BLOCK_REGISTRY, inputWorld, inputCache, timestamp)

def levelInitializer():
    global dim_x, dim_y, dim_z, initLevel
    

if __name__ == "__main__":
    mcaInitializer()
    print(BLOCK_REGISTRY)