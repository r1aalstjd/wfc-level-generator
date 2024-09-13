from Structures.Structure import Structure

class Platform(Structure):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, pos1:tuple[int], pos2:tuple[int]) -> "Platform":
        self.setShape(pos1, pos2)
        return self
    
    def setShape(self, pos1:tuple[int], pos2:tuple[int]) -> None:
        self.x1 = min(pos1[0], pos2[0])
        self.y1 = min(pos1[1], pos2[1])
        self.z1 = min(pos1[2], pos2[2])
        self.x2 = max(pos1[0], pos2[0])
        self.y2 = max(pos1[1], pos2[1])
        self.z2 = max(pos1[2], pos2[2])