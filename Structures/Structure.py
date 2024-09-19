class Structure():
    def __init__(self, id:int) -> None:
        self.adjacent = dict()
        self.id = id
    
    def addAdjacent(self, structure:"Structure", direction:int) -> None:
        self.adjacent[direction] = structure.id