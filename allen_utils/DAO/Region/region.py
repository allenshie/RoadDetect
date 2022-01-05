class Region:
    def __init__(self, ID = 0 ,coordinate = [], state = "null"):
        self.ID = ID
        self.coordinate = coordinate
        self.state = state
    def setRegion(self, ID, coordinate, state):
        self.ID = ID
        self.coordinate = coordinate
        self.state = state
    def getInfo(self):
        return {
            "ID": self.ID,
            "coordinate": self.coordinate,
            "state":  self.state}