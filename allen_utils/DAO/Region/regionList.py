class RegionList:
    def __init__(self):
        self.regionList={}
        self.regionDict = {}

    def setRegion(self, region):
        self.region = region

    def addRegion(self, region, ID):
        info = {"id{}".format(ID):self.region.getInfo()}
        self.regionList.update(info)
    def update(self):
        self.regionDict.update(self.regionList)
    def getRegionList(self):
        self.update()
        return self.regionDict

