import configparser

class ConfigService:
    def __init__(self):
        self.config = configparser.ConfigParser()
    def setFieldConfig(self, videoPath):
        self.field = videoPath
    def getFieldFloor(self):
        floor=self.field.split("../")[1].split("_")[0]
        print(floor)
        return floor
    def loadConfig(self, floor):
        self.config.read("./config/{}.ini".format(floor))
    def getConfigParameter(self):
        init_mask = self.config['sql']['init_mask']
        checkpoint = self.config['sql']['checkpoint']
        standardError = self.config['sql']['standardError']
        return checkpoint, init_mask, standardError

