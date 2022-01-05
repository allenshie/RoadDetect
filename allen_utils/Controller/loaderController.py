from ..Service.loadService import LoadService
class LoadController:
    def __init__(self, path):
        self.loadService = LoadService()
        self.loadService.setPath(path)
        self.path = path
    def run(self):
        self.loadService.run()
    def getLoaderInfo(self):
        return self.path
    def getframe(self):
        return self.loadService.getframe()