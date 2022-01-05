import cv2
import time
class LoadService:
    def __init__(self):
        self.frame = None
    def setPath(self, path):
        self.path =path
    def getPath(self):
        return self.path
    def run(self):
        cap = cv2.VideoCapture(self.getPath())
        while(cap.isOpened()):
            time.sleep(0.01)
            ret, self.frame = cap.read()
            if ret==False:
                break
    def getframe(self):
        if self.frame is not None:
            return self.frame
        else:
            return