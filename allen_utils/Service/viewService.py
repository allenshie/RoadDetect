import cv2
class ViewService:
    def __init__(self):
        pass
    def setViewInput(self, regionList, predImg):
        self.regionList = regionList
        self.predImg = predImg
    def getRegionList(self):
        img = self.predImg
        for i in self.regionList:
            coordinate = self.regionList[i]["coordinate"]
            if self.regionList[i]["state"]=="obstance":
                img = self.draw_Xmark(coordinate, img)
        img = cv2.resize(img, (640,480))
        cv2.imshow("framed", self._drawGrid(img, 5))
        cv2.waitKey(1)


    def _drawGrid(self, img, grid_number):
        h, w, _ = img.shape
        for i in range(1,grid_number):
            img[int(h/grid_number*i),:]=0
            img[:,int(w/grid_number*i)]=0  
        return img
    def draw_Xmark(self, coordinate, img):
        x1, y1, x2, y2 =coordinate
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
        cv2.line(img, (x2, y1), (x1, y2), (0, 0, 255), 5)
        return img
        