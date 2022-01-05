from ..Service.yolactService import YolactService
from ..Service.visualService import VisualService
from ..Service.regionService import RegionService
from ..Service.configService import ConfigService
from ..Service.viewService import ViewService
import cv2
class DetectObstacleController:
    def __init__(self):
        self.configService = ConfigService()
        self.visualService = VisualService()
        self.regionService = RegionService()
        self.viewService = ViewService()
        
    def setConfig(self, path):
        self.configService.setFieldConfig(path)
        self.configService.loadConfig(self.configService.getFieldFloor()) #依據載入影像名稱獲得對應場域並載入對應config檔
        weight_path, init_mask_path, standardError = self.configService.getConfigParameter() #config檔內共三個參數分別為模型權重、初始圖路徑、誤差衡量值
        self.init_mask = cv2.imread(init_mask_path)
        self.standardError = float(standardError)
        self.yolactService = YolactService(weight_path)

    # def obtain_init_mask(self, img):
    #     img, mask = self.yolactService.detectImg(img, self.visualService)
    #     self.regionService.setRegionInput(mask)
    #     init_mask_list = self.regionService.get_init_mask() 
    #     return init_mask_list

    def detect(self, img, init_mask, standardError):
        img, mask = self.yolactService.detectImg(img, self.visualService) #img: normal img + mask     mask: white img + mask 
        self.regionService.setRegionInput(mask)
        regionList = self.regionService.getRegionList(init_mask, standardError)
        return regionList, img
    

    def run(self, loaderController):

        while 1:
            img = loaderController.getframe()
            if img is not None:
                regionList, predImg = self.detect(img, self.init_mask, self.standardError)
                self.viewService.setViewInput(regionList, predImg)
                self.viewService.getRegionList()

        


        