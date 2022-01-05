from ..DAO.Region.region import Region
from ..DAO.Region.regionList import RegionList
import numpy as np
import cv2
from numpy import linalg as LA

class RegionService:
    def __init__(self):
        self.region = Region()
        self.regionList = RegionList()
        self.regionList.setRegion(self.region)
    def setRegionInput(self, mask):
        self.mask = mask
        self.matrix_coordinate = self._create_coordinate(5, mask)       

    def get_init_mask(self):
        output_file = "init_mask.txt"
        init_mask_list =[]
        for i in range(len(self.matrix_coordinate)):
            init_cell = self._getMaskCellValue(self.mask ,self.matrix_coordinate[i])
            init_mask_list.append(init_cell)
        f = open(output_file, 'w')
        for i in init_mask_list:
            f.write(str(i))
            f.write("\n")
        f.close()
        return init_mask_list

    def getRegionList(self, init_mask, standardError):
        for i in range(len(self.matrix_coordinate)):
            error_cell = self._getErrorCell(self.mask, init_mask, self.matrix_coordinate[i])
            state = self._detect_obstacle(error_cell, standardError)
            self.region.setRegion(ID=i, coordinate=self.matrix_coordinate[i], state=state)
            self.regionList.addRegion(self.region.getInfo(), i)
        return self.regionList.getRegionList()
        
    def _create_coordinate(self, matrix_shape, img):
        h, w, _ = img.shape
        hr, wr = int(h/matrix_shape), int(w/matrix_shape)
        matrix_coordinate=[]
        y1 = 0
        for i in range(matrix_shape):
            x1 = 0
            y2 = y1 + hr
            for j in range(matrix_shape):
                x2 = x1 + wr
                matrix_coordinate.append([x1,y1,x2,y2])
                x1=x2
            y1=y2
        return matrix_coordinate
    def _detect_obstacle(self, error_cell, init_error_cell):
        if error_cell < init_error_cell:
            state = "clearance"
        else:
            state = "obstance"
        return state 

    def _getErrorCell(self, img, init_mask, coordinate):
        x1,y1,x2,y2 = coordinate
        crop_img  = img[y1:y2, x1:x2]
        init_crop = init_mask[y1:y2, x1:x2]
        h, w, _ = crop_img.shape	
        whiteImg = np.full((h, w, 3), 255).astype(np.uint8)
        init_error = np.sum(whiteImg)-np.sum(init_crop)
        current_crror = self.compare_test(init_crop, crop_img, init_error)
        # print(current_crror)
        return current_crror
    def compare_test(self,init_mask, mask, init_error):
        if np.sum(init_mask) >= np.sum(mask):
            return 0
        else:
            return (np.sum(mask) -np.sum(init_mask))/init_error
    def _compareMaskArea(self, init_mask, mask):
        if np.sum(init_mask) > np.sum(mask):
            error_cell = 0
        else: 
            error_cell = LA.norm(mask-init_mask)/LA.norm(init_mask)
        return error_cell

