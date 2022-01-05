import torch
from yolact_edge.yolact import Yolact
from yolact_edge.utils.augmentations import FastBaseTransform_for_cpu

class YolactService:
    def __init__(self, path):
        net = Yolact(training=False)
        net.load_weights(path)
        net.detect.use_fast_nms = True
        self.net =net
        self.threshold = 0.3
    def detectImg(self, img, visualService):
        frame = torch.from_numpy(img).float()
        batch = FastBaseTransform_for_cpu()(frame.unsqueeze(0))
        extras = {"backbone": "full", "interrupt": False, "keep_statistics": False, "moving_statistics": None}
        preds = self.net(batch, extras=extras)["pred_outs"]
        img_numpy, mask = visualService.prep_display(preds, frame, None, None, undo_transform=False)
        return img_numpy, mask
    
        