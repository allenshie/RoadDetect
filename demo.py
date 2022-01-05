from allen_utils.Controller.detectController import DetectObstacleController
from allen_utils.Controller.loaderController import LoadController
import threading

if __name__ == '__main__':
    detectController = DetectObstacleController()
    loaderController = LoadController("../1F_person.mp4")
    detectController.setConfig(loaderController.getLoaderInfo())
    t1 = threading.Thread(target=detectController.run, args=(loaderController,), daemon=True)
    t1.start()
    loaderController.run()
