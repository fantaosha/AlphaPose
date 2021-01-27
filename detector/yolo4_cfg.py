from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = 'detector/yolo4_csp/cfg/yolov4-csp.cfg'
cfg.WEIGHTS = 'detector/yolo4_csp/weights/yolov4-csp.weights'
cfg.INP_DIM = 608
cfg.NMS_THRES = 0.6
cfg.CONFIDENCE = 0.05
cfg.NUM_CLASSES = 80
