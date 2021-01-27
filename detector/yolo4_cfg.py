from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = 'detector/yolo4_csp/cfg/yolo4-csp.cfg'
cfg.WEIGHTS = 'detector/yolo4_csp/weights/yolo4-csp.weights'
cfg.INP_DIM = 512
cfg.NMS_THRES = 0.6
cfg.CONFIDENCE = 0.4
cfg.NUM_CLASSES = 80
