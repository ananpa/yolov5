ACTIVATION_NAME = 'gsigmoid'
TRAINING_NAME = 'train-v5s-150e'
VALID_NAME = 'val-v5s-150e'

from models.yolo import Model
#from utils.activations import GeneralizedSigmoid

#model = Model(f'models/yolov5s-gsigmoid.yaml', ch=3, nc=1)
#model = Model(f'models/yolov5s-pelu.yaml', ch=3, nc=1)
#model = Model(f'models/yolov5s-relusilu.yaml', ch=3, nc=1)
#model = Model(f'models/yolov5s-reluelu.yaml', ch=3, nc=1)
model = Model(f'models/yolov5s-smish.yaml', ch=3, nc=1)
print(model.model[:5])