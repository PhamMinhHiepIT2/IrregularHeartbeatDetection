from scripts.app import MODEL_PATH


RESNET50_MODEL = 'resnet50'
RESNET152_MODEL = 'resnet152'
VGG16_MODEL = 'vgg16'
VGG19_MODEL = 'vgg19'
WEIGHT_FOLDER = 'weight'
METRICS_FOLDER = 'metrics'

RESNET50_WEIGHT = "resnet50.h5"
RESNET152_WEIGHT = "resnet152.h5"
VGG16_WEIGHT = "vgg16.h5"
VGG19_WEIGHT = "vgg19.h5"

RESNET50_METRICS = "resnet50.npy"
RESNET152_METRICS = "resnet152.npy"
VGG16_METRICS = "vgg16.npy"
VGG19_METRICS = "vgg19.npy"

# classes model needs to learn to classify
CLASSES_TO_CHECK = ['L', 'N', 'V', 'A', 'R']
NUMBER_OF_CLASSES = len(CLASSES_TO_CHECK)
IMAGES_TO_TRAIN = 5000

# Model path
MODEL_PATH = "models/vgg16-on.h5"
