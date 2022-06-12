# number of heartbeats to extract
NUM_HEARTBEATS_TO_EXTRACT = 1
BEAT_START_OFFSET = 100
BEAT_END_OFFSET = 100
WRITE_DIR = 'hearbeat-data'
DATA_DIR = 'data/mit-bih_waveform/'
# classes model needs to learn to classify
CLASSES_TO_CHECK = ['L', 'N', 'V', 'A', 'R']
NUMBER_OF_CLASSES = len(CLASSES_TO_CHECK)
IMAGES_TO_TRAIN = 5000
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
