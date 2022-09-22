from yacs.config import CfgNode as CN

cfg = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

cfg.MODEL = CN()
cfg.MODEL.NAME = 'protonet_resnet'
# dimensionality of input images
cfg.MODEL.INPUT_DIM = (84, 84, 3)
# dimensionality of hidden layers
cfg.MODEL.HIDDEN_DIM = 64
# dimensionality of input images
cfg.MODEL.Z_DIM = 64
# Distance function
cfg.MODEL.DISTANCE = 'euclidean'
# Scaling parameter for cosine distance
cfg.MODEL.ALPHA = 20
# location of pretrained model to retrain in trainval mode
cfg.MODEL.PATH = 'best_model.pt'
# path to pretrained model
cfg.MODEL.PRETRAINED = ''
# dropout value during training. Set 0 to disable
cfg.MODEL.DROPOUT = 0.

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

cfg.DATASET = CN()
cfg.DATASET.NAME = 'miniImagenet'
cfg.DATASET.DATA_DIR = 'datasets'
# number of classes per episode
cfg.DATASET.WAY = 30
# number of support examples per class
cfg.DATASET.SHOT = 10
# number of query examples per class
cfg.DATASET.QUERY = 5
# number of classes per episode in test. 0 means same as data.way
cfg.DATASET.TEST_WAY = 5
# number of support examples per class in test. 0 means same as data.shot
cfg.DATASET.TEST_SHOT = 10
# number of query examples per class in test. 0 means same as data.query
cfg.DATASET.TEST_QUERY = 15
# number of test examples for human-in-the-loop simulation
cfg.DATASET.SIMULATION_TEST = 0
# number of train episodes per epoch
cfg.DATASET.TRAIN_EPISODES = 100
# number of test episodes per epoch. 0 means same as DATASET.TRAIN_EPISODES
cfg.DATASET.TEST_EPISODES = 100
# run in train+validation mode
cfg.DATASET.TRAINVAL = False
# evaluate base classes
cfg.DATASET.EVAL_BASE = False
# use sequential sampler instead of episodic
cfg.DATASET.SEQUENTIAL = False

# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------

cfg.TRAIN = CN()
# number of epochs to train
cfg.TRAIN.EPOCHS = 100
cfg.TRAIN.OPTIMIZER = 'Adam'
cfg.TRAIN.LEARNING_RATE = 0.001
# number of epochs after which to decay the learning rate
cfg.TRAIN.DECAY_EVERY = 20
cfg.TRAIN.WEIGHT_DECAY = 0.
# number of epochs to wait before validation improvement
cfg.TRAIN.PATIENCE = 200

# -----------------------------------------------------------------------------
# Data Augmentation
# -----------------------------------------------------------------------------

cfg.DATA_AUGMENTATION = CN()
cfg.DATA_AUGMENTATION.NORMALIZE = True
# Mean and std of pixels to center values
cfg.DATA_AUGMENTATION.PIXEL_MEAN = [0.485, 0.456, 0.406]  # average values from Imagenet
cfg.DATA_AUGMENTATION.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# REST
# -----------------------------------------------------------------------------

# fields to monitor during training
cfg.LOG_FIELDS = ('loss', 'acc', 'lr')
# directory where experiments should be saved
cfg.OUTPUT_DIR = 'outputs/miniImagenet'
# method for human-in-the-loop selection of query image (uncertainty measurement)
cfg.ACQUISITION_FUNCTIONS = [

    'oracle model', 'oracle method', 'random', 'class iteration',
    'maxentropy', 'margin', 'min confidence',
    'maxentropy distance', 'maxmin distance', 'margin distance',
    'cluster maxentropy', 'cluster maxdistance', 'cluster margin',
    # 'BALD', 'variation ratio',
]

# iterations for Monte Carlo Dropout for BALD and variational ratio
cfg.MC_DROPOUT_ITERATIONS = 0
