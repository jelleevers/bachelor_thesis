@echo off

SET TRAIN_TEXT=data/ptb/train.txt
SET VALID_TEXT=data/ptb/valid.txt
SET TEST_TEXT=data/ptb/test.txt

SET VOCAB_FILE=ptb_vocab.bin
SET MODEL_FILE0=ptb_model_medium_0.bin
SET MODEL_FILE1=ptb_model_medium_1.bin
SET MODEL_FILE2=ptb_model_medium_2.bin
SET MODEL_FILE3=ptb_model_medium_3.bin
SET MODEL_FILE4=ptb_model_medium_4.bin
SET MODEL_FILE5=ptb_model_medium_5.bin
SET MODEL_FILE6=ptb_model_medium_6.bin
SET MODEL_FILE7=ptb_model_medium_7.bin
SET MODEL_FILE8=ptb_model_medium_8.bin
SET MODEL_FILE9=ptb_model_medium_9.bin

SET NUM_LAYERS=2
SET HIDDEN_SIZES=650,650
SET INIT_SCALE=0.05
SET TRAIN_BATCH_SIZE=20
SET TRAIN_NUM_STEPS=35
SET MAX_EPOCH=6
SET MAX_MAX_EPOCH=39
SET MAX_GRAD_NORM=5
SET LEARNING_RATE=1.0
SET LR_DECAY=0.8
SET MIN_LR=0.001
SET KEEP_PROB=0.5
SET EARLY_STOP=0
SET MAX_VALID_ERROR=5

SET TEST_BATCH_SIZE=1
SET TEST_NUM_STEPS=1
