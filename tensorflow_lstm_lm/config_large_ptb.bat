@echo off

SET TRAIN_TEXT=data/ptb/train.txt
SET VALID_TEXT=data/ptb/valid.txt
SET TEST_TEXT=data/ptb/test.txt

SET VOCAB_FILE=ptb_vocab.bin
SET MODEL_FILE0=ptb_model_large_0.bin
SET MODEL_FILE1=ptb_model_large_1.bin
SET MODEL_FILE2=ptb_model_large_2.bin
SET MODEL_FILE3=ptb_model_large_3.bin
SET MODEL_FILE4=ptb_model_large_4.bin
SET MODEL_FILE5=ptb_model_large_5.bin
SET MODEL_FILE6=ptb_model_large_6.bin
SET MODEL_FILE7=ptb_model_large_7.bin
SET MODEL_FILE8=ptb_model_large_8.bin
SET MODEL_FILE9=ptb_model_large_9.bin

SET NUM_LAYERS=2
SET HIDDEN_SIZES=1500,1500
SET INIT_SCALE=0.04
SET TRAIN_BATCH_SIZE=20
SET TRAIN_NUM_STEPS=35
SET MAX_EPOCH=14
SET MAX_MAX_EPOCH=55
SET MAX_GRAD_NORM=10
SET LEARNING_RATE=1.0
SET LR_DECAY=0.8696
SET MIN_LR=0.0001
SET KEEP_PROB=0.35
SET EARLY_STOP=0
SET MAX_VALID_ERROR=4

SET TEST_BATCH_SIZE=1
SET TEST_NUM_STEPS=1
