@echo off

SET TRAIN_TEXT=data/cgn/train.txt
SET VALID_TEXT=data/cgn/valid.txt
SET TEST_TEXT=data/cgn/test.txt

SET VOCAB_FILE=cgn_vocab.bin
SET MODEL_FILE0=cgn_model_large_0.bin
SET MODEL_FILE1=cgn_model_large_1.bin
SET MODEL_FILE2=cgn_model_large_2.bin
SET MODEL_FILE3=cgn_model_large_3.bin
SET MODEL_FILE4=cgn_model_large_4.bin
SET MODEL_FILE5=cgn_model_large_5.bin
SET MODEL_FILE6=cgn_model_large_6.bin
SET MODEL_FILE7=cgn_model_large_7.bin
SET MODEL_FILE8=cgn_model_large_8.bin
SET MODEL_FILE9=cgn_model_large_9.bin

SET NUM_LAYERS=2
SET HIDDEN_SIZES=1500,1500
SET INIT_SCALE=0.04
SET TRAIN_BATCH_SIZE=20
SET TRAIN_NUM_STEPS=35
SET MAX_EPOCH=12
SET MAX_MAX_EPOCH=80
SET MAX_GRAD_NORM=5
SET LEARNING_RATE=1.0
SET LR_DECAY=0.85
SET MIN_LR=0.001
SET KEEP_PROB=0.30
SET EARLY_STOP=1
SET MAX_VALID_ERROR=4

SET TEST_BATCH_SIZE=1
SET TEST_NUM_STEPS=1
