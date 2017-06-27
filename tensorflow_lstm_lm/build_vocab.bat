@echo off

py tensorflow_lstm_lm.py --action=build_vocab --vocab_text="%TRAIN_TEXT%" --save_file="%VOCAB_FILE%"
