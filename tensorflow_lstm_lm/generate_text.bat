@echo off

echo Generating using model 1...
py tensorflow_lstm_lm_update.py --action=generate --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE0%" --gen_count=100 --gen_one_sentence --gen_input_text=data/gen_input.txt
echo.
