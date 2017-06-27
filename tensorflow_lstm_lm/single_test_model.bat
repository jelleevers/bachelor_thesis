@echo off

echo Testing model 1...
echo Time started: %time%
py tensorflow_lstm_lm.py --action=test_model_single --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE0%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo Time ended: %time%
echo.
