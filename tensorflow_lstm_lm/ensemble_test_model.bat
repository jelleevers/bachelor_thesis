@echo off

echo Testing model 1...
py tensorflow_lstm_lm.py --action=test_model_single --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE0%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing model 2...
py tensorflow_lstm_lm.py --action=test_model_single --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE1%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing model 3...
py tensorflow_lstm_lm.py --action=test_model_single --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE2%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing model 4...
py tensorflow_lstm_lm.py --action=test_model_single --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE3%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing model 5...
py tensorflow_lstm_lm.py --action=test_model_single --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE4%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing model 6...
py tensorflow_lstm_lm.py --action=test_model_single --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE5%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing model 7...
py tensorflow_lstm_lm.py --action=test_model_single --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE6%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing model 8...
py tensorflow_lstm_lm.py --action=test_model_single --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE7%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing model 9...
py tensorflow_lstm_lm.py --action=test_model_single --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE8%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing model 10...
py tensorflow_lstm_lm.py --action=test_model_single --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE9%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing 2-model ensemble 1...
py tensorflow_lstm_lm.py --action=test_model_ensemble --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE0%|%MODEL_FILE1%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing 2-model ensemble 2...
py tensorflow_lstm_lm.py --action=test_model_ensemble --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE2%|%MODEL_FILE3%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing 2-model ensemble 3...
py tensorflow_lstm_lm.py --action=test_model_ensemble --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE4%|%MODEL_FILE5%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing 2-model ensemble 4...
py tensorflow_lstm_lm.py --action=test_model_ensemble --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE6%|%MODEL_FILE7%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing 2-model ensemble 5...
py tensorflow_lstm_lm.py --action=test_model_ensemble --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE8%|%MODEL_FILE9%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing 5-model ensemble 1...
py tensorflow_lstm_lm.py --action=test_model_ensemble --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE0%|%MODEL_FILE1%|%MODEL_FILE2%|%MODEL_FILE3%|%MODEL_FILE4%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing 5-model ensemble 2...
py tensorflow_lstm_lm.py --action=test_model_ensemble --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE5%|%MODEL_FILE6%|%MODEL_FILE7%|%MODEL_FILE8%|%MODEL_FILE9%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.

echo Testing 10-model ensemble...
py tensorflow_lstm_lm.py --action=test_model_ensemble --vocab_file="%VOCAB_FILE%" --model_file="%MODEL_FILE0%|%MODEL_FILE1%|%MODEL_FILE2%|%MODEL_FILE3%|%MODEL_FILE4%|%MODEL_FILE5%|%MODEL_FILE6%|%MODEL_FILE7%|%MODEL_FILE8%|%MODEL_FILE9%" --test_text="%TEST_TEXT%" --test_batch_size=%TEST_BATCH_SIZE% --test_num_steps=%TEST_NUM_STEPS%
echo.
