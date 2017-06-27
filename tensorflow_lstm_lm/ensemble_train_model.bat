@echo off

echo Training model 1...
py tensorflow_lstm_lm.py --action=train_model --vocab_file="%VOCAB_FILE%" --num_layers=%NUM_LAYERS% --hidden_sizes=%HIDDEN_SIZES% --init_scale=%INIT_SCALE% --train_text="%TRAIN_TEXT%" --train_batch_size=%TRAIN_BATCH_SIZE% --train_num_steps=%TRAIN_NUM_STEPS% --max_epoch=%MAX_EPOCH% --max_max_epoch=%MAX_MAX_EPOCH% --max_grad_norm=%MAX_GRAD_NORM% --learning_rate=%LEARNING_RATE% --lr_decay=%LR_DECAY% --min_lr=%MIN_LR% --keep_prob=%KEEP_PROB% --valid_text=%VALID_TEXT% --early_stop=%EARLY_STOP% --max_valid_error=%MAX_VALID_ERROR% --save_file="%MODEL_FILE0%"
echo.

echo Training model 2...
py tensorflow_lstm_lm.py --action=train_model --vocab_file="%VOCAB_FILE%" --num_layers=%NUM_LAYERS% --hidden_sizes=%HIDDEN_SIZES% --init_scale=%INIT_SCALE% --train_text="%TRAIN_TEXT%" --train_batch_size=%TRAIN_BATCH_SIZE% --train_num_steps=%TRAIN_NUM_STEPS% --max_epoch=%MAX_EPOCH% --max_max_epoch=%MAX_MAX_EPOCH% --max_grad_norm=%MAX_GRAD_NORM% --learning_rate=%LEARNING_RATE% --lr_decay=%LR_DECAY% --min_lr=%MIN_LR% --keep_prob=%KEEP_PROB% --valid_text=%VALID_TEXT% --early_stop=%EARLY_STOP% --max_valid_error=%MAX_VALID_ERROR% --save_file="%MODEL_FILE1%"
echo.

echo Training model 3...
py tensorflow_lstm_lm.py --action=train_model --vocab_file="%VOCAB_FILE%" --num_layers=%NUM_LAYERS% --hidden_sizes=%HIDDEN_SIZES% --init_scale=%INIT_SCALE% --train_text="%TRAIN_TEXT%" --train_batch_size=%TRAIN_BATCH_SIZE% --train_num_steps=%TRAIN_NUM_STEPS% --max_epoch=%MAX_EPOCH% --max_max_epoch=%MAX_MAX_EPOCH% --max_grad_norm=%MAX_GRAD_NORM% --learning_rate=%LEARNING_RATE% --lr_decay=%LR_DECAY% --min_lr=%MIN_LR% --keep_prob=%KEEP_PROB% --valid_text=%VALID_TEXT% --early_stop=%EARLY_STOP% --max_valid_error=%MAX_VALID_ERROR% --save_file="%MODEL_FILE2%"
echo.

echo Training model 4...
py tensorflow_lstm_lm.py --action=train_model --vocab_file="%VOCAB_FILE%" --num_layers=%NUM_LAYERS% --hidden_sizes=%HIDDEN_SIZES% --init_scale=%INIT_SCALE% --train_text="%TRAIN_TEXT%" --train_batch_size=%TRAIN_BATCH_SIZE% --train_num_steps=%TRAIN_NUM_STEPS% --max_epoch=%MAX_EPOCH% --max_max_epoch=%MAX_MAX_EPOCH% --max_grad_norm=%MAX_GRAD_NORM% --learning_rate=%LEARNING_RATE% --lr_decay=%LR_DECAY% --min_lr=%MIN_LR% --keep_prob=%KEEP_PROB% --valid_text=%VALID_TEXT% --early_stop=%EARLY_STOP% --max_valid_error=%MAX_VALID_ERROR% --save_file="%MODEL_FILE3%"
echo.

echo Training model 5...
py tensorflow_lstm_lm.py --action=train_model --vocab_file="%VOCAB_FILE%" --num_layers=%NUM_LAYERS% --hidden_sizes=%HIDDEN_SIZES% --init_scale=%INIT_SCALE% --train_text="%TRAIN_TEXT%" --train_batch_size=%TRAIN_BATCH_SIZE% --train_num_steps=%TRAIN_NUM_STEPS% --max_epoch=%MAX_EPOCH% --max_max_epoch=%MAX_MAX_EPOCH% --max_grad_norm=%MAX_GRAD_NORM% --learning_rate=%LEARNING_RATE% --lr_decay=%LR_DECAY% --min_lr=%MIN_LR% --keep_prob=%KEEP_PROB% --valid_text=%VALID_TEXT% --early_stop=%EARLY_STOP% --max_valid_error=%MAX_VALID_ERROR% --save_file="%MODEL_FILE4%"
echo.

echo Training model 6...
py tensorflow_lstm_lm.py --action=train_model --vocab_file="%VOCAB_FILE%" --num_layers=%NUM_LAYERS% --hidden_sizes=%HIDDEN_SIZES% --init_scale=%INIT_SCALE% --train_text="%TRAIN_TEXT%" --train_batch_size=%TRAIN_BATCH_SIZE% --train_num_steps=%TRAIN_NUM_STEPS% --max_epoch=%MAX_EPOCH% --max_max_epoch=%MAX_MAX_EPOCH% --max_grad_norm=%MAX_GRAD_NORM% --learning_rate=%LEARNING_RATE% --lr_decay=%LR_DECAY% --min_lr=%MIN_LR% --keep_prob=%KEEP_PROB% --valid_text=%VALID_TEXT% --early_stop=%EARLY_STOP% --max_valid_error=%MAX_VALID_ERROR% --save_file="%MODEL_FILE5%"
echo.

echo Training model 7...
py tensorflow_lstm_lm.py --action=train_model --vocab_file="%VOCAB_FILE%" --num_layers=%NUM_LAYERS% --hidden_sizes=%HIDDEN_SIZES% --init_scale=%INIT_SCALE% --train_text="%TRAIN_TEXT%" --train_batch_size=%TRAIN_BATCH_SIZE% --train_num_steps=%TRAIN_NUM_STEPS% --max_epoch=%MAX_EPOCH% --max_max_epoch=%MAX_MAX_EPOCH% --max_grad_norm=%MAX_GRAD_NORM% --learning_rate=%LEARNING_RATE% --lr_decay=%LR_DECAY% --min_lr=%MIN_LR% --keep_prob=%KEEP_PROB% --valid_text=%VALID_TEXT% --early_stop=%EARLY_STOP% --max_valid_error=%MAX_VALID_ERROR% --save_file="%MODEL_FILE6%"
echo.

echo Training model 8...
py tensorflow_lstm_lm.py --action=train_model --vocab_file="%VOCAB_FILE%" --num_layers=%NUM_LAYERS% --hidden_sizes=%HIDDEN_SIZES% --init_scale=%INIT_SCALE% --train_text="%TRAIN_TEXT%" --train_batch_size=%TRAIN_BATCH_SIZE% --train_num_steps=%TRAIN_NUM_STEPS% --max_epoch=%MAX_EPOCH% --max_max_epoch=%MAX_MAX_EPOCH% --max_grad_norm=%MAX_GRAD_NORM% --learning_rate=%LEARNING_RATE% --lr_decay=%LR_DECAY% --min_lr=%MIN_LR% --keep_prob=%KEEP_PROB% --valid_text=%VALID_TEXT% --early_stop=%EARLY_STOP% --max_valid_error=%MAX_VALID_ERROR% --save_file="%MODEL_FILE7%"
echo.

echo Training model 9...
py tensorflow_lstm_lm.py --action=train_model --vocab_file="%VOCAB_FILE%" --num_layers=%NUM_LAYERS% --hidden_sizes=%HIDDEN_SIZES% --init_scale=%INIT_SCALE% --train_text="%TRAIN_TEXT%" --train_batch_size=%TRAIN_BATCH_SIZE% --train_num_steps=%TRAIN_NUM_STEPS% --max_epoch=%MAX_EPOCH% --max_max_epoch=%MAX_MAX_EPOCH% --max_grad_norm=%MAX_GRAD_NORM% --learning_rate=%LEARNING_RATE% --lr_decay=%LR_DECAY% --min_lr=%MIN_LR% --keep_prob=%KEEP_PROB% --valid_text=%VALID_TEXT% --early_stop=%EARLY_STOP% --max_valid_error=%MAX_VALID_ERROR% --save_file="%MODEL_FILE8%"
echo.

echo Training model 10...
py tensorflow_lstm_lm.py --action=train_model --vocab_file="%VOCAB_FILE%" --num_layers=%NUM_LAYERS% --hidden_sizes=%HIDDEN_SIZES% --init_scale=%INIT_SCALE% --train_text="%TRAIN_TEXT%" --train_batch_size=%TRAIN_BATCH_SIZE% --train_num_steps=%TRAIN_NUM_STEPS% --max_epoch=%MAX_EPOCH% --max_max_epoch=%MAX_MAX_EPOCH% --max_grad_norm=%MAX_GRAD_NORM% --learning_rate=%LEARNING_RATE% --lr_decay=%LR_DECAY% --min_lr=%MIN_LR% --keep_prob=%KEEP_PROB% --valid_text=%VALID_TEXT% --early_stop=%EARLY_STOP% --max_valid_error=%MAX_VALID_ERROR% --save_file="%MODEL_FILE9%"
echo.
