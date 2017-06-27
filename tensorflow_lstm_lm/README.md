# TensorFlow LSTM Language Model Tool

This tool builds LSTMs language models using TensorFlow. It has some basic capabilities already implemented, but should be considered a work in progress.

Usage: 
    tensorflow_lstm_lm.py [-h] [--action ACTION] [--vocab_file VOCAB_FILE]
                             [--model_file MODEL_FILE] [--save_file SAVE_FILE]
                             [--vocab_text VOCAB_TEXT]
                             [--num_layers NUM_LAYERS]
                             [--hidden_sizes HIDDEN_SIZES]
                             [--init_scale INIT_SCALE]
                             [--forget_bias FORGET_BIAS]
                             [--use_fp16 [USE_FP16]] [--nouse_fp16]
                             [--train_text TRAIN_TEXT]
                             [--train_batch_size TRAIN_BATCH_SIZE]
                             [--train_num_steps TRAIN_NUM_STEPS]
                             [--max_epoch MAX_EPOCH]
                             [--max_max_epoch MAX_MAX_EPOCH]
                             [--max_grad_norm MAX_GRAD_NORM]
                             [--learning_rate LEARNING_RATE]
                             [--lr_decay LR_DECAY] [--min_lr MIN_LR]
                             [--keep_prob KEEP_PROB] [--valid_text VALID_TEXT]
                             [--valid_batch_size VALID_BATCH_SIZE]
                             [--valid_num_steps VALID_NUM_STEPS]
                             [--early_stop [EARLY_STOP]] [--noearly_stop]
                             [--max_valid_error MAX_VALID_ERROR]
                             [--test_text TEST_TEXT]
                             [--test_batch_size TEST_BATCH_SIZE]
                             [--test_num_steps TEST_NUM_STEPS]
                             [--gen_input_text GEN_INPUT_TEXT]
                             [--gen_count GEN_COUNT]
                             [--gen_start_random [GEN_START_RANDOM]]
                             [--nogen_start_random]
                             [--gen_one_sentence [GEN_ONE_SENTENCE]]
                             [--nogen_one_sentence]

Arguments:
  -h, --help            show this help message and exit
  --action ACTION       An action. Possible options are: build_vocab,
                        train_model, test_model_single, test_model_ensemble,
                        generate.
  --vocab_file VOCAB_FILE
                        Where the vocabulary data is stored.
  --model_file MODEL_FILE
                        Where the model data is stored.
  --save_file SAVE_FILE
                        Output file.
  --vocab_text VOCAB_TEXT
                        Where the vocabulary build text is stored.
  --num_layers NUM_LAYERS
                        Number of layers in LSTM.
  --hidden_sizes HIDDEN_SIZES
                        Number of nodes in each LSTM layer separated by
                        commas.
  --init_scale INIT_SCALE
                        The initial scale of the LSTM weights.
  --forget_bias FORGET_BIAS
                        The LSTM forget bias.
  --use_fp16 [USE_FP16]
                        Train using 16-bit floats instead of 32bit floats.
  --nouse_fp16
  --train_text TRAIN_TEXT
                        Where the train text is stored.
  --train_batch_size TRAIN_BATCH_SIZE
                        The train data batch size.
  --train_num_steps TRAIN_NUM_STEPS
                        The number of unrolled LSTM steps on the train data.
  --max_epoch MAX_EPOCH
                        The number of epochs trained with the initial learning
                        rate.
  --max_max_epoch MAX_MAX_EPOCH
                        The total number of epochs for training.
  --max_grad_norm MAX_GRAD_NORM
                        The maximum permissible norm of the gradient.
  --learning_rate LEARNING_RATE
                        The initial learning rate.
  --lr_decay LR_DECAY   The decay of the learning rate for each epoch after
                        'max_epoch'.
  --min_lr MIN_LR       The minimum learning rate.
  --keep_prob KEEP_PROB
                        The probability of keeping weights in the dropout
                        layer.
  --valid_text VALID_TEXT
                        Where the validation text is stored.
  --valid_batch_size VALID_BATCH_SIZE
                        The validation data batch size.
  --valid_num_steps VALID_NUM_STEPS
                        The number of unrolled LSTM steps on the validation
                        data.
  --early_stop [EARLY_STOP]
                        Train using early stopping based on validation errors.
  --noearly_stop
  --max_valid_error MAX_VALID_ERROR
                        The maximum validation errors to trigger an early
                        stop.
  --test_text TEST_TEXT
                        Where the test text is stored.
  --test_batch_size TEST_BATCH_SIZE
                        The test data batch size.
  --test_num_steps TEST_NUM_STEPS
                        The number of unrolled LSTM steps on the test data.
  --gen_input_text GEN_INPUT_TEXT
                        Where the generator input text is stored.
  --gen_count GEN_COUNT
                        The amount of words to generate.
  --gen_start_random [GEN_START_RANDOM]
                        If no input text is specified, start with a random
                        word.
  --nogen_start_random
  --gen_one_sentence [GEN_ONE_SENTENCE]
                        Keep generating until one sentence has been completed.
  --nogen_one_sentence
