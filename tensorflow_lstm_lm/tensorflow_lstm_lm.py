# Copyright 2017 Jelle Evers. All Rights Reserved.
#
# Some parts of this code are based on:
# https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb
# by The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time
import collections
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pickle

flags = tf.flags

#general flags
flags.DEFINE_string("action", None, "An action. Possible options are: build_vocab, train_model, test_model_single, test_model_ensemble, generate.")
flags.DEFINE_string("vocab_file", None, "Where the vocabulary data is stored.")
flags.DEFINE_string("model_file", None, "Where the model data is stored.")
flags.DEFINE_string("save_file", None, "Output file.")

#build vocabulary flags
flags.DEFINE_string("vocab_text", None, "Where the vocabulary build text is stored.")

#train model flags
flags.DEFINE_integer("num_layers", None, "Number of layers in LSTM.")
flags.DEFINE_string("hidden_sizes", None, "Number of nodes in each LSTM layer separated by commas.")
flags.DEFINE_float("init_scale", 0.5, "The initial scale of the LSTM weights.")
flags.DEFINE_float("forget_bias", 0.0, "The LSTM forget bias.")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats.")

flags.DEFINE_string("train_text", None, "Where the train text is stored.")
flags.DEFINE_integer("train_batch_size", None, "The train data batch size.")
flags.DEFINE_integer("train_num_steps", None, "The number of unrolled LSTM steps on the train data.")
flags.DEFINE_integer("max_epoch", None, "The number of epochs trained with the initial learning rate.")
flags.DEFINE_integer("max_max_epoch", None, "The total number of epochs for training.")
flags.DEFINE_float("max_grad_norm", None, "The maximum permissible norm of the gradient.")
flags.DEFINE_float("learning_rate", 1.0, "The initial learning rate.")
flags.DEFINE_float("lr_decay", 0.5, "The decay of the learning rate for each epoch after 'max_epoch'.")
flags.DEFINE_float("min_lr", 0.001, "The minimum learning rate.")
flags.DEFINE_float("keep_prob", 1.0, "The probability of keeping weights in the dropout layer.")

flags.DEFINE_string("valid_text", None, "Where the validation text is stored.")
flags.DEFINE_integer("valid_batch_size", None, "The validation data batch size.")
flags.DEFINE_integer("valid_num_steps", None, "The number of unrolled LSTM steps on the validation data.")
flags.DEFINE_bool("early_stop", False, "Train using early stopping based on validation errors.")
flags.DEFINE_integer("max_valid_error", 4, "The maximum validation errors to trigger an early stop.")

#test model flags
flags.DEFINE_string("test_text", None, "Where the test text is stored.")
flags.DEFINE_integer("test_batch_size", None, "The test data batch size.")
flags.DEFINE_integer("test_num_steps", None, "The number of unrolled LSTM steps on the test data.")

#generate flags
flags.DEFINE_string("gen_input_text", None, "Where the generator input text is stored.")
flags.DEFINE_integer("gen_count", 100, "The amount of words to generate.")
flags.DEFINE_bool("gen_start_random", False, "If no input text is specified, start with a random word.")
flags.DEFINE_bool("gen_one_sentence", False, "Keep generating until one sentence has been completed.")


FLAGS = flags.FLAGS


class LSTMVocabulary(object):
    def __init__(self, load_file=None, build_words=None):
        self._vocab_size = -1
        self._vocab_dict = None
        self._vocab_dict_inv = None
        
        if load_file:
            f = open(load_file, 'rb')
            self._vocab_dict = pickle.load(f)
            self._vocab_dict_inv = {v: k for k, v in self._vocab_dict.items()}
            self._vocab_size = len(self._vocab_dict)
            f.close()
        elif build_words:
            build_words = ["<eos>", "<unk>"] + build_words
            counter = collections.Counter(build_words)
            count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
            words, _ = list(zip(*count_pairs))
            
            self._vocab_size = len(words)
            self._vocab_dict = dict(zip(words, range(self._vocab_size)))
            self._vocab_dict_inv = {v: k for k, v in self._vocab_dict.items()}
            
        self._eos_id = self._vocab_dict["<eos>"]
        self._unk_id = self._vocab_dict["<unk>"]
    
    @property
    def vocab_size(self):
        return self._vocab_size
        
    @property
    def eos_id(self):
        return self._eos_id
        
    @property
    def unk_id(self):
        return self._unk_id
        
    def words_to_ids(self, words):
        return [self._vocab_dict[word] if word in self._vocab_dict else self._unk_id for word in words]
        
    def ids_to_words(self, ids):
        return [self._vocab_dict_inv[id] if id in self._vocab_dict_inv else "<unk>" for id in ids]
        
    def save_vocab_file(self, file):
        f = open(file, 'wb')
        pickle.dump(self._vocab_dict, f)
        f.close()
        
    
class LSTMInput(object):
    def __init__(self, vocab, input_data, batch_size, num_steps):
        self._vocab = vocab
        self._batch_size = batch_size
        self._num_steps = num_steps
        
        raw_data = self._vocab.words_to_ids(input_data)
        data_len = len(raw_data)
        
        raw_data = tf.convert_to_tensor(raw_data, dtype=tf.int32)
        
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
            
        epoch_size = (batch_len - 1) // num_steps
        self._epoch_size = epoch_size
        
        assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size)

        self._slice_index = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        
        x = tf.strided_slice(data, [0, self._slice_index * num_steps],
                             [batch_size, (self._slice_index + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        self._inputs = x
        
        y = tf.strided_slice(data, [0, self._slice_index * num_steps + 1],
                             [batch_size, (self._slice_index + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        self._targets = y
    
    @property
    def vocab_size(self):
        return self._vocab.vocab_size
    
    @property
    def batch_size(self):
        return self._batch_size
        
    @property
    def num_steps(self):
        return self._num_steps
    
    @property
    def epoch_size(self):
        return self._epoch_size
    
    @property
    def slice_index(self):
        return self._slice_index
    
    @property    
    def inputs(self):
        return self._inputs
    
    @property
    def targets(self):
        return self._targets
        

class LSTMModel(object):
    def __init__(self, vocab_size, num_layers, hidden_sizes, init_scale, forget_bias=0.0, name=None, use_fp16=False):
        self._vocab_size = vocab_size
        self._num_layers = num_layers
        self._hidden_sizes = hidden_sizes
        self._forget_bias = forget_bias
        self._use_fp16 = use_fp16
        self._data_type = tf.float16 if use_fp16 else tf.float32
        
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        
        with tf.variable_scope(name, "LSTMModel", reuse=None, initializer=initializer) as self._var_scope:
            pass
        
        lstm_net = self.lstm_net(self.lstm_cell)
        
        inputs = tf.placeholder(self._data_type, shape=(1, 1, self._hidden_sizes[0]))
        inputs = tf.unstack(inputs, num=1, axis=1)
        
        with tf.variable_scope(self._var_scope):
            with tf.device("/cpu:0"):
                tf.get_variable("embedding", [self._vocab_size, self._hidden_sizes[0]], dtype=self._data_type)
            tf.contrib.rnn.static_rnn(lstm_net, inputs, dtype=self._data_type)
            tf.get_variable("softmax_w", [self._hidden_sizes[-1], self._vocab_size], dtype=self._data_type)
            tf.get_variable("softmax_b", [self._vocab_size], dtype=self._data_type)
        
        self._var_scope.reuse_variables()
        
        self._model_param_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._var_scope.name)
        self._model_param_vars_assign_ops = [tf.get_default_graph().get_operation_by_name(v.op.name + "/Assign") for v in self._model_param_vars]
        self._model_param_vars_init_vals = [assign_op.inputs[1] for assign_op in self._model_param_vars_assign_ops]
    
    @property
    def vocab_size(self):
        return self._vocab_size
    
    @property
    def hidden_sizes(self):
        return self._hidden_sizes
        
    @property
    def num_layers(self):
        return self._num_layers
        
    @property
    def vocab_size(self):
        return self._vocab_size
    
    @property
    def data_type(self):
        return self._data_type
    
    @property
    def var_scope(self):
        return self._var_scope
    
    @property
    def model_param_vars(self):
        return self._model_param_vars
        
    def lstm_cell(self, layer_idx):
        with tf.variable_scope(self._var_scope):
            if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(self._hidden_sizes[layer_idx], forget_bias=self._forget_bias, state_is_tuple=True, reuse=self._var_scope.reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(self._hidden_sizes[layer_idx], forget_bias=self._forget_bias, state_is_tuple=True)
    
    def lstm_net(self, lstm_cell_func):
        with tf.variable_scope(self._var_scope):
            return tf.contrib.rnn.MultiRNNCell([lstm_cell_func(i) for i in range(self._num_layers)], state_is_tuple=True)
            
    def init_model_params(self, session):
        session.run(self._model_param_vars_assign_ops)
        
    def load_model_params(self, session, params):
        feed_dict = {init_value: param for init_value, param in zip(self._model_param_vars_init_vals, params)}
        session.run(self._model_param_vars_assign_ops, feed_dict=feed_dict)
        
    def save_model_params(self, session):
        return session.run(self._model_param_vars)

    @staticmethod
    def create_model_from_file(file, name=None):
        f = open(file, 'rb')
        store_dict = pickle.load(f)
        f.close()
        if isinstance(store_dict, dict):
            return LSTMModel(store_dict["vocab_size"], store_dict["num_layers"], store_dict["hidden_sizes"], store_dict["forget_bias"], name=name, use_fp16=store_dict["use_fp16"])
        else:
            print("Failed to load model design from '%s'." % (file))
    
    def load_model_file(self, session, file):
        f = open(file, 'rb')
        store_dict = pickle.load(f)
        f.close()
        if isinstance(store_dict, dict) and store_dict["vocab_size"] == self._vocab_size and store_dict["num_layers"] == self._num_layers and store_dict["hidden_sizes"] == self._hidden_sizes and store_dict["forget_bias"] == self._forget_bias and store_dict["use_fp16"] == self._use_fp16:
            self.load_model_params(session, store_dict["model_params"]) 
        else:
            print("Failed to load model params from '%s'. Model design does not match." % (file))
    
    def save_model_file(self, session, file):
        store_dict = {
            "vocab_size": self._vocab_size,
            "num_layers": self._num_layers,
            "hidden_sizes": self._hidden_sizes,
            "forget_bias": self._forget_bias,
            "use_fp16": self._use_fp16,
            "model_params": self.save_model_params(session)
        }
        f = open(file, 'wb')
        pickle.dump(store_dict, f)
        f.close()
        
        
class LSTMTesterSingle(object):
    def __init__(self, model, input):
        self._model = model
        self._input = input
        
        if self._model.vocab_size != self._input.vocab_size:
            raise ValueError("Model vocabulary size does not match input vocabulary size.")
        
        net = self._model.lstm_net(self._model.lstm_cell)
        
        self._initial_state = net.zero_state(self._input.batch_size, self._model.data_type)
         
        with tf.variable_scope(self._model.var_scope):
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [self._model.vocab_size, self._model.hidden_sizes[0]], dtype=self._model.data_type)
                inputs = tf.nn.embedding_lookup(embedding, self._input.inputs)
                
        inputs = tf.unstack(inputs, num=self._input.num_steps, axis=1)
        with tf.variable_scope(self._model.var_scope):
            outputs, state = tf.contrib.rnn.static_rnn(net, inputs, initial_state=self._initial_state)
            
        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, self._model.hidden_sizes[-1]])
        
        with tf.variable_scope(self._model.var_scope):
            softmax_w = tf.get_variable("softmax_w", [self._model.hidden_sizes[-1], self._model.vocab_size], dtype=self._model.data_type)
            softmax_b = tf.get_variable("softmax_b", [self._model.vocab_size], dtype=self._model.data_type)
        
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._input.targets, [-1])], [tf.ones([self._input.batch_size * self._input.num_steps], dtype=self._model.data_type)])
        
        self._cost = tf.reduce_sum(loss) / self._input.batch_size
        self._final_state = state
        
    def test_lstm(self, session, verbose=False):
        costs = 0.0
        iters = 0
        state = session.run(self._initial_state)

        fetches = {
            "cost": self._cost,
            "final_state": self._final_state
        }
        
        if verbose:
            print("Test Epoch")
            
        start_time = time.time()

        for step in range(self._input.epoch_size):
            feed_dict = {}
            for i, (c, h) in enumerate(self._initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            state = vals["final_state"]

            costs += cost
            iters += self._input.num_steps

            if verbose and step % (self._input.epoch_size // 10) == 10:
                print("  %.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / self._input.epoch_size, np.exp(costs / iters), iters * self._input.batch_size / (time.time() - start_time)))

        test_ppl = np.exp(costs / iters)   
        
        if verbose:
            print("Test Epoch Perplexity: %.3f" % (test_ppl))
        
        return test_ppl

        
class LSTMTesterEnsemble(object):
    def __init__(self, models, input):
        self._models = models
        self._model_count = len(self._models)
        self._input = input
        
        if self._model_count < 1:
            raise ValueError("Model count must be greater than zero.")

        self._vocab_size = self._models[0].vocab_size
        self._data_type = self._models[0].data_type
            
        for i in range(1, self._model_count):
            if self._models[i].vocab_size != self._vocab_size:
                raise ValueError("Model vocabulary sizes do not match.")
            if self._models[i].data_type != self._data_type:
                raise ValueError("Model data types do not match.")
        if self._vocab_size != self._input.vocab_size:
            raise ValueError("Model vocabulary sizes do not match input vocabulary size.")
            
        nets = [model.lstm_net(model.lstm_cell) for model in self._models]
        
        self._initial_states = [net.zero_state(self._input.batch_size, self._data_type) for net in nets]
        
        input_inputs = self._input.inputs
        inputss = [None] * self._model_count
        for i in range(self._model_count):
            with tf.variable_scope(self._models[i].var_scope):
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable("embedding", [self._vocab_size, self._models[i].hidden_sizes[0]], dtype=self._data_type)
                    inputss[i] = tf.nn.embedding_lookup(embedding, input_inputs)
                    
        inputss = [tf.unstack(inputs, num=self._input.num_steps, axis=1) for inputs in inputss]
        
        states = [None] * self._model_count
        outputss = [None] * self._model_count
        for i in range(self._model_count):
            with tf.variable_scope(self._models[i].var_scope):
                outputss[i], states[i] = tf.contrib.rnn.static_rnn(nets[i], inputss[i], initial_state=self._initial_states[i])
        
        outputs = [tf.reshape(tf.stack(axis=1, values=outputss[i]), [-1, self._models[i].hidden_sizes[-1]]) for i in range(self._model_count)]
        
        softmax_ws = [None] * self._model_count
        softmax_bs = [None] * self._model_count
        for i in range(self._model_count):
            with tf.variable_scope(self._models[i].var_scope):
                softmax_ws[i] = tf.get_variable("softmax_w", [self._models[i].hidden_sizes[-1], self._vocab_size], dtype=self._data_type)
                softmax_bs[i] = tf.get_variable("softmax_b", [self._vocab_size], dtype=self._data_type)
                
        logitss = [tf.nn.softmax(tf.matmul(outputs[i], softmax_ws[i]) + softmax_bs[i]) for i in range(self._model_count)]
        logits = tf.add_n(logitss) / float(self._model_count)
        loss = -tf.reduce_sum(tf.one_hot(tf.reshape(self._input.targets, [-1]), self._vocab_size) * tf.log(logits + 1e-12), reduction_indices=[1])
        
        self._cost = tf.reduce_sum(loss) / self._input.batch_size
        self._final_states = states

    def test_lstm(self, session, verbose=False):
        costs = 0.0
        iters = 0
        states = session.run(self._initial_states)

        fetches = {
            "cost": self._cost,
            "final_states": self._final_states
        }
        
        if verbose:
            print("Ensemble Test Epoch")
            
        start_time = time.time()

        for step in range(self._input.epoch_size):
            feed_dict = {}
            for n in range(self._model_count):
                for i, (c, h) in enumerate(self._initial_states[n]):
                    feed_dict[c] = states[n][i].c
                    feed_dict[h] = states[n][i].h

            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            states = vals["final_states"]

            costs += cost
            iters += self._input.num_steps

            if verbose and step % (self._input.epoch_size // 10) == 10:
                print("  %.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / self._input.epoch_size, np.exp(costs / iters), iters * self._input.batch_size / (time.time() - start_time)))

        test_ppl = np.exp(costs / iters)   
        
        if verbose:
            print("Ensemble Test Epoch Perplexity: %.3f" % (test_ppl))
        
        return test_ppl


class LSTMGenerator(object):
    def __init__(self, models, vocab):
        self._models = models
        self._model_count = len(self._models)
        self._vocab = vocab
        
        if self._model_count < 1:
            raise ValueError("Model count must be greater than zero.")

        self._vocab_size = self._models[0].vocab_size
        self._data_type = self._models[0].data_type
            
        for i in range(1, self._model_count):
            if self._models[i].vocab_size != self._vocab_size:
                raise ValueError("Model vocabulary sizes do not match.")
            if self._models[i].data_type != self._data_type:
                raise ValueError("Model data types do not match.")
        if self._vocab_size != self._vocab.vocab_size:
            raise ValueError("Model vocabulary sizes do not match specified vocabulary size.")
            
        nets = [model.lstm_net(model.lstm_cell) for model in self._models]
        
        self._initial_states = [net.zero_state(1, self._data_type) for net in nets]
        
        self._input_word = tf.placeholder(tf.int32, shape=())
        
        inputss = [None] * self._model_count
        for i in range(self._model_count):
            with tf.variable_scope(self._models[i].var_scope):
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable("embedding", [self._vocab_size, self._models[i].hidden_sizes[0]], dtype=self._data_type)
                    inputss[i] = tf.nn.embedding_lookup(embedding, [self._input_word])
        
        states = [None] * self._model_count
        outputss = [None] * self._model_count
        for i in range(self._model_count):
            with tf.variable_scope(self._models[i].var_scope):
                with tf.variable_scope("rnn"):
                    outputss[i], states[i] = nets[i](inputss[i], self._initial_states[i])
        
        softmax_ws = [None] * self._model_count
        softmax_bs = [None] * self._model_count
        for i in range(self._model_count):
            with tf.variable_scope(self._models[i].var_scope):
                softmax_ws[i] = tf.get_variable("softmax_w", [self._models[i].hidden_sizes[-1], self._vocab_size], dtype=self._data_type)
                softmax_bs[i] = tf.get_variable("softmax_b", [self._vocab_size], dtype=self._data_type)
                
        logitss = [tf.nn.softmax(tf.matmul(outputss[i], softmax_ws[i]) + softmax_bs[i]) for i in range(self._model_count)]
        logits = tf.squeeze(tf.add_n(logitss) / float(self._model_count), [0])

        self._output_word = tf.argmax(logits)
        self._final_states = states
        
    def generate_lstm(self, session, input_words, generate_count=100, start_random=False, one_sentence=False, verbose=False):
        if (not input_words) or len(input_words) == 0:
            if start_random:
                output_ids = input_ids = [np.random.randint(0, self._vocab_size)]
            else:
                input_ids = [self._vocab.eos_id]
                output_ids = []
            input_count = 1
        else:
            output_ids = input_ids = self._vocab.words_to_ids(input_words)
            input_count = len(input_ids)
            
        print_cnt = 0
            
        states = session.run(self._initial_states)
                
        fetches = {
            "output_word": self._output_word,
            "final_states": self._final_states
        }
        
        if verbose:
            print("Generator Epoch")
            
        start_time = time.time()
        
        for step in range(input_count):
            if verbose:
                print_cnt += 1
                print("%s " % (self._vocab.ids_to_words([input_ids[step]])[0]), end='')
                if print_cnt % 10 == 0:
                    print("")
        
            feed_dict = {self._input_word: input_ids[step]}
            for n in range(self._model_count):
                for i, (c, h) in enumerate(self._initial_states[n]):
                    feed_dict[c] = states[n][i].c
                    feed_dict[h] = states[n][i].h
            
            vals = session.run(fetches, feed_dict)
            states = vals["final_states"]
            last_out_id = vals["output_word"]
            
        output_ids += [last_out_id]
        gen_idx = 1
        
        if verbose:
            print_cnt += 1
            print("%s " % (self._vocab.ids_to_words([last_out_id])[0]), end='')
            if print_cnt % 10 == 0:
                print("")
        
        while gen_idx < generate_count and ((not one_sentence) or last_out_id != self._vocab.eos_id):
            feed_dict = {self._input_word: last_out_id}
            for n in range(self._model_count):
                for i, (c, h) in enumerate(self._initial_states[n]):
                    feed_dict[c] = states[n][i].c
                    feed_dict[h] = states[n][i].h
            
            vals = session.run(fetches, feed_dict)
            states = vals["final_states"]
            last_out_id = vals["output_word"]
            
            output_ids += [last_out_id]
            gen_idx += 1
            
            if verbose:
                print_cnt += 1
                print("%s " % (self._vocab.ids_to_words([last_out_id])[0]), end='')
                if print_cnt % 10 == 0:
                    print("")
        
        if verbose:
            if print_cnt % 10 != 0:
                print("")
            print("Generator Epoch Speed: %.0f" % ((input_count + gen_idx - 1) / (time.time() - start_time)))
        
        return self._vocab.ids_to_words(output_ids)
        
        
class LSTMTrainer(object):
    def __init__(self, model, train_input, max_epoch, max_max_epoch, max_grad_norm, learning_rate, lr_decay, min_lr, keep_prob, valid_input=None, early_stop=False, max_valid_error=4):
        self._model = model
        self._input = train_input
        
        if self._model.vocab_size != self._input.vocab_size:
            raise ValueError("Model vocabulary size does not match train input vocabulary size.")
        if valid_input and self._model.vocab_size != valid_input.vocab_size:
            raise ValueError("Model vocabulary size does not match valid input vocabulary size.")
        
        self._valid_tester = LSTMTesterSingle(model, valid_input) if valid_input else None
        self._early_stop = early_stop
        self._max_valid_error = max_valid_error
        
        self._max_epoch = max_epoch
        self._max_max_epoch = max_max_epoch
        self._max_grad_norm = max_grad_norm
        self._learning_rate = learning_rate
        self._lr_decay = lr_decay
        self._min_lr = min_lr
        
        attn_cell = self._model.lstm_cell
        if keep_prob < 1:
            def attn_cell(layer_idx):
                return tf.contrib.rnn.DropoutWrapper(self._model.lstm_cell(layer_idx), output_keep_prob=keep_prob)
        
        net = self._model.lstm_net(attn_cell)
        
        self._initial_state = net.zero_state(self._input.batch_size, self._model.data_type)
    
        with tf.variable_scope(self._model.var_scope):
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [self._model.vocab_size, self._model.hidden_sizes[0]], dtype=self._model.data_type)
                inputs = tf.nn.embedding_lookup(embedding, self._input.inputs)
                
        if keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)
            
        inputs = tf.unstack(inputs, num=self._input.num_steps, axis=1)
        with tf.variable_scope(self._model.var_scope):
            outputs, state = tf.contrib.rnn.static_rnn(net, inputs, initial_state=self._initial_state)
            
        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, self._model.hidden_sizes[-1]])
        
        with tf.variable_scope(self._model.var_scope):
            softmax_w = tf.get_variable("softmax_w", [self._model.hidden_sizes[-1], self._model.vocab_size], dtype=self._model.data_type)
            softmax_b = tf.get_variable("softmax_b", [self._model.vocab_size], dtype=self._model.data_type)
        
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._input.targets, [-1])], [tf.ones([self._input.batch_size * self._input.num_steps], dtype=self._model.data_type)])
        
        self._cost = cost = tf.reduce_sum(loss) / self._input.batch_size
        self._final_state = state
        
        self._lr = tf.Variable(0.0, trainable=False)
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, self._model.model_param_vars), max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, self._model.model_param_vars))

        self._new_lr = tf.placeholder(tf.float32, shape=[])
        self._lr_update = tf.assign(self._lr, self._new_lr)
        
    def train_lstm(self, session):
        self._model.init_model_params(session)
        
        if self._valid_tester:
            best_model_params = self._model.save_model_params(session)
        best_valid_ppl = np.inf
        valid_error_count = 0
        
        fetches = {
            "cost": self._cost,
            "final_state": self._final_state,
            "train_op": self._train_op
        }
        
        for n in range(self._max_max_epoch):
            lr_decay = max(self._lr_decay ** max(n + 1 - self._max_epoch, 0.0), self._min_lr)
            session.run(self._lr_update, feed_dict={self._new_lr: self._learning_rate * lr_decay})

            print("Train Epoch: %d Learning rate: %.3f" % (n + 1, session.run(self._lr)))
            
            start_time = time.time()
            costs = 0.0
            iters = 0
            state = session.run(self._initial_state)
            
            for step in range(self._input.epoch_size):
                feed_dict = {}
                for i, (c, h) in enumerate(self._initial_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

                vals = session.run(fetches, feed_dict)
                cost = vals["cost"]
                state = vals["final_state"]    

                costs += cost
                iters += self._input.num_steps

                if step % (self._input.epoch_size // 10) == 10:
                    print("  %.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / self._input.epoch_size, np.exp(costs / iters), iters * self._input.batch_size / (time.time() - start_time)))
            
            print("Train Epoch: %d Perplexity: %.3f" % (n + 1, np.exp(costs / iters)))
            
            if self._valid_tester:
                valid_ppl = self._valid_tester.test_lstm(session)
                
                if valid_ppl < best_valid_ppl:
                    best_model_params = self._model.save_model_params(session)
                    best_valid_ppl = valid_ppl
                    valid_error_count = 0
                else:
                    valid_error_count += 1
                    
                print("Valid Epoch: %d Current Perplexity: %.3f Best Perplexity: %.3f Valid Errors: %d" % (n + 1, valid_ppl, best_valid_ppl, valid_error_count))
                
                if self._early_stop and valid_error_count == self._max_valid_error:
                    print("Valid error max reached! Early stop.")
                    break
        
        if self._valid_tester:
            print("Best Valid Perplexity: %.3f Restoring model parameters..." % (best_valid_ppl))
            
            self._model.load_model_params(session, best_model_params)
            
        print ("Training Done!")

    
def read_words_from_string(str):
    return str.replace("\n", "<eos>").split()
    
def read_words_from_file(file):
    f = open(file, 'r')
    str = f.read()
    f.close()
    return read_words_from_string(str)

def main(_):
    with tf.Graph().as_default():
        if FLAGS.action:
            if FLAGS.action == "build_vocab":
                if not FLAGS.vocab_text:
                    print("No vocabulary text file specified with '--vocab_text'.")
                    sys.exit(-1)
                
                vocab_text = FLAGS.vocab_text
                save_file = FLAGS.save_file
                
                print("Building vocabulary...")
                vocab = LSTMVocabulary(build_words=read_words_from_file(vocab_text))
                
                if save_file:
                    print("Saving vocabulary to file...")
                    vocab.save_vocab_file(save_file)
            
            elif FLAGS.action == "train_model":
                if not FLAGS.vocab_file:
                    print("No vocabulary file specified with '--vocab_file'.")
                    sys.exit(-1)
                if not FLAGS.num_layers:
                    print("No number of layers specified with '--num_layers'.")
                    sys.exit(-1)
                if not FLAGS.hidden_sizes:
                    print("No hidden sizes specified with '--hidden_sizes'.")
                    sys.exit(-1)
                if not FLAGS.train_text:
                    print("No train text file specified with '--train_text'.")
                    sys.exit(-1)
                if not FLAGS.train_batch_size:
                    print("No train batch size specified with '--train_batch_size'.")
                    sys.exit(-1)
                if not FLAGS.train_num_steps:
                    print("No train unroll steps specified with '--train_num_steps'.")
                    sys.exit(-1)
                if not FLAGS.max_epoch:
                    print("No max epoch specified with '--max_epoch'.")
                    sys.exit(-1)
                if not FLAGS.max_max_epoch:
                    print("No max max epoch specified with '--max_max_epoch'.")
                    sys.exit(-1)
                if not FLAGS.max_grad_norm:
                    print("No max gradient norm specified with '--max_grad_norm'.")
                    sys.exit(-1)
                
                vocab_file = FLAGS.vocab_file
                save_file = FLAGS.save_file
                num_layers = FLAGS.num_layers
                hidden_sizes = FLAGS.hidden_sizes
                init_scale = FLAGS.init_scale
                forget_bias = FLAGS.forget_bias
                use_fp16 = FLAGS.use_fp16
                train_text = FLAGS.train_text
                train_batch_size = FLAGS.train_batch_size
                train_num_steps = FLAGS.train_num_steps
                max_epoch = FLAGS.max_epoch
                max_max_epoch = FLAGS.max_max_epoch
                max_grad_norm = FLAGS.max_grad_norm
                learning_rate = FLAGS.learning_rate
                lr_decay = FLAGS.lr_decay
                min_lr = FLAGS.min_lr
                keep_prob = FLAGS.keep_prob
                valid_text = FLAGS.valid_text
                valid_batch_size = FLAGS.valid_batch_size
                valid_num_steps = FLAGS.valid_num_steps
                early_stop = FLAGS.early_stop
                max_valid_error = FLAGS.max_valid_error
                
                if num_layers < 1:
                    print("Number of layers must be greater than zero.")
                    sys.exit(-1)
                hidden_sizes = hidden_sizes.split(",")
                hidden_sizes = [int(x) for x in hidden_sizes]
                if num_layers != len(hidden_sizes):
                    print("Dimension of hidden sizes does not match number of layers.")
                    sys.exit(-1)
                for size in hidden_sizes:
                    if size < 1:
                        print("Hidden layer sizes must be greater than zero.")
                        sys.exit(-1)
                if train_batch_size < 1:
                    print("Train batch size must be greater than zero.")
                    sys.exit(-1)
                if train_num_steps < 1:
                    print("Train unroll steps must be greater than zero.")
                    sys.exit(-1)
                if max_epoch < 1:
                    print("Max epoch must be greater than zero.")
                    sys.exit(-1)
                if max_max_epoch < 1:
                    print("Max max epoch must be greater than zero.")
                    sys.exit(-1)
                if max_grad_norm <= 0:
                    print("Max gradient norm must be greater than zero.")
                    sys.exit(-1)
                if valid_text:
                    if not valid_batch_size:
                        valid_batch_size = train_batch_size
                    elif valid_batch_size < 1:
                        print("Validation batch size must be greater than zero.")
                        sys.exit(-1)
                    if not valid_num_steps:
                        valid_num_steps = train_num_steps
                    elif valid_num_steps < 1:
                        print("Validation unroll steps must be greater than zero.")
                        sys.exit(-1)
                    if max_valid_error < 1:
                        print("Max validation errors must be greater than zero.")
                        sys.exit(-1)
                        
                print("Loading vocabulary from file...")
                vocab = LSTMVocabulary(load_file=vocab_file)
                
                print("Creating new model...")
                model = LSTMModel(vocab.vocab_size, num_layers, hidden_sizes, init_scale, forget_bias=forget_bias, use_fp16=use_fp16)
                
                print("Creating train input...")
                train_input = LSTMInput(vocab, read_words_from_file(train_text), train_batch_size, train_num_steps)
                
                if valid_text:
                    print("Creating validation input...")
                    valid_input = LSTMInput(vocab, read_words_from_file(valid_text), valid_batch_size, valid_num_steps)
                else:
                    valid_input = None
                
                print("Creating LSTM trainer...")
                trainer = LSTMTrainer(model, train_input, max_epoch, max_max_epoch, max_grad_norm, learning_rate, lr_decay, min_lr, keep_prob, valid_input=valid_input, early_stop=early_stop, max_valid_error=max_valid_error)
                
                print("Creating TensorFlow session...")
                with tf.Session() as sess:
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    
                    print("Initializing model parameters...")
                    model.init_model_params(sess)
                    
                    print("Training model...")
                    trainer.train_lstm(sess)
                    
                    if save_file:
                        print("Saving model to file...")
                        model.save_model_file(sess, save_file)
                        
                    coord.request_stop()
                    coord.join(threads)
                    
            elif FLAGS.action == "test_model_single":
                if not FLAGS.vocab_file:
                    print("No vocabulary file specified with '--vocab_file'.")
                    sys.exit(-1)
                if not FLAGS.model_file:
                    print("No model file specified with '--model_file'.")
                    sys.exit(-1)
                if not FLAGS.test_text:
                    print("No test text file specified with '--test_text'.")
                    sys.exit(-1)
                if not FLAGS.test_batch_size:
                    print("No test batch size specified with '--test_batch_size'.")
                    sys.exit(-1)
                if not FLAGS.test_num_steps:
                    print("No test unroll steps specified with '--test_num_steps'.")
                    sys.exit(-1)
                    
                vocab_file = FLAGS.vocab_file
                model_file = FLAGS.model_file
                test_text = FLAGS.test_text
                test_batch_size = FLAGS.test_batch_size
                test_num_steps = FLAGS.test_num_steps
                
                if test_batch_size < 1:
                    print("Test batch size must be greater than zero.")
                    sys.exit(-1)
                if test_num_steps < 1:
                    print("Test unroll steps must be greater than zero.")
                    sys.exit(-1)
                    
                print("Loading vocabulary from file...")
                vocab = LSTMVocabulary(load_file=vocab_file)
                
                print("Creating model from file...")
                model = LSTMModel.create_model_from_file(model_file)
                
                print("Creating test input...")
                test_input = LSTMInput(vocab, read_words_from_file(test_text), test_batch_size, test_num_steps)
                
                print("Creating LSTM tester...")
                tester = LSTMTesterSingle(model, test_input)
                
                print("Creating TensorFlow session...")
                with tf.Session() as sess:
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    
                    print("Loading model parameters from file...")
                    model.load_model_file(sess, model_file)
                    
                    print("Testing model...")
                    tester.test_lstm(sess, verbose=True)
                    
                    coord.request_stop()
                    coord.join(threads)
                    
            elif FLAGS.action == "test_model_ensemble":
                if not FLAGS.vocab_file:
                    print("No vocabulary file specified with '--vocab_file'.")
                    sys.exit(-1)
                if not FLAGS.model_file:
                    print("No model file specified with '--model_file'.")
                    sys.exit(-1)
                if not FLAGS.test_text:
                    print("No test text file specified with '--test_text'.")
                    sys.exit(-1)
                if not FLAGS.test_batch_size:
                    print("No test batch size specified with '--test_batch_size'.")
                    sys.exit(-1)
                if not FLAGS.test_num_steps:
                    print("No test unroll steps specified with '--test_num_steps'.")
                    sys.exit(-1)
                    
                vocab_file = FLAGS.vocab_file
                model_file = FLAGS.model_file
                test_text = FLAGS.test_text
                test_batch_size = FLAGS.test_batch_size
                test_num_steps = FLAGS.test_num_steps
                
                model_files = model_file.split("|")
                model_file_count = len(model_files)
                if model_file_count < 1:
                    print("Model file count must be greater than zero.")
                    sys.exit(-1)
                if test_batch_size < 1:
                    print("Test batch size must be greater than zero.")
                    sys.exit(-1)
                if test_num_steps < 1:
                    print("Test unroll steps must be greater than zero.")
                    sys.exit(-1)
                    
                print("Loading vocabulary from file...")
                vocab = LSTMVocabulary(load_file=vocab_file)
                
                print("Creating models from files...")
                models = [LSTMModel.create_model_from_file(model_files[i]) for i in range(model_file_count)]
                
                print("Creating test input...")
                test_input = LSTMInput(vocab, read_words_from_file(test_text), test_batch_size, test_num_steps)
                
                print("Creating LSTM ensemble tester...")
                ensemble_tester = LSTMTesterEnsemble(models, test_input)
                
                print("Creating TensorFlow session...")
                with tf.Session() as sess:
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    
                    print("Loading model parameters from files...")
                    for i in range(model_file_count):
                        models[i].load_model_file(sess, model_files[i])
                    
                    print("Testing model ensemble...")
                    ensemble_tester.test_lstm(sess, verbose=True)
                    
                    coord.request_stop()
                    coord.join(threads)
                    
            elif FLAGS.action == "generate":
                if not FLAGS.vocab_file:
                    print("No vocabulary file specified with '--vocab_file'.")
                    sys.exit(-1)
                if not FLAGS.model_file:
                    print("No model file specified with '--model_file'.")
                    sys.exit(-1)
                    
                vocab_file = FLAGS.vocab_file
                model_file = FLAGS.model_file
                gen_input_text = FLAGS.gen_input_text
                gen_count = FLAGS.gen_count
                gen_start_random = FLAGS.gen_start_random
                gen_one_sentence = FLAGS.gen_one_sentence
                    
                model_files = model_file.split("|")
                model_file_count = len(model_files)
                if model_file_count < 1:
                    print("Model file count must be greater than zero.")
                    sys.exit(-1)
                if gen_input_text:
                    gen_input = read_words_from_file(gen_input_text)
                else:
                    gen_input = None
                if gen_count < 1:
                    print("Generate count must be greater than zero.")
                    sys.exit(-1)
                    
                print("Loading vocabulary from file...")
                vocab = LSTMVocabulary(load_file=vocab_file)
                
                print("Creating models from files...")
                models = [LSTMModel.create_model_from_file(model_files[i]) for i in range(model_file_count)]
                
                print("Creating LSTM generator...")
                generator = LSTMGenerator(models, vocab)
                
                print("Creating TensorFlow session...")
                with tf.Session() as sess:
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    
                    print("Loading model parameters from files...")
                    for i in range(model_file_count):
                        models[i].load_model_file(sess, model_files[i])
                    
                    print("Generating...")
                    generator.generate_lstm(sess, gen_input, generate_count=gen_count, start_random=gen_start_random, one_sentence=gen_one_sentence, verbose=True)
                    
                    coord.request_stop()
                    coord.join(threads)
                    
            else:
                print("Invalid action specified, see '--help'.")
                sys.exit(-1)
                    
        else:
            print("No action specified, see '--help'.")
            sys.exit(-1)
            

if __name__ == "__main__":
  tf.app.run()

