# CGN2Train Tool

This tool reduces size of and prepares the CGN dataset for use by common language modelling toolkit such as SRILM and RNNLM. The TensorFlow LSTM LM tool provided in this repository also uses this format.

This tool is written in C#. The Visual Studio project files can be found in the 'source' directory; a pre-compiled executable for Windows is available in the 'bin' directory.

Usage:
    
    cgn2train <cgn_input_directory> <output_directory> <reduced_word_count> <reduced_vocabulary_size> <validation_fraction> <test_fraction> <random_seed>