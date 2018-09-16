## Neural Turing Machine

A basic TensorFlow implementation of a [Neural Turing Machine](https://arxiv.org/abs/1410.5401).  
The code is heavily based on the implementation by [carpedm20](https://github.com/carpedm20/NTM-tensorflow) and [camigord](https://github.com/camigord/Neural-Turing-Machine).



### Copy Task
I've only implemented the copy task as of yet  
1. Currently, only works with `num_write_heads=1`  
2. Sequence lengths >= 20 start showing vanishing gradients

#### Prerequisites
* Python 3.5
* TensorFlow >= 1.2.0
* NumPy
* Matplotlib

#### Usage
~~~~
python copy_task.py [-h] [--mode MODE] [--model MODEL]
                    [--iterations ITERATIONS]
                    [--restore_training RESTORE_TRAINING]
                    [--learning_rate LEARNING_RATE] [--momentum MOMENTUM]
                    [--decay DECAY] [--save_location SAVE_LOCATION]
                    [--size_input SIZE_INPUT]
                    [--size_input_sequence SIZE_INPUT_SEQUENCE]
                    [--size_output SIZE_OUTPUT]
                    [--num_memory_vectors NUM_MEMORY_VECTORS]
                    [--size_memory_vector SIZE_MEMORY_VECTOR]
                    [--num_read_heads NUM_READ_HEADS]
                    [--num_write_heads NUM_WRITE_HEADS]
                    [--size_conv_shift SIZE_CONV_SHIFT]
                    [--batch_size BATCH_SIZE]
~~~~
| Command Line Arguments | Options                  | Description |
| ---------------------- | ------------------------ | ----------- |
| mode                   | `train`, `test`          | training mode vs testing mode |
| model                  | `lstm`, `feedforward`    | LSTM Controller vs Feedforward Controller | 
| iterations             | `<Int>`                  | total number of training steps |
| restore_training       | `<Bool>`                 | if `True`, restore training from checkpoints |
| learning_rate          | `<Float>`                | learning rate |
| decay                  | `<Float>`                | decay rate for RMSProp |
| size_input             | `<Int>`                  | size of input vector |
| size_input_sequence    | `<Int>`                  | length of sequence containing input vectors |
| size_output            | `<Int>`                  | size of output vector |
| num_memory_vectors     | `<int>`                  | total number of memory slots (*N* from the paper)
| size_memory_vector     | `<Int>`                  | size of each memory slot (*M* from the paper)
| num_read_heads         | `<Int>`                  | total number of read heads |
| num_write_heads        | `<Int>`                  | total number of write heads (Note: currently only works for **1** write head) |
| size_conv_shift        | `<Int>`                  | convolutional shift value (Eg: if `2`, the shift weightings vector has size `2 * 2 + 1 = 5` |
| batch_size             | `<Int>`                  | batch size |

#### Results
The network was trained with the LSTM controller for `iterations=100000`.  
The number of reading heads and the number of write heads were both 1. (The code currently only works for `num_write_heads=1`)  
The sequence length for training was 10.
  
The training loss as a function of iterations is as follows:  
![training loss](https://github.com/kelaaditya/machine-learning-algorithms/blob/master/neural_turing_machine/data/training_loss.png)


The copy task was tested with input data (with the starting and ending delimiters):  
![input data](https://github.com/kelaaditya/machine-learning-algorithms/blob/master/neural_turing_machine/data/input_data.png)

The output from the NTM was:  
![output data](https://github.com/kelaaditya/machine-learning-algorithms/blob/master/neural_turing_machine/data/ntm_output.png)






