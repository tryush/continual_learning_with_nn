# continual_learning_with_nn
This project is about implementation of continual learning using neural network.

In this project, I am given 3 dataset namely G1, G2 and G3, each dataset has 2 columns namely text and tags where tags is basically named entity recognition.
1st task is to train a model on G1 train data and test it on G1 test data and again train the 1st model on G2 data along with 100 samples of G1 data so that trained model does not forget the learning from the G1 dataset and again apply the same process with G3 data.

2nd Task is to create a model on combined G1, G2 and G3 dataset and test it on combined test data of G1, G2 and G3.
