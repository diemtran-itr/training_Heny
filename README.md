AI Model for Classification of Rhythms:

This repository contains code for an AI model that can classify electrocardiogram (ECG) signals into different rhythms. The code is divided into several Python files, each responsible for a different aspect of the system. The code uses the MIT-BIH Arrhythmia Database and PyTorch.

Python Files:
main.py: The main entry point for the code. Calls the train and evaluate functions to train and evaluate the model.
model.py: Contains the ECGModel class that defines the neural network architecture.
train.py: Defines the train function that trains the model on the training set.
evaluation.py: Defines the evaluate function that evaluates the trained model on the test set.
data_utils.py: Contains utility functions for loading and preprocessing the dataset.

Usage:
To use the code, you can follow these steps:
1.Clone the repository to your local machine.
2.Install the required dependencies listed in requirements.txt.
3.Run the main.py file.

The main.py file will call the train function defined in train.py to train the model on the training set. After training, it will call the evaluate function defined in evaluation.py to evaluate the trained model on the test set. The evaluation results will be printed to the console.

You can modify different aspects of the system by modifying the corresponding Python files. For example, you can experiment with different neural network architectures by modifying the ECGModel class in model.py. You can also change the dataset or preprocessing steps by modifying the code in data_utils.py.

Acknowledgments:
This code uses the MIT-BIH Arrhythmia Database, which is available from Physionet. Please see the data/README file for more details.

References:
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Bio 20(3):45-50 (May-June 2001). DOI: 10.1109/51.932724
