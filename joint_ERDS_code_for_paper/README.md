# Early Exiting-Aware Joint Resource Allocation and DNN  Splitting for Multi-Sensor Digital Twin in Edge-Cloud  Collaborative System (2024)
Python implementation of the early exiting-aware joint resource allocation and DNN splitting for multi-sensor digital twin in edge-cloud collaborative system.

## Requirements
Due to property issues, we cannot provide the exact values related to the trained model, early exit probabilities, and the execution time. 
Before running the simulation, please follow the instructions and achieve trained model with essential files.

```
1. First, we need to train the DNN model with early exit structures. Please check the 'model.py' file. 
   'class DDNN' is for the ResNet based model explained in the paper. / 'class DDNN_DenseNet' is for the DenseNet based model explained in the supplementary paper.

2.Then run 'train_model_for_ERDS.py' to train and evaluate the selected model with the dataset (CIFAR-10 / CIFAR-100 / IMAGENET)
  While training the model, we must store the following information that will be used by the algorithm.
  1) The trained model should be saved. ex) 'models/sync_test_model_Cifar10:dense.pth'
  2) The number of times each image data early exited at a particular exit point should be saved. ex) 'files/sync_test_exit_place_CIFAR10.pkl'
  3) The execution time of the model for each image data should be measured and saved. ex) 'files/sync_test_exit_time_CIFAR10.pkl'
```


# Usage
```
run.py --scenario SCENARIO --scenario2 SCENARIO2 --digital_twin DIGITALTWIN
```
Required argument:
*  --scenario: types of scenario {original: orignal scenario, dynamic_tr: dynamic transmission time scenarios, dynamic_cloud: dynamic cloud execution time scenarios, dynamic_edge: dynamic edge execution time scenarios}
*  --scenario2: types of specific scenario of dynamic_tr/dynamic_cloud/dynamic_edge  {0: 0.5 / 1.5 / 1.5, 1: 1 / 1 / 1, 2: 1.5 / 0.666 / 0.666, 3: 2 / 0.5 / 0.5, 4: 3 / 0.333 / 0.5}
*  --digital_twin: the numbers of digital twins that need to be synchronized {1, 2, 3}

Example 
```
python run.py --scenario original --scenario2 0 --digital_twin 3
```
Parameters of scenarios can be adjusted in: 
* run.py:get_parser()



# Results
- This implementation produces the data used in Figures 5 ~ 7, and Table 4 ~ 5 in the paper.
- This implementation also produces the data used in Table 1 ~ 2 in Supplementary Material of the paper, which is considering the DenseNet model.
- Plotting data is not implemented here. We plotted the figures by using MATLAB. The results are saved in a form of '.mat' files


