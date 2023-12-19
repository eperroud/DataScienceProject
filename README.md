# DataScienceProject
DataScience Project

# Detecting the difficulty level of French texts

This project aims to create a model that can detect the level of difficult of a french text based on the CECR grading method.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Description
This project aims to predict text difficulty. The methodolgy is to train your model on the training dataset and finally to predict text difficulty on a text dataset that will be uploaded on a Kaggle competition and give the final accuracy of the model. 
The steps are the following : 
1. Perform data transformation on training and tests set to find the best explanatory variables
2. Upload these datasets and train different models with the training dataset
3. Choose model with highest accuracy
4. Predict text difficulty level on the test dataset
5. Upload it on the Kaggle competition to see resulting accuracy
   
## File description
All jupyter notebooks contain necessary installations within given code

1. Folder [data](data) : Contains datasets and a jupyter notebook making transformation
   - This one contains the needed [`Data_Transformation_Final.ipynb`](data/Data_Transformation_Final.ipynb) to perform all necessary transformation
   - [`final_training.csv`](data/final_training.csv) is the training data with all necessary transformations already done
   - [`final_test.csv`](data/final_test.csv) is the test data with all necessary transformations already done
     
2. Folder [Predictions](Predictions) : Different models predicting text difficulty
   - File [`DT_GS.ipynb`](Predictions/DT_GS.ipynb) uses a Decision Tree method 
   - File [`RandomForest.ipynb`](Predictions/RandomForest.ipynb) uses a Random Forest method 
   - File [`Ridge.ipynb`](Predictions/Ridge.ipynb) uses a Ridge method 
   - File [`LinReg.ipynb`](Predictions/LinReg.ipynb) uses a Linear regression method
   - File [`kNNReg.ipynb`](Predictions/kNNReg.ipynb) uses a kNN regression method
   - File [`nnReg.ipynb`](Predictions/nnReg.ipynb) uses a Neural Network method 

   - File [`LogReg_Cam(0,556).ipynb`](Predictions/LogReg_Cam(0,556).ipynb) uses a Logistic regression method taking BERT embeddings as explanatory variable among others
   - File [`LogReg_Cam_COG(0,558).ipynb`](Predictions/LogReg_Cam_COG(0,558).ipynb) uses a Logistic regression method taking BERT embeddings and cognates as explanatory variable among others
   - File [`LogReg_Cam_COG_Full(0,56).ipynb`](Predictions/LogReg_Cam_COG_Full(0,56).ipynb) uses a Logistic regression method taking BERT embeddings and cognates as explanatory variable among others with re-training on the full dataset
   - File [`LogReg_Cam_COG_Full(0,562).ipynb`](Predictions/LogReg_Cam_COG_Full(0,562).ipynb) uses a Logistic regression method taking BERT embeddings and cognates as explanatory variable among others with re-training on the full dataset



## Usage
In order to run the code properly you have two options : 
  1. Run [`Data_Transformation_Final.ipynb`](data/Data_Transformation_Final.ipynb) first which will provide two new datasets with all necessary modifications : [`data/final_test.csv`](data/final_test.csv) and [`data/final_training.csv`](data/final_training.csv)

  2. Directly import the existing [`data/final_test.csv`](data/final_test.csv) and [`data/final_training.csv`](data/final_training.csv) available from the data folder in the repository.

Once this is done, you can select which of the model you would like to use to make predictions in the folder [Predictions](Predictions). 

## Data

Initial training and unlabelled dataset were taken directly from the kaggle competition interface. 

## Results

The model with the best final accuracy on the unlabelled test set is [`LogReg_Cam_COG_Full(0,562).ipynb`](Predictions/LogReg_Cam_COG_Full(0,562).ipynb) implying an accuracy of 0.562 on the kaggle competition. 

The confusion matrix of our best predictions on the training set is the following : 

|                  | Level 0 - A1 | Level 1 - A2 | Level 2 - B1 | Level 3 - B2 | Level 4 - C1 | Level 5 - C2 |
|------------------|---------|---------|---------|---------|---------|---------|
| **Level 0 - A1**      | 125     | 33      | 7       | 1       | 0       | 0       |
| **Level 1 - A2**      | 29      | 83      | 41      | 2       | 2       | 1       |
| **Level 2 - B1**      | 17      | 44      | 79      | 12      | 7       | 7       |
| **Level 3 - B2**      | 4       | 2       | 17      | 82      | 36      | 12      |
| **Level 4 - C1**      | 0       | 0       | 4       | 29      | 75      | 44      |
| **Level 5 - C2**      | 0       | 1       | 3       | 23      | 37      | 101     |


## Model Evaluation Metrics

| Model                | Logistic Regression | kNN | Decision Tree | Random Forests |Linear Regression | Neural Network | Ridge 
|----------------------|---------------------|-----|---------------|-----------------|----------------------|--------------|--------|
| Precision            | 0.56                | 0.11| 0.41          | 0.45           | 0.52                |     0.51         | 0.52    
| Recall               | 0.57                | 017.| 0.39         | 0.50           | 0.47                 |       0.46       | 0.47
| F1-score             | 0.56                | 0.09| 0.39          | 0.50            | 0.48                 |       0.45       | 0.47
| Accuracy             | 0.56                | 0.17| 0.39          | 0.50            | 0.74                 |   0.46           | 0.76

## UI with streamlit - application
Here is an application using our best model in which you can enter any text to get it evaluated and find its difficulty level. 

## Video
Here is a video explaining in more details the ideas and implementation of the model : 
## Dependencies

- **spacy:** 3.2.0
- **sentencepiece:** 0.1.96
- **transformers:** 4.12.2

  
## Contributing

[Guidelines for others who want to contribute to your project. Include information on submitting bug reports or feature requests.]

## License

[Specify the license under which your project is released. Include a link to the license file for more details.]

## Contact

[Provide your contact information or ways for others to reach out to you. Include links to your social media profiles or personal website.]

## Acknowledgments

[Give credit to individuals, organizations, or tools that contributed to your project.]
