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
   - This one contains the needed DataTransformation.ipynb to perform all necessary transformation
   - [`data/final_training.csv`](data/final_training.csv) is the training data with all necessary transformations already done
   - [`data/final_test.csv`](data/final_test.csv) is the test data with all necessary transformations already done
     
2. Folder [Predictions](Predictions) : Different models predicting text difficulty
   - File [`DT_GS.ipynb`](Predictions/DT_GS.ipynb) uses a Decision Tree method 
   - File [`RandomForest.ipynb`](Predictions/RandomForest.ipynb) uses a Random Forest method 
   - File [`Ridge.ipynb`](Predictions/Ridge.ipynb) uses a Ridge method 
   - File [`LinReg.ipynb`](Predictions/LinReg.ipynb) uses a Linear regression method 
   - File [`LogReg_Cam(0,556).ipynb`](Predictions/LogReg_Cam(0,556).ipynb) uses a Logistic regression method taking BERT embeddings as explanatory variable among others
   - File [`LogReg_Cam_COG(0_558).ipynb`](Predictions/LogReg_Cam_COG(0_558).ipynb) uses a Logistic regression method taking BERT embeddings and cognates as explanatory variable among others
   - File [`LogReg_Cam_COG_Full(0,56).ipynb`](Predictions/LogReg_Cam_COG_Full(0,56).ipynb) uses a Logistic regression method taking BERT embeddings and cognates as explanatory variable among others with re-training on the full dataset
   - File [`LogReg_Cam_COG_Full(0,562).ipynb`](Predictions/LogReg_Cam_COG_Full(0,562).ipynb) uses a Logistic regression method taking BERT embeddings and cognates as explanatory variable among others with re-training on the full dataset



## Usage
In order to run the code properly you have two options : 
  1. Run the DataTransformation.ipynb first which will provide two new datasets with all necessary modifications : [`data/final_test.csv`](data/final_test.csv) and [`data/final_training.csv`](data/final_training.csv)

  2. Directly import the existing [`data/final_test.csv`](data/final_test.csv) and [`data/final_training.csv`](data/final_training.csv) available from the data folder in the repository.

Once this is done, you can select which of the model you would like to use to make predictions in the folder [Predictions](Predictions). 

## Data

[Information about the dataset(s) used in your project, such as source, format, and preprocessing steps.]

## Model Architecture

[Details about the architecture of your machine learning model, potentially including a diagram for better understanding.]

## Training

[Instructions on how to train the model or provide information about a pre-trained model. Include hyperparameters and relevant training details.]

## Results

The model with the best final accuracy on the unlabelled test set is [`LogReg_Cam_COG_Full(0,562).ipynb`](Predictions/LogReg_Cam_COG_Full(0,562).ipynb) implying an accuracy of 0.56 

## Model Evaluation Metrics

| Model                | Logistic Regression | kNN | Decision Tree | Random Forests |Linear Regression | Neural Network | Ridge 
|----------------------|---------------------|-----|---------------|-----------------|----------------------|--------------|--------|
| Precision            | 0.562                | 0.| 0.408          | 0.449            | -                 |              |     
| Recall               | 0.565                | 0.| 0.390          | 0.503           | -                 |              |
| F1-score             | 0.563                | 0.| 0.389          | 0.499            | -                 |              |
| Accuracy             | 0.56                | 0.| 0.39          | 0.50            | 0.74                 |              | 0.76

## UI with streamlit - application
Here is an application using our best model in which you can enter any text to get it evaluated and find its difficulty level. 

## Video
Here is a video explaining in more details the ideas and implementation of the model : 
## Dependencies

[List all dependencies and versions required to run your project. Include links to relevant documentation.]

## Contributing

[Guidelines for others who want to contribute to your project. Include information on submitting bug reports or feature requests.]

## License

[Specify the license under which your project is released. Include a link to the license file for more details.]

## Contact

[Provide your contact information or ways for others to reach out to you. Include links to your social media profiles or personal website.]

## Acknowledgments

[Give credit to individuals, organizations, or tools that contributed to your project.]
