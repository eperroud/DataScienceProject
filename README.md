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

[Detailed description of your machine learning project, its goals, and main functionalities.]

## File description
1. Folder [data](data) :
   - This one contains the needed DataTransformation.ipynb to perform all necessary transformation
   - [`data/final_training.csv`](data/final_training.csv) is the training data with all necessary transformations already done
   - [`data/final_test.csv`](data/final_test.csv) is the test data with all necessary transformations already done
2. Folder [Predictions](Predictions) :
   - File [`DT_GS.ipynb`](Predictions/DT_GS.ipynb) uses a Decision Tree method to predict text difficulty
   - File [`RandomForest.ipynb`](Predictions/RandomForest.ipynb) uses a Random Forest method to predict text difficulty
   - File [`Ridge.ipynb`](Predictions/Ridge.ipynb) uses a Ridge method to predict text difficulty
   - File [`LinReg.ipynb`](Predictions/LinReg.ipynb) uses a Linear regression method to predict text difficulty
   - File [`LogReg_Cam(0,556).ipynb`](Predictions/LogReg_Cam(0,556).ipynb) uses a Logistic regression method to predict text difficulty taking BERT embeddings as explanatory variable among others
   - File [`LogReg_Cam_COG(0_558).ipynb`](Predictions/LogReg_Cam_COG(0_558).ipynb) uses a Logistic regression method to predict text difficulty taking BERT embeddings and cognates as explanatory variable among others
   - File [`LogReg_Cam_COG_Full(0,56).ipynb`](Predictions/LogReg_Cam_COG_Full(0,56).ipynb) uses a Logistic regression method to predict text difficulty taking BERT embeddings and cognates as explanatory variable among others with re-training on the full dataset
   - File [`LogReg_Cam_COG_Full(0,562).ipynb`](Predictions/LogReg_Cam_COG_Full(0,562).ipynb) uses a Logistic regression method to predict text difficulty taking BERT embeddings and cognates as explanatory variable among others with re-training on the full dataset






[Step-by-step instructions on how to install and set up your project, including any dependencies or requirements.]

## Usage
In order to run the code properly you have two options : 
  1. Run the DataTransformation.ipynb first which will provide two new datasets with all necessary modifications :       data/final_test.csv and data/final_training.csv

  2. Directly import the existing [`data/final_test.csv`](data/final_test.csv) and [`data/final_training.csv`](data/final_training.csv) available from the data folder in the repository.

Once this is done, you can select which of the model you would like to use to make predictions in the folder Predictions. 

## Data

[Information about the dataset(s) used in your project, such as source, format, and preprocessing steps.]

## Model Architecture

[Details about the architecture of your machine learning model, potentially including a diagram for better understanding.]

## Training

[Instructions on how to train the model or provide information about a pre-trained model. Include hyperparameters and relevant training details.]

## Results

[Share the results of your model, including performance metrics and visualizations. Compare your model's performance to other benchmarks if applicable.]

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
