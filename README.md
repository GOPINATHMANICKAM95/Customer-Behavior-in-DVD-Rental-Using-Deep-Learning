# Predicting Customer Behavior in DVD Rental with Deep Learning (AWS Deployment)

This project predicts customer behavior for a DVD rental business using deep learning models deployed on AWS. It leverages historical rental data to forecast customer churn, rental frequency, and genre preferences.

## Features
- **Deep Learning Model**: A neural network built using TensorFlow/Keras for predicting customer behavior.
- **Data Preprocessing**: Includes cleaning, feature engineering, and scaling customer rental data to prepare it for the model.
- **Customer Insights**: The model predicts key metrics such as:
  - Rental frequency
  - Customer churn likelihood
  - Popular DVD genres and categories
- **Scalable Predictions**: The AWS deployment ensures the system can handle large amounts of data and provide predictions on-demand.

## Project Structure
- **data/**: Contains sample datasets and scripts for data preprocessing and feature engineering.
- **model/**: Includes the code for building, training, and evaluating the deep learning model.
- **encode/**: Contains encoded files for feature representation.

## Technologies Used
- **Python**: Main programming language used for data processing and model development.
- **TensorFlow/Keras**: For building and training the deep learning model.
- **Jupyter Notebooks**: For experimentation, data exploration, and model training.

## How to Run
1. Clone the repository:
   ```bash
   git clone 
   cd dvd-rental-prediction
