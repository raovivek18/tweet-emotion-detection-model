# Tweet Emotion Detection Model

## Overview

The **Tweet Emotion Detection Model** is a Python-based application leveraging **Natural Language Processing (NLP)** techniques to classify the emotions conveyed in tweets. This tool aims to analyze tweet content and detect emotions such as **joy**, **sadness**, **fear**, **neutral**, **anger**, and **shame**. It uses a trained machine learning model and offers a user-friendly interface powered by **Streamlit**.

This project addresses the growing need for emotion analysis in social media, where users frequently express their emotions through text.

## Features

- **Tweet Link Input**: Accepts a tweet URL, scrapes the content of the tweet, and analyzes its emotion.
- **Emotion Prediction**: Predicts the emotion from the input text or tweet with a confidence score.
- **Visualization**: Displays a bar graph of the probabilities for each emotion.

## Technologies Used

- **Python**: Backend and machine learning development.
- **Streamlit**: Web application framework for interactive user interfaces.
- **Scikit-learn**: Machine learning library used for model training.
- **Joblib**: For saving and loading the trained model.
- **Altair**: For creating visualizations.
- **Requests**: For fetching tweet content from URLs.

## Project Structure

- **`app.py`**: The main file for the Streamlit web application.
- **`text_emotion.pkl`**: Pre-trained model file saved using Joblib.
- **`tweet_dataset.csv`**: Cleaned and processed dataset used to train the model.
- **`Text_Emotion_Detection.ipynb`**: Jupyter Notebook containing model training, data preprocessing, and evaluation.

## Dataset Details

The dataset used for training contains text samples labeled with emotions. It underwent several preprocessing steps:
- Removal of stop words, user handles, and special characters using the **Neat Text** library.
- Addition of a column for cleaned text to prepare the data for model training.

## Model Training

The model was trained using Scikit-learn with the following algorithms:
1. **Support Vector Machines (SVM)**: For robust classification by finding optimal decision boundaries.
2. **Random Forest Classifier**: Utilizes an ensemble of decision trees for accurate predictions.
3. **Logistic Regression**: Effective for categorical output prediction.

The dataset was split:
- **70% Training Set**: Used for model training.
- **30% Testing Set**: Used for evaluating the model's performance.

## Deployment

The model is deployed as a **Streamlit** web application:
1. **User Interface**: Includes fields for tweet URL or text input.
2. **Output**: Displays the detected emotion, a corresponding emoji, and a confidence score.
3. **Bar Graph**: Visualizes the probabilities for all detected emotions.

## How to Run

### 1. Clone the repository:
```bash
git clone https://github.com/raovivek18/tweet-emotion-detection-model
cd tweet-emotion-detection-model

2. Installation:
Install dependencies
Navigate to the project directory and install the required Python dependencies:

bash
Copy code
pip install -r requirements.txt
3. Train the model:
Run the Text_Emotion_Detection.ipynb notebook to train the model.

4. Run the application:
bash
Copy code
streamlit run app.py
