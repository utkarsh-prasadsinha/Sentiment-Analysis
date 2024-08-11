# Sentiment Analysis Application

This project is a sentiment analysis application that analyzes text input and determines whether the sentiment is positive or negative. The application uses a deep learning model trained on text data, with results stored in MongoDB.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results Storage in MongoDB](#results-storage-in-mongodb)
- [Testing](#testing)
- [Deployment](#deployment)

## Introduction

This application is built using Flask, TensorFlow, and MongoDB. It allows users to input text, analyzes the sentiment, and stores the analysis results in MongoDB for future reference.

## Features
- Sentiment analysis using a trained LSTM model.
- User-friendly web interface.
- Results storage in MongoDB.

## System Requirements
- Python 3.7+
- TensorFlow 2.x
- Flask
- MongoDB

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/username/repository.git
cd repository
```


### 2. Navigate to the project directory:
```bash
cd your-repo-name
```

### 3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Execution

### Train the model
```bash
python train_model.py
```

### Run the application
```bash
python main.py
```