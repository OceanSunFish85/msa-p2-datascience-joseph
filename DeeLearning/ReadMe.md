# CIFAR-10 Classification using CNN

This deep learning project utilizes a Convolutional Neural Network (CNN) to classify images in the CIFAR-10 dataset. The project structure includes trained model files stored in the `models` directory, evaluation logs and visualizations in the `results` directory, and a Jupyter Notebook detailing each step of the training process, analysis, and a summary in the `ipynb` directory.

## Project Structure

- `models/`: Contains the trained model files.
- `results/`: Includes textual evaluation logs and visualizations (e.g., confusion matrices, ROC curves).
- `ipynb/`: Jupyter Notebook detailing the training process, analysis, and summary.
- `train.py`: Script to train the model.
- `evaluate.py`: Script to generate evaluation results.
- `predict.py`: Script to perform predictions on new data.
- `requirements.txt`: Automatically generated from the Conda environment used to train the model.

## Training the Model

Due to the extended time required for training the model within a Jupyter Notebook, it is recommended to execute the training process using the `train.py` script. This script provides an efficient way to train the model outside of the notebook environment.

```bash
python train.py
```

## Evaluating the Model

To generate evaluation results, use the evaluate.py script. This will produce evaluation metrics and visualizations stored in the results directory.

```bash
python evaluate.py
```

## Making Predictions

To perform predictions on new data, use the predict.py script. This script will load the trained model and make predictions on the provided dataset.

```bash
python predict.py
```

## Setup and Installation

The project environment is based on Anaconda and PyTorch. To set up the environment, first install the required packages:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Additionally, install other dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```
predictions.csv is a copy of the results submitted in the kaggle competition

requirements.txt was generated directly from the Conda environment used to train the model, ensuring all necessary dependencies are included.
