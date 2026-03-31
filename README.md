# Autonomous Vehicle Trajectory Prediction

## Project Overview
Hey! This repository contains our code for the autonomous vehicle trajectory prediction challenge. We implemented two distinct approaches to tackle this: a Social LSTM baseline and an advanced Transformer-based model with social pooling for multi-modal predictions.

### The Challenge: Intent & Trajectory Prediction
**Focus:** Behavioral AI & Temporal Modeling

In an L4 urban environment, reacting to where a pedestrian is isn't enough; the vehicle must predict where they will be. The challenge was to develop a model that predicts the future coordinates (next 3 seconds) of pedestrians and cyclists based on 2 seconds of past motion. 

**Key Objectives & Focus Areas:**
- **Process temporal sequence data** (coordinates/velocity).
- **Account for "Social Context"** (how pedestrians and vehicles avoid each other).
- **Generate multi-modal predictions** (e.g., the 3 most likely paths).
- **Focus Areas:** LSTMs/GRUs, Transformers, Social-Pooling layers, and Goal-conditioned prediction.

**Expected Outcomes & Evaluation Metrics:**
- **Outcome:** A model that inputs a history of (x, y) coordinates and outputs a sequence of predicted future (x, y) points.
- **ADE (Average Displacement Error):** Mean Euclidean distance between predicted and ground truth points.
- **FDE (Final Displacement Error):** Distance between the final predicted point and the actual final position.

## Model Architecture
We experimented with two main architectures for this project:

**1. Social LSTM (`Social lstm.py`)**
This is our baseline model. It takes the past coordinates of an agent and uses a spatial grid to encode the positions of nearby neighbors. 
- The target's positions and the social grid are embedded through linear layers.
- We combine them and pass the sequence through an LSTM encoder.
- An LSTM decoder unrolls step-by-step to predict the future trajectory.

**2. Multi-Modal Transformer (`transformermodel.py` & `transformermodel.ipynb`)**
This is our advanced model built for handling more complex driving scenarios.
- It calculates coordinate offsets (deltas) instead of absolute positions for better training stability.
- **Social Pooling**: It embeds the target's past and the neighbors' past trajectories separately, then applies max-pooling over the neighbor dimension to form a unified social context vector.
- This is concatenated with the target features and fed into a Transformer Encoder.
- The model outputs **multiple modes** (3 possible future paths) along with a probability score for each path, which is really important since predicting the future is inherently uncertain.

## Dataset Used
We used the **nuScenes** dataset for training and evaluation. For quick iteration and testing, the code is set up for `v1.0-mini`, but it scales to the full dataset. 
Our data extraction pipeline:
- Filters for relevant categories like `human.pedestrian.adult` and `vehicle.bicycle`.
- Anchors the trajectories to the current position (last observed frame).
- Extracts data for up to 5 valid nearby neighbors to generate the social context.

## Dataset

Download the nuScenes `v1.0-mini` dataset from [nuscenes.org](https://www.nuscenes.org/download) and upload it to your Google Drive. Set the `DATAROOT` variable in the code to point to its location.

## Setup & Installation Instructions
You'll need Python 3 and a few standard ML libraries. To get everything running, install the dependencies using pip:

```bash
pip install nuscenes-devkit torch torchvision numpy matplotlib pandas scikit-learn
```

*Note: Make sure your `DATAROOT` variable in `transformermodel.py` or the paths in `Social lstm.py` point to your downloaded nuScenes dataset folder.*

## How to run the code

**Running the Social LSTM model:**
Make sure the dataset JSON files are in the same directory, then run the script directly. It will build the observation/target windows and start the training loop automatically.
```bash
python "Social lstm.py"
```

**Running the Advanced Transformer model:**
You can either run the Python script locally or use the Jupyter Notebook (`transformermodel.ipynb`).
```bash
python transformermodel.py
```
*Tip: We included the `.ipynb` file because we mostly used Google Colab for GPU training. If you're using Colab, simply upload the notebook, mount your drive, and run the cells.*

## Example Outputs / Results
During training, the models will log the Mean Squared Error loss along with the ADE (Average Displacement Error) and FDE (Final Displacement Error) metrics. 

At the end of evaluation, the Transformer model plots out some random examples from the validation set using matplotlib. These plots clearly show:
- The **Agent History** (where they came from)
- The **Ground Truth** (where they actually went)
- **Neighbors** walking around in the background (faded gray)
- **3 Mode Predictions** showing alternative paths the model thinks the agent might take, complete with probability scores!

We use these visualizations to verify that the model's highest probability mode actually aligns with the ground truth in most cases.
