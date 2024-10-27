Employee Attrition & Department Prediction Model

Overview

This project is focused on predicting employee attrition and department assignment using a neural network model. The dataset contains various employee features, and the goal is to build a multi-output neural network capable of predicting:

	•	Whether an employee is likely to leave the company (attrition prediction).
	•	The department to which the employee belongs (department classification).

The model leverages three output branches with individual accuracy metrics to measure performance for both tasks.

Project Objectives

	1.	Attrition Prediction: Identify if an employee is likely to leave based on various features.
	2.	Department Classification: Determine the correct department for an employee using the given dataset.
	3.	Model Optimization: Achieve high prediction accuracy while minimizing loss for each output task.

Technologies & Libraries

	•	Python
	•	TensorFlow/Keras: For building and training the neural network model.
	•	NumPy: For numerical operations and data handling.
	•	Pandas: For data preprocessing and manipulation.
	•	Scikit-learn: For data analysis, evaluation metrics, and potential model enhancements.

Model Architecture

The model is structured as a multi-output neural network with:

	•	A shared input layer feeding common features into multiple layers.
	•	Separate branches for:
	•	Attrition Prediction using a sigmoid activation for binary classification.
	•	Department Classification using a sigmoid activation for binary classification.
	•	A final combined output layer for aggregated prediction.

Data Preprocessing

	•	Scaling: Numerical features were standardized using StandardScaler.
	•	Encoding: Categorical features like department information were encoded using OneHotEncoder.
	•	Reshaping: Targets were reshaped to match the model’s expected format.
	•	Handling Imbalance: If class imbalance was detected, potential solutions included class weights and adjusting thresholds.

Key Results

	•	The model achieved promising results:
	•	Attrition Accuracy: ~85.37%
	•	Department Accuracy: ~86.73%
	•	Final Combined Accuracy: ~86.05%
	•	The loss for the combined outputs was 1.2119, indicating room for optimization.

Future Improvements

	•	Hyperparameter Tuning: Optimize neural network parameters using Grid Search or Random Search.
	•	Class Imbalance Solutions: Explore class weights or synthetic over-sampling to improve classification accuracy.
	•	Feature Engineering: Add or transform features to increase model predictive power.
	•	Ensemble Learning: Combine predictions with other models like Random Forest or XGBoost for a more robust prediction.

Usage

	1.	Clone the Repository:

git clone <repository-url>


	2.	Install Dependencies:

pip install -r requirements.txt


	3.	Run the Training Script:

python train_model.py


	4.	Evaluate the Model:
Use the evaluation script to check the performance metrics for both attrition and department prediction.

