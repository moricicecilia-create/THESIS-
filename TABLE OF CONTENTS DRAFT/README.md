Robustness and Probability Calibration of Pretrained Tabular Foundation Models under Missing-Data Shift
Table of contents 
Introduction 
1.  Research Background and Motivation
2.  Problem Statement
3.  Research Objectives and Questions
Chapter 1. Tabular Data and Foundation Models 
1.1  Machine Learning for Tabular Data
1.2  Binary Classification and Probabilistic Predictions
1.3  Traditional Machine Learning Approaches for Tabular Data
1.4  Deep Learning for Tabular Data
1.5  Foundation Models and Pretraining
1.6  In-Context Learning
1.7  Tabular Foundation Models
Chapter 2. Predictive Robustness and Probability Calibration 
2.1  Trustworthy Artificial Intelligence
2.2  Distribution Shift in Machine Learning
2.3  Missing Data and Missingness Mechanisms
2.4  Missing-Data Shift
2.5  Predictive Robustness under Incomplete Information
2.6  Probability Calibration and Predictive Discrimination 
2.8  Post-hoc Calibration Methods
2.8.1  Temperature Scaling
2.8.2  Calibration under Distribution Shift
Chapter 3. Experimental Framework 
3.1  Research Questions and Experimental Objectives
3.2  Models Included in the Study
3.2.1  TabPFN-v2
3.2.2  TabICLv2
3.2.3  TabDPT
3.2.4  CatBoost
3.2.5  Multilayer Perceptron (MLP)
3.3  Dataset Selection and Characteristics
3.4  Data Partitioning Strategy
3.5  Preprocessing and Missing-Value Handling
3.6  Missing-Data Scenarios
3.6.1  Reference Condition
3.6.2  Random Cell-wise Missingness
3.6.3  Complete Feature Unavailability
3.7  Calibration Strategies
3.7.1  No Additional Calibration
3.7.2  Reference Calibration
3.7.3  Missingness-Augmented Calibration
3.8  Evaluation Metrics
3.8.1  AUROC and Balanced Accuracy
3.8.2  Log-Loss
3.8.3  Brier Score
3.8.4  Expected Calibration Error
3.8.5  Reliability Diagrams
3.9  Computational Cost Evaluation
3.10  Experimental Repetitions and Reproducibility
Chapter 4. Experimental Results 
4.1  Overview of Experimental Results
4.2  Model Performance under Reference Conditions
4.3  Robustness under Random Cell-wise Missingness
4.4  Robustness under Complete Feature Unavailability
4.5  Comparison of Missingness Patterns and Intensities
4.6  Probability Calibration under Reference Conditions
4.7  Effects of Temperature Scaling
4.8  Transferability of Calibration under Missing-Data Shift
4.8.1  Transfer across Missingness Intensities
4.8.2  Transfer across Missingness Structures
4.9  Reliability Diagram Analysis
4.10  Computational Performance
4.10.1  Training and Inference Time
4.10.2  Memory Consumption
4.12  Overall Comparison of Models 
Chapter 5. Discussion 
5.1  Summary of Main Findings
5.2  Comparative Robustness of Tabular Foundation Models
5.3  Effectiveness of Post-hoc Calibration
5.4  Generalization of Calibration across Missingness Conditions
5.5  Performance-Reliability-Computational Cost Trade-offs
5.7  Limitations of the Study 
Chapter 6. Conclusions
6.1  Research Summary
6.2  Answers to the Research Questions
6.3  Main Scientific Contributions
References
Appendices
A.  Detailed Dataset Descriptions
B.  Model Configurations and Hyperparameters

