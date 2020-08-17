# Sparkify-capstone-Udacity-project
Project for Udacity nanodegree program
Libraries required for the project:
Python
Pandas
Matplotlib
Seaborn
PySpark
Spark

Used libraries:
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
from pyspark.sql.functions import sum as Fsum
import pyspark.sql.functions as F
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, split, udf, count, when, isnull, collect_list
from pyspark.sql import Window
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
from pyspark.sql import functions as sF
from pyspark.sql import types as sT
import seaborn as sns
from functools import reduce

from pyspark.ml.feature import Normalizer, StandardScaler, VectorAssembler
from pyspark.ml.classification import LinearSVC, NaiveBayes,RandomForestClassifier, DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StopWordsRemover, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

Project Motivation
The aim of the project was to learn how to manipulate large and datasets with Spark MLlib as part of the Udacity “Data Scientist Nanodegree” capstone project, to understand why customers churn and identify those who will churn. Our task is to develop a model that predicts which users are at risk to churn either by downgrading from premium or by cancelling their service, the business can offer them incentives and discounts potentially saving millions in revenues.

We will use the F1 score metric to measure the efficiency of the models, which is useful for balancing precision and recall. This would be the solution approach:
1.	Initial Data Exploration: we used the spark environment for the analysis and we need to create a spark session to start the data exploration
2.	Churn definition: We analyze the page column to define the users who churn. These are the ones that have a cancelation event for a given user ID.
3.	Exploratory Data Analysis: We explore the distribution of churn and non-churn users
4.	Feature Engineering: We select the features for the analysis and modeling
5.	Machine Learning Models: Before running our models, we vectorize (VectorAssembler) and then standardize (StandardScaler) our feature set and then split (randomSplit) the data into train and test sets. For our prediction exercise, we implemented three models and then evaluated all of them on Accuracy and F1 Score.
6.	Hyper Parameter Tuning: used ParamGridbuilder to tune hyper-parameters
7.	Model Performance Evaluation and Results

Files Description
the main notebook is the sparkly project where we do all the preprocessing, feature engineering and modelling.

Results
Three different models were trained with the dataset provided:  Random Forest, Support Vector Machine and Gradient Boosted Trees. 
Their performance was compared using the F1 metric score 
The best result was obtained for the Support Vector Machine:
The F-1 Score is 0.8744588744588744
The accuracy is 0.8571428571428571

The blog post for this project can be found here: https://www.blogger.com/blog/post/edit/8632551251090685809/7586831341330096298
The code can be found at Sparkify_20200817.ipynb

References
Dataset provided by Udacity (www.udacity.com)
https://spark.apache.org/docs/2.1.1/



