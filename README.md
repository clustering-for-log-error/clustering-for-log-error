# What is driving the errors in the Zestimates?

![Parcel Density Plot of our area of interest](/parcel_density.png)

### Improve our original estimate of the log error by using Clustering ML Algorithm

### *The Notebook*

> The main notebook in this repo will serve as the primary findings of our research. 

## *Background*:

> We've used Linear Regression models to start to tackle the home value estimate. We will now work on the 2017 Zillow Kaggle issue of improving Zillow's log error.

> As the competiton has past, we will be using the main dataset provided by Zillow. The audience for this project is a data science team, and we want to demo the discoveries we made and work we've have done related to uncovering what the drivers of the error in the Zestimate is.

> We also want to show our team how we work through a problem like this from a data science perspective.

> Like always for this project we need to have our code documentation and commenting buttoned-up.

# Specification

## *Goals*

We will be using the ML Clustering Algorithm KMeans to find clusters within the data to improve our estimate of the log error. Once those clusters have been identified, we will use Regression models to test the efficacy of the clustering work.

## *Deliverables*

**What should our team expect to receive from us?**

1. A report (in the form of a detailed, high level notebook)

The notebook should cover all the steps from acquistion through modeling. There is no need to dive too deep into any of the particular part, only the highlights that we feel will be important to the team. As well as a breakdown of our underlying results.

2. A detailed README explaing the project and step to reproduce the results.

***model must be reproducible by someone with their own env.py file***

3. Finally, all the modules where the functions are necessary, indicated or ideal, it runs the functions in the notebook, but the functions are stored in their respective modules (prepare.py, e.g.). You will have an acquire.py, prepare.py (or wrangle to cover both acquire and prepare), and model.py (fit, predict, evaluate the final model) - and any other necessary modules.

# Data Dictionary

| Feature        | Count                  | Description                                                                                       |
|----------------|------------------------|---------------------------------------------------------------------------------------------------|
| latitude       | 59930 non-null float64 | Geographic coordinate that specifies the north–south position of each home on the Earth's surface |
| longitude      | 59930 non-null float64 | Geographic coordinate that specifies the east–west position of each home on the Earth's surface   |
| age            | 59930 non-null int64   | Derived column which indicates the age of the home at the time it was sold                        |
| square_footage | 59930 non-null float64 | Total square foot of home                                                                         |
| lot_size       | 59930 non-null float64 | Total square foot of entire lot                                                                   |
| full_value     | 59930 non-null float64 | Total actual price of home                                                                        |
| Los_Angeles    | 59930 non-null int64   | Indicates if the home is in Los Angeles county                                                    |
| Orange         | 59930 non-null int64   | Indicates if the home is in Orange county                                                         |
| Ventura        | 59930 non-null int64   | Indicates if the home is in Ventura county                                                        |
| logerror       | 59930 non-null float64 | Log of the error between actual home price and estimated home price                               |
| lot_cluster    | 59930 non-null int64   | Derived column which indicates if each home is in the cluster containing lot_size                 |
| loc_cluster    | 59930 non-null int64   | Derived column which indicates if each home is in the cluster containing latitude and longitude   |

## *The Pipeline*

### PROJECT PLANNING & README

> Brainstorming ideas, hypotheses, related to how variables might impact or relate to each other, both within independent variables and between the independent variables and dependent variable, and also related to any ideas for new features you may have while first looking at the existing variables and challenge ahead of you.

> In addition: we will summarize our project and goals. We will task out how we will work through the pipeline, in as much detail as we need to keep on track.

### ACQUIRE:

**Goal**: leave this section with a dataframe ready to prepare.

The ad hoc part includes summarizing your data as you read it in and begin to explore, look at the first few rows, data types, summary stats, column names, shape of the data frame, etc.

acquire.py: The reproducible part is the gathering data from SQL.

### PREP:

**Goal**: leave this section with a dataset that is ready to be analyzed. Data types are appropriate, missing values have been addressed, as have any data integrity issues.

The ad hoc part includes plotting the distributions of individual variables and using those plots to identify outliers and if those should be handled (and if so, how), identify unit scales to identify how to best scale the numeric data, as well as finding erroneous or invalid data that may exist in your dataframe.

Some items to consider:

- [X] split data to train/test<br>
- [X] Handle Missing Values
- [X] Handle erroneous data and/or outliers you wish to address
- [X] encode variables as needed
- [X] scale data as needed
- [X] cluster the target variable
- [X] cluster independent variables
- [X] test the significance of and visualize clusters

prep.py: The reproducible part is the handling of missing values, fixing data integrity issues, changing data types, etc.

### DATA EXPLORATION & FEATURE SELECTION

**Goal**: Address each of the questions posed in our planning and brainstorming phase - as time permits. As well as any uncovered that come up during the visual or statistical analysis.

When you have completed this step, we will have the findings from our analysis that will be used in the final notebook, and information to move forward toward building a model.

### MODELING & EVALUATION

**Goal**: use regression models against the results found while clustering to identify a model that performs better than a baseline.

1. Train (fit, transform, evaluate) multiple different models, varying the model type and your meta-parameters.

2. Compare evaluation metrics across all the models, and select the best performing model.

3. Test the final model (transform, evaluate) on your out-of-sample data (the testing data set). Summarize the performance. Interpret your results.

model.py: will have the functions to fit, predict and evaluate the model

# SUMMARY

## *SQL Data Acquisition*

Must use your own env file to access data.

***

## *Technical Skills used*

* Python
* SQL
* Sequel Pro
* Jupyter Notebook
* VS Code
* Various data science libraries (Pandas, Numpy, Matplotlib, Seaborn, Sklearn, etc.)
* Stats (Hypothesis testing, correlation tests)
* Clustering Model (KMeans)
* Regression Models (Linear Regression, Decision Tree Regressor, Random Forest Regressor)

***

## *Executive Summary*

1. Hypothesis and conclusion is unclear. Our derived variables proved useful, but not significantly. 

2. Our main drivers appeared to hover around the overarching geological data and clustering using the selected features associated with those data points. 

3. The linear regression model performed quite poorly. However, the decision tree and random forest regressors did slightly better than baseline.

4. We observed some statistical difference between log error with regards to these features:
    - Longitude/Latitude
    - Lot size
    - Square footage
    - Age of the home
    
It appears either more time is necessary to evaluate the different clustering opportunities within the data. Or that, perhaps, clustering is not the best approach for this data.