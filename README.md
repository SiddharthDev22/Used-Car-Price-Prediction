 Used-Car-Price-Prediction
 
 1.1 Introduction
 
This project aims to solve the problem of predicting the price of a used car, using  supervised machine learning techniques integrated with Spark-Sklearn library. It is clearly a regression problem and predictions are carried out on dataset of used car sales in the Indian car market. Several regression techniques have been studied, including Linear Regression and Random forests of decision trees. Their performances were compared in order to determine which one works best with out dataset.

1.2 Tools
Most of the project has been developed using Python as the programming language of choice and the following libraries:

    Scikit-Learn, regression models and cross validation techniques.
    Spark-Sklearn, parallelization of the hyperparameter tuning process.
    Pandas, data analysis purposes.
    ELK Stack, data analysis too.
    Rfpimp, feature importances in random forests.

3 Regression Analysis
Formally, a regression analysis consists of a series of statistical processes aimed at estimating the relationships existing between a set of variables; in particular we try to estimate the relationship between a special variable called dependent (in our case the price) and the remaining independent variables (the other features). This analysis makes it possible to understand how the value of the dependent variable changes as the value of any of the independent variables changes, keeping the others fixed.
To carry out the prediction, various techniques have been studied including linear regression, decision trees and decision tree forests.

3.1 Data Analysis

Before preprocessing the data we must take a look to how the dataset shows up. In particular we carry out an analysis on the price attribute: describing it allows us to appreciate some informations such as min and max values and standard deviation. We then proceed to compute skewness and kurtosis of the distribution. Next, we observe its relationship with numerical and categorical features by plotting some graphs.
Feature importance computation showed us that Year and Mileage are both important for Price attribute and through correlation matrices we can learn a bit more about that.

3.2 Data Preprocessing

The dataset used to carry out the analysis is one of the best available in terms of cleanliness. Despite this, it was necessary to perform preprocessing in order to minimize the probability of incorrect learning by the models.
First it was ascertained that none of the attributes of the dataset presented null values; surprisingly, no feature presented null values, so no action was required to do so. Subsequently, the plausibility of the values for each of the numerical attributes (Price, Imm. Year, Mileage) present in the dataset was verified. We observed some cars with an extremely high mileage and we applied a filter on mileage, taking all cars with a mileage between 5000 and 250000

1.3 Used Car Price Prediction Problem
Used car price prediction problem has a certain value because different studies show that the market of used cars is destined to a continuous growth in the short term. In fact, leasing cars is now a common practice through which it is possible to get get hold of a car by paying a fixed sum for an agreed number of months rather than buying it in its entirety. Once leasing period is over, it is possible to buy the car by paying the residual value, i.e. at the expected resale price. It is therefore in the interest of vendors to be able to predict this value with a certain degree of accuracy, since if this value is initially underestimated, the installment will be higher for the customer which will most likely opt for another dealership. It is therefore clear that the price prediction of used cars has a high commercial value, especially in developed countries where the economy of leasing has a certain volume.
This problem, however, is not easy to solve as the car's value depends on many factor including year of registration, manufacturer, model, mileage, horsepower, origin and several other specific informations such as type of fuel and braking sysrem, condition of bodywork and interiors, interior materials (plastics of leather), safery index, type of change (manual, assisted, automatic, semi-automatic), number of doors, number of previous owners, if it was previously owned by a private individual or by a company and the prestige of the manufacturer.
Unfortunately, only a small part of this information is available and therefore it is very important to relate the results obtained in terms of accuracy to the features available for the analysis. Moreover, not all the previously listed features have the same importance, some are more so than others and therefore is essential to identify the most important ones, on which to perform the analysis.
Since some attributes of the dataset aren't relevant to our analysis, they have been discarded; so, as mentioned above, this fact must be taken into account when conclusions on the accuracy are drawn.

Conclusion
As we can see in the cross validation scores table, linear regressor and random forest are the ones that perform better, with a training RMSE of around 2600 and a standard deviation decisively lower than decision tree's one.
Saying that one model is objectively better than another is difficult, especially in this situation where linear regressor is working on a OHencoded dataset and random forest regressor on a label encoded one. Random forests are almost always preferable to linear regressors because they don't need much preprocessing and sometimes they produce good results even in presence of outliers. In our case the differences in performance between linear regressor and random forest are not enough to justify the exaggeratedly high number of attributes introduces by one hot encoding.
Furthermore, random forest regressor fits the data in a fraction of the time required by linear regressor

