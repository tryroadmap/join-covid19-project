### Project Description
In the context of the global COVID-19 pandemic, Kaggle has launched several challenges in order to provide useful insights that may answer some of the open scientific questions about the virus. This is the case of the COVID19 Global Forecasting, in which participants are encouraged to fit worldwide data in order to predict the pandemic evolution, hopefully helping to determine which factors impact the transmission behavior of COVID-19.
![image](https://github.com/lotusxai/covid19-project/blob/master/download.png)




### Access Practice Notebook
```
git clone https://github.com/lotusxai/covid19-project.git
cd covid19-project/notebooks
jupyter notebook covid19-task-notebook.ipynb

#The global tendency graph excluding China and in China:
#Example
./run.sh tendency

#The prediction graph of a certain country by using linear regression:
#Example
./run.sh linear Spain


#The prediction graph of a certain country by using logistic regression:
#Example
./run.sh logistic Spain
```

### Acknowledgments
The source data of this project comes from the Kaggle website https://www.kaggle.com/c/covid19-global-forecasting-week-1.
This evaluation data for this competition comes from John Hopkins CSSE, which is uninvolved in the competition.


The initial notebook has been received and analyzed during a collaboration of PhillyTalent and [Patrick Sánchez](https://www.kaggle.com/saga21)

#### Tasks
Feature engineering with SageMaker Notebooks


#### Technology Stack
AWS SageMaker, etc etc



#### Data Science Manager
Hasan
