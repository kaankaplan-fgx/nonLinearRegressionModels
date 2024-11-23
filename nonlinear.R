library(e1071)
library(caret)
library(rpart)
library(tidyverse)
library(AppliedPredictiveModeling)
library(dslabs)
library(rpart.plot)
library(partykit)
library(ipred)
library(FNN)
library(pls)
library(elasticnet)
library(broom)
library(glmnet)
library(MASS)
library(ISLR)
library(PerformanceAnalytics)
library(funModeling)
library(Matrix)
library(kernlab)
library(randomForest)
library(gbm)
library(nnet)
library(neuralnet)
library(GGally)
library(pls)
library(NeuralNetTools)
library(ISLR)


Hitters
df1 <- Hitters
df1 <- na.omit(df1)
rownames(df1) <- c()
set.seed(1212)
educationIndex <- createDataPartition(df1$Salary, times = 1, p=0.8, list= FALSE)
education <- df1[educationIndex,]
test <- df1[-educationIndex,]

education_x <- education %>% dplyr::select(-c("League", "NewLeague", "Division")) #Bağımsız Değişkenler
education_y <- education$Salary #Bağımlı Değişken






