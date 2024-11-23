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
educationIndex <- createDataPartition(df1$Salary, times = 1, p=0.8, list= FALSE) #Modelin %80inin eğitip %20 test olacak burada onu yapıyoruz


education <- df1[educationIndex,]
test <- df1[-educationIndex,]

education_x <- education %>% dplyr::select(-c("League", "NewLeague", "Division", "Salary")) 
education_y <- education$Salary #Bağımlı Değişken

test_x <- test%>% dplyr::select(-c("League", "NewLeague", "Division", "Salary")) 
test_y <- test$Salary #Bağımlı Değişken


#------------------------------- KNN (k-Nearest Neighbor) ------------------------------


knn_model <- knn.reg(train = education_x, y = education_y, k = 3, test = test_x)

#hata hesaplama
defaultSummary(data.frame(
  obs=test_y,
  pred=knn_model$pred
))


#optimum k bulma
knn_control <- trainControl(
  method = "cv", number = 10
)
knn_grid <- data.frame(
  .k = 1:10
)
set.seed(3232)
knn_model_tuning <- train(
  education_x,
  education_y,
  method = "knn",
  trControl = knn_control,
  tuneGrid = knn_grid,
  preProcess = c("center", "scale")
)
knn_model_tuning
plot(knn_model_tuning)

defaultSummary(data.frame(
  obs=test_y,
  pred=predict(knn_model_tuning, test_x)
))


#------------------------------- SVR (Support Vector Regresion) ------------------------------

svr_model <- svm(education_x, education_y)
svr_model
names(svr_model)
svr_model$kernel
svr_model$gamma
svr_model$epsilon

predict(svr_model, test_x)

defaultSummary(data.frame(
  obs=test_y,
  pred=predict(svr_model, test_x)
))


svr_control <- trainControl(
  method = "cv", number = 10
)

set.seed(3232)
svr_model_tuning <- train(
  education_x,
  education_y,
  method = "svmRadial",
  tuneLength = 20,
  trControl = svr_control,
  preProcess = c("center", "scale")
)
svr_model_tuning
plot(svr_model_tuning)
svr_model_tuning$finalModel

defaultSummary(data.frame(
  obs=test_y,
  pred=predict(svr_model_tuning, test_x)
))

#------------------------------- yapay sinir ağları ------------------------------

normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

normalized_test_x <- test_x %>% mutate_all(normalize)
normalized_test_y <- data.frame(test_y) %>% mutate_all(normalize)
normalized_test <- test %>% dplyr::select(-c("League", "NewLeague", "Division"))
normalized_test <- normalized_test %>% mutate_all(normalize)

normalized_education_x <- education_x %>% mutate_all(normalize)
normalized_education_y <- data.frame(education_y) %>% mutate_all(normalize)
normalized_education <- education %>% dplyr::select(-c("League", "NewLeague", "Division"))
normalized_education <- normalized_education %>% mutate_all(normalize)

ann_formula <- Salary ~ AtBat + Hits + HmRun + Runs + RBI + Walks + Years + CAtBat + CHits+
  CHmRun + CRuns + CRBI + CWalks + PutOuts + Assists + Errors

ann_model <- neuralnet(ann_formula, data=normalized_education)
plot(ann_model)
ann_model$result.matrix


plot(neuralnet(ann_formula, data=normalized_education, hidden = c(1,1))) #Her virgül bir katman, değerler ise o katmandaki nöron sayısı

ann_model2 <- neuralnet(ann_formula, data=normalized_education, hidden = c(1,1))
ann_model3 <- neuralnet(ann_formula, data=normalized_education, hidden = 10)
plot(ann_model3)


defaultSummary(data.frame(
  obs=test_y,
  pred=predict(ann_model, test_x)
))

ann_control <- trainControl(
  method = "cv", number = 10
)
ann_grid <- expand.grid(
  size = 1:5,
  decay = c(0.0001, 0.001, 0.01, 0.1)
)
set.seed(3232)

ann_model_tuning <- train(
  education_x,
  education_y,
  method = "mlpWeightDecay",
  tuneGrid = ann_grid,
  trControl = ann_control,
  preProcess = c("center", "scale")
)


plot(ann_model_tuning)


defaultSummary(data.frame(
  obs=test_y,
  pred=predict(ann_model_tuning, test_x)
))


#------------------------------- CART (Classification and Regression Tree)------------------------------

cart_model <- rpart(Salary ~ ., data = df1)
cart_model
cart_model$variable.importance
plot(cart_model)
text(cart_model)


prp(cart_model)
rpart.plot(cart_model)
plotcp(cart_model) #değişkenlerin önem değeri = cp

df1 %>% mutate(y=predict(cart_model)) %>% ggplot() + geom_point(aes(CHits, Salary)) + geom_step(aes(CHits, y), col="blue")

cart_model2 <- rpart(Salary ~ CHits, data = df1, control = rpart.control(cp=0, minsplit = 2))

df1 %>% mutate(y=predict(cart_model2)) %>% ggplot() + geom_point(aes(CHits, Salary)) + geom_step(aes(CHits, y), col="blue")


cart_model_edited <- prune(cart_model2, cp=0.012)
df1 %>% mutate(y=predict(cart_model_edited)) %>% ggplot() + geom_point(aes(CHits, Salary)) + geom_step(aes(CHits, y), col="blue")
rpart.plot(cart_model_edited)

cart_model3 <- rpart(Salary ~ ., data = education, control = rpart.control(cp = 0.2, minsplit = 2))
test_x_x <- test %>% dplyr::select(-Salary)
cart_model3 <- rpart(Salary ~ ., data = education)
defaultSummary(data.frame(
  obs=test_y,
  pred=predict(cart_model3, test_x_x)
))

#en uygun cp bulma

cart_control <- trainControl(
  method = "cv", number = 10
)
cart_grid <- data.frame(
  cp=seq(0,0.01, length = 25)
)
set.seed(3232)

cart_model_tuning <- train(
  education_x,
  education_y,
  method = "rpart",
  tuneGrid = cart_grid,
  trControl = cart_control,
  preProcess = c("center", "scale")
)

cart_model_tuning
plot(cart_model_tuning)




