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
library(xgboost)

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


#------------------------------- BTR (Bayesian Topic Regression)------------------------------


#bagging

bagging_model <- ipredbagg(education_y, education_x)
bagging_model


education_c <- education %>% dplyr::select(-c("League", "NewLeague", "Division"))
bagging_model2 <- bagging(Salary~., data= education_c)
bagging_model2


bagging_model_3 <- randomForest(
  Salary~.,
  data = education_c,
  mtry = ncol(education_x), #degisken rastegeleliğini bagging için kaldırıyoruz
  importance = TRUE, 
  ntrees=500
)
bagging_model_3

predict(bagging_model, test_x)
predict(bagging_model2, test_x)
predict(bagging_model_3, test_x)


defaultSummary(data.frame(
  obs=test_y,
  pred=predict(bagging_model_3, test_x)
))

plot(bagging_model_3)

bagging_control <- trainControl(
  method = "cv", number = 10
)
bagging_grid <- expand.grid(
  mtry=ncol(education_x)
)
set.seed(3232)

bagging_model_tuning <- train(
  education_x,
  education_y,
  method = "rf", #random forest
  tuneGrid = bagging_grid,
  trControl = bagging_control
)
bagging_model_tuning

defaultSummary(data.frame(
  obs=test_y,
  pred=predict(bagging_model_3, test_x)
))

#Random Forest

rf_model <- randomForest(education_x, education_y, importance = TRUE)
importance(rf_model)
varImpPlot(rf_model)
predict(rf_model, test_x)

defaultSummary(data.frame(
  obs=test_y,
  pred=predict(rf_model, test_x)
))


rf_control <- trainControl(
  method = "cv", number = 10
)

rf_grid <- expand.grid(
  mtry=c(1:16)
)
set.seed(3232)

rf_model_tuning <- train(
  education_x,
  education_y,
  method = "rf", #random forest
  tuneGrid = rf_grid,
  trControl = rf_control
)


rf_model_tuning
defaultSummary(data.frame(
  obs=test_y,
  pred=predict(rf_model_tuning, test_x)
))

#BOOSTİNG
#GBM

gbm_model <- gbm(Salary~. , education_c,
                distribution = "gaussian",
                n.trees = 3700,
                interaction.depth = 1,
                shrinkage = 0.05,
                cv.folds = 10
                )

summary(gbm_model)

defaultSummary(data.frame(
  obs=education_y,
  pred=gbm_model$fit
))

gbm.perf(gbm_model, method = "cv")

defaultSummary(data.frame(
  obs=test_y,
  pred=predict(gbm_model, test_x)
))

gbm_control <- trainControl(
  method = "cv", number = 10, search = "grid"
)

gbm_grid <- expand.grid(
  n.trees = seq(100, 3000, by=100),
  interaction.depth = seq(1,10, by = 2),
  shrinkage = seq(0.01, 0.05, by = 0.01),
  n.minobsinnode = c(10:15)
)
set.seed(3232)

gbm_model_tuning <- train(
  education_x,
  education_y,
  method = "gbm", 
  tuneGrid = gbm_grid,
  trControl = gbm_control
)
plot(gbm_model_tuning)

defaultSummary(data.frame(
  obs=test_y,
  pred=predict(gbm_model_tuning, test_x)
))


#XGBoost

xgb_model <- xgboost(data = as.matrix(education_x),
                     label = education_y,
                     booster = "gblinear",
                     max.depth =1,
                     eta = 0.5,
                     nrounds = 500,
                     nthread = 5
                     )

xgb_education <- xgb.DMatrix(data = as.matrix(education_x), label = education_y)
xgb_test <- xgb.DMatrix(data = as.matrix(test_x), label = test_y)

xgb_model2 <- xgboost(data = xgb_education,
                     booster = "gblinear",
                     max.depth =1,
                     eta = 0.5,
                     nrounds = 500,
                     nthread = 5
)

xgb.plot.importance(xgb.importance(model = xgb_model))

watchlist  = list(train = xgb_education, test = xgb_test)
xgb_model3 <- xgb.train(data = xgb_education,
                      booster = "gblinear",
                      max.depth =1,
                      eta = 0.5,
                      nrounds = 500,
                      nthread = 5,
                      watchlist = watchlist
)

predict(xgb_model3, as.matrix(test_x))
plot(predict(xgb_model3, as.matrix(test_x)), test_y, xlab = "Tahmin", ylab = "Gerçek", main = "Tahmin - Gerçek", col = "red", pch = 20)
grid()
abline(0,1,col = "blue")

xgb_control <- trainControl(
  method = "cv", number = 10, search = "grid"
)

xgb_grid <- expand.grid(
  nrounds = seq(1000, 10000,by= 500),
  eta = seq(0,1, by = 0.1),
  lambda = seq(1,5, by=1),
  alpha = seq(0,1, by = 0.1)
)
set.seed(3232)

xgb_model_tuning <- train(
  x = data.matrix(education_x),
  y = education_y,
  method = "xgbLinear", 
  tuneGrid = xgb_grid,
  trControl = xgb_control
) ## çok uzun sürüyor


#hyperparameter % tree

xgb_grid_2 <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
) #default values from xgb

xgb_control2 <- trainControl(
  method = "none"
)

xgb_model4 <- train(
  x = as.matrix(education_x),
  y = education_y,
  tuneGrid = xgb_grid_2,
  trControl = xgb_control2,
  method = "xgbTree"
)

defaultSummary(data.frame(
  obs=test_y,
  pred=predict(xgb_model4, as.matrix(test_x))
))






