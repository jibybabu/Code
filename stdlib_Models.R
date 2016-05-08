############################################################################################################
# File : stdlib_Models.R
# Contents : Standard Library for all the Modeling functions explored so far
# Goal : Can be used in any R projects for speeding up the Machine learning & Data Science Process
# Author: Jiby Babu
############################################################################################################

#Faster reading
library(data.table)
library(xgboost)
library(mRMRe)
library(caret)
library(Rtsne)
library(Matrix)

setwd("/Users/jibybabu/Desktop/Work2/BNBParibas/")

setResponseVariable = function (target) {
  responseVariable <<- target 
}

getResponseVariable = function () {
  return (responseVariable)
}

# Logistic Regression
runLogisticRegression <- function() {
  logitRegModel <- glm(train$target ~ . , data = train, family = "binomial")
  summary(logitRegModel)
  logitRegModelPrediction <- predict(logitRegModel, newdata = val, type = "response")
  return (logitRegModelPrediction)
}

# Decision Tree
runDecisionTree <- function() {
  decisionTreeFit <- rpart(train$target ~ ., data=train, method="class")
  decisionTreeFit$variable.importance
  decisionTreePrediction <- predict(decisionTreeFit, val, type="class")
  return (decisionTreePrediction)
}

# Decision Tree Diagram
getDecisionTreeDiagram <- function(decisionTreeFit) {
  prp(decisionTreeFit)
  png("123.png", res=80, height=800, width=1600) 
  fancyRpartPlot(decisionTreeFit, palettes=c("Greys", "Oranges"))
  dev.off()
}


## Random Forest
runDecisionTree <- function() {
  randomForestfit <- randomForest(train$target~., data=train, importance=TRUE, ntree=500)
  randomForestfit$importance
  randomForestfitPrediction <- predict(randomForestfit, val, type="prob")[,2]
  return (randomForestfitPrediction)
}

setXgTest = function (df) {
  xgtest <<- xgb.DMatrix(as.matrix(df))
  xgtest <<- as(xgtest, "sparseMatrix") 
}

setXgTrain = function (df) {
  xgtrain <<- xgb.DMatrix(as.matrix(df), label = getResponseVariable())
  xgtrain <<- as(xgtrain, "sparseMatrix") 
}

# Do cross-validation with xgboost - xgb.cv
docv <- function(param0, iter) {
  model_cv = xgb.cv(
    params = param0
    , nrounds = iter
    , nfold = 10
    , data = xgtrain
    , early.stop.round = 50
    , maximize = FALSE
    , nthread = 3
  )
  gc()
  best <- min(model_cv$test.logloss.mean)
  bestIter <- which(model_cv$test.logloss.mean==best)
  
  cat("\n",best, bestIter,"\n")
  print(model_cv[bestIter])
  
  bestIter-1
}

# Do cross-validation with xgboost - xgb.cv
docvAUC <- function(param0, iter) {
  model_cv = xgb.cv(
    params = param0
    , nrounds = iter
    , nfold = 10
    , data = xgtrain
    , early.stop.round = 50
    , maximize = TRUE
    , nthread = 3
  )
  gc()
  best <- max(model_cv$test.auc.mean)
  bestIter <- which(model_cv$test.auc.mean==best)
  
  cat("\n",best, bestIter,"\n")
  print(model_cv[bestIter])
  
  bestIter-1
}
doTest <- function(param0, iter) {
  watchlist <- list('train' = xgtrain)
  model = xgb.train(
    nrounds = iter
    , params = param0
    , data = xgtrain
    , watchlist = watchlist
    , print.every.n = 20
    , nthread = 4
  )
  p <- predict(model, xgtest)
  rm(model)
  gc()
  p
}

initializeH2o <- function() {
  library(h2o)
  
  ### Start a local cluster with 4GB RAM
  localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE,nthreads=2,max_mem_size = "4g")
}

setH2oTrain <- function (df) {
  htrain<-df
  htrain$TARGET<-as.factor(getResponseVariable())
  htrain <<- as.h2o(htrain)
}

setH2oTest <- function (df) {
  htest <<- as.h2o(df)
}

setH2oRandomForest <- function() {
  rf_model<-h2o.randomForest(x=1:294, y=295, training_frame = dat_h2o, nfolds=5,
                             seed = 1234 ,balance_classes=T,
                             mtries=-1,sample_rate=0.01,ntrees=850,max_depth=5,
                             keep_cross_validation_predictions=T)
  
  h2o_yhat_test <- h2o.predict(rf_model, test_h2o)
  df_yhat_test <- as.data.frame(h2o_yhat_test)
}

