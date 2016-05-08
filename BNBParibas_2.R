####################################################################################
# Kaggle Competition: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management
# Sponsor : BNBParibas 
# Authors: Jiby Babu
####################################################################################

# removing the variables from the environment
rm(list = ls())

# Sourcing the standard libraries
source("/Users/jibybabu/Desktop/Work2/stdlib/stdlib_dataProcessing.R")
source("/Users/jibybabu/Desktop/Work2/stdlib/stdlib_Modeling.R")
source("/Users/jibybabu/Desktop/Work2/stdlib/stdlib_utils.R")

# Setting the file path
setwd("/Users/jibybabu/Desktop/Work2/BNBParibas/")

# Start the clock!
start_time <- t1 <- Sys.time()

# Setting a seed for reproducibility
set.seed(2016)

# Read the training data set
cat("reading the train data.........\n")
train_raw <- fread("train.csv", stringsAsFactors=TRUE) 

# Read the test data set
cat("reading the test data\n")
test_raw <- fread("test.csv", stringsAsFactors=TRUE) 

# Summary of the train data set
summary(train_raw)
# Printing the dimensions of the train data set
print(dim(train_raw))
# Printing the classes of the train data set
print(sapply(train_raw, class))
# Getting the number of rows of in test data set
nrowsOfTrainSet <- nrow(train_raw)

# Summary of the test data set
summary(test_raw)
# Printing the dimensions of the test data set
print(dim(test_raw))
# Printing the classes of the test data set
print(sapply(test_raw, class))
# Getting the number of rows of in test data set
nrowsOfTestSet <- nrow(test_raw)

# Setting the response variable column name
setResponseVariable(train_raw$target)

###################
# Data Prepration #
################### 

# Convert the integer64 data tyes columns in both test and train data set to integer data type
train_raw <- convertInteger64ToInt(train_raw)
test_raw <- convertInteger64ToInt(test_raw)

# Checking if any duplicate row exists. 
# If duplicate rowss exist, remove based on further study on later stage.
cat("Checking for the duplicate rows in training data\n")
any(duplicated(train_raw))
train_raw <- train_raw[!duplicated(train_raw),]

## Variable Adjustments for Merging Training and Testing Data sets
# Nullifying the target column in training set. Because, the testing set wont be having the target column 
train_raw$target <- NULL

# Nullifying the train_raw ids
train_raw$ID <- NULL

# Nullifying the test_raw ids
test_raw$ID <- NULL

# Merging Training and Testing Data sets
cat("Combining the rows of training and testing data sets before imputation\n")
train_test_merged_data <- rbind(train_raw,test_raw)

# Convert the merged 'train_test_merged_data' data table to data frame
train_test_merged_data <- as.data.frame(train_test_merged_data) 

###########################
# Pre-Feature Engineering #
###########################

# # removing the constant features in the data set
# dim(train_test_merged_data)
# constantFeatures <- getConstantFeatures(train_test_merged_data)
# nonConstantFeatures <- setdiff(names(train_test_merged_data), constantFeatures)
# train_test_merged_data <- train_test_merged_data[, nonConstantFeatures]
# dim(train_test_merged_data)
# 
# # removing the identical(duplicate) features in the data set
# dim(train_test_merged_data)
# identicalFeatures <- getIdenticalFeatures(train_test_merged_data)
# nonIdenticalFeatures <- setdiff(names(train_test_merged_data), identicalFeatures)
# train_test_merged_data <- train_test_merged_data[, nonIdenticalFeatures]
# dim(train_test_merged_data)

# Checking the missing values
isMissingValueExists(train_test_merged_data)

# Taking a backup for the raw data before imputing
train_test_raw <- train_test_merged_data

# imputing with missing data in train_test_merged_data with -1
train_test_merged_data <- na.roughfix2(train_test_merged_data,-1)

# Taking a backup for the raw data after imputation
train_test_merged_original <- train_test_merged_data

# # Removing the zero Variance columns
# dim(train_test_merged_data)
# train_test_merged_data <- train_test_merged_data[,!nearZeroVar(train_test_merged_data, saveMetrics=TRUE)$zeroVar]
# dim(train_test_merged_data)
# 
# # removing the highly correlated variables
# dim(train_test_merged_data)
# train_test_merged_data= train_test_merged_data[,-c(getHighlyCorelatedNumericalColmns(train_test_merged_data,.99999))]
# dim(train_test_merged_data)

#######################
# Feature Engineering #
#######################

#Create T-sne 
train_test_merged_data = getTsne(train_test_merged_original,train_test_merged_data,"No")

# Convert V22 column data from AZ into integers
train_test_merged_data$v22<-sapply(train_test_merged_data$v22, az_to_int)

# get PCA
getPCA(train_test_merged_original)

# Count NA and Count NA percentage
train_test_merged_data <- createFeature_CountNA(train_test_raw,train_test_merged_data)

# Count number of 0's
train_test_merged_data <- createFeature_ZeroCount(train_test_merged_original,train_test_merged_data)

# Count the number of -ve numbers
train_test_merged_data <- createFeature_Below0Count(train_test_merged_original,train_test_merged_data)

# Find the Maximum number in the row
train_test_merged_data <- createFeature_Max(train_test_merged_original,train_test_merged_data)

# Find the Minimum number in the row
train_test_merged_data <- createFeature_Min(train_test_merged_original,train_test_merged_data)

# Find the range in the row
train_test_merged_data <- createFeature_Range(train_test_merged_original,train_test_merged_data)

# Find the Median of each row
train_test_merged_data <- createFeature_Median(train_test_merged_original,train_test_merged_data)

# Find the Standard Deviation of each row
train_test_merged_data <- createFeature_SD(train_test_merged_original,train_test_merged_data)

# Find the Relative Standard deviation of each row.
train_test_merged_data <- createFeature_RSD(train_test_merged_original,train_test_merged_data)

# Find the Skewness of each row.
train_test_merged_data <- createFeature_Skewness(train_test_merged_original,train_test_merged_data)

# Find the Kurtosis of each row.
train_test_merged_data <- createFeature_Kurtosis(train_test_merged_original,train_test_merged_data)

# Find the number of Unique values in the row.
train_test_merged_data <- createFeature_UniqueValues(train_test_merged_original,train_test_merged_data)

# Find the number of Outliers in the row.
train_test_merged_data <- createFeature_NoOfOutliers(train_test_merged_original,train_test_merged_data)

# Find the Centroid of each row
train_test_merged_data <- createFeature_Centroid(train_test_merged_original,train_test_merged_data,"Yes")

# Find the Euclidian Distance of each row
train_test_merged_data <- createFeature_EuclidianDistance(train_test_merged_original,train_test_merged_data,"Yes")

# Find the EuclidianDistancePhyRatio of each row
train_test_merged_data <- createFeature_EuclidianDistancePhyRatio(train_test_merged_original,train_test_merged_data,"No")

# Find the Cluster Number
train_test_merged_data <- createFeature_ClusterNumber(train_test_merged_original,train_test_merged_data,"Yes")

##################################################################
# Split the merged train_test data into training and testing again
##################################################################
#cat("Splitting the train_test_merged_data training and testing data sets\n")
train <- train_test_merged_data[1:nrowsOfTrainSet,]
test <- train_test_merged_data[(nrowsOfTrainSet+1):nrow(train_test_merged_data),] 

validFeatureAfter_KSTest = KSTest(train,test)
train <- train[,validFeatureAfter_KSTest]
test <- test[,validFeatureAfter_KSTest]


#################
# Modeling: Start 
#################

# Convert all the Charectors and Factors into integers
train <- convertCharFactsToInt(train)
test  <- convertCharFactsToInt(test)

print( difftime( Sys.time(), start_time, units = 'sec'))

# set up the training set in matrix format for xgboost
setXgTrain(train)
# set up the testing set in matrix format for xgboost
setXgTest(test)

param0 <- list(
  # some generic, non specific params
  "objective"  = "binary:logistic"
  , "booster"    = "gbtree"
  , "eval_metric" = "logloss"
  , "eta" = 0.01
  , "max_depth" = 12
  , "subsample" = 0.90
  , "colsample_bytree" = 0.45
  , "min_child_weight" = 1
  , "num_parallel_tree"= 1
)

# Baseline - imputing -1 - 10 fold - [617]	train-logloss:0.245780+0.001598	test-logloss:0.481877+0.005919
# Baseline - imputing -1 - 5  fold - [683]	train-logloss:0.236385+0.000901	test-logloss:0.481875+0.003960
# removing the Zero Count column - [628]	train-logloss:0.234642+0.001370	test-logloss:0.482813+0.002505
# adding the proper NA_Count_N column - [660]	train-logloss:0.238014+0.001266	test-logloss:0.481695+0.004266
# 0.45080
# 0.45024
# removing the Sum Zero and Sum one - [660]	train-logloss:0.238014+0.001266	test-logloss:0.481695+0.004266
# adding lambda = 3, [874]	train-logloss:0.236814+0.003522	test-logloss:0.480427+0.003683
# adding lambda = 5, [1014]	train-logloss:0.240392+0.001470	test-logloss:0.479925+0.002628
# adding lambda = 5 and alpha = 3, [1220]	train-logloss:0.247482+0.001635	test-logloss:0.479605+0.005444
# Complete Cross Validation - [1160]	train-logloss:0.262003+0.000850	test-logloss:0.458595+0.004962
# 0.45137
# .45070
# 0.45068
# 0.45058
# 0.45047
# Baseline - 0.45017 - 1851
# 0.45020 - 1851
# .45017
# .45003


#newtrain<-quick_cv(train)
#setXgTrain(newtrain)
#cv<-docv(param0,10000)

print( difftime( Sys.time(), start_time, units = 'min'))
#cv <- 1160
cv <- 1900
cv <- round(cv * 1.11)
cat("Calculated rounds:", cv, " Starting ensemble\n")

print( difftime( Sys.time(), start_time, units = 'sec'))

cat("Training a XGBoost classifier with cross-validation\n")

# Bagging of single xgboost for ensembling
# change to e.g. 1:10 to get quite good results

ensemble <- data.frame(output=rep(0, nrow(test)))

bagCount <- 10
for (i in 1:bagCount) {
  print(i)
  set.seed(i + 2015)
  p <- doTest(param0, cv)
  
  # use 20% to 30% more than the best iter rounds from your cross-fold number.
  # as you have another 20% or 25% training data now, which gives longer optimal training time
  #ensemble <- ensemble + p
  ensemble$output <- ensemble$output + as.numeric(p)
  ensemble<-cbind(ensemble,as.numeric(p))
}

# sample submission total analysis
submission <- read.csv("/Users/jibybabu/Desktop/Work/BNBParibas/sample_submission.csv")

cat("reading the xtress data  \n")
xtrees <- fread("extra_trees.csv", stringsAsFactors=TRUE) 
ensemble<-cbind(ensemble,xtrees$PredictedProb)

cat("reading the addNNLinearFt data  \n")
addNNLinearFt <- fread("addNNLinearFt.csv", stringsAsFactors=TRUE)

cat("reading the xgboost_owl data  \n")
xgboost_owl <- fread("xgboost_owl.csv", stringsAsFactors=TRUE)


# Finalise prediction of the ensemble
cat("Making predictions\n")
submission$PredictedProb <- (((ensemble$output/i) + xtrees$PredictedProb + addNNLinearFt$PredictedProb)/3)
write.csv(submission, "final_submission.csv", row.names=F, quote=F)

#submission$PredictedProb <- (((ensemble$output/i) + xtrees$PredictedProb + addNNLinearFt$PredictedProb + xgboost_owl$PredictedProb)/4)
#write.csv(submission, "final_submission.csv", row.names=F, quote=F)



#https://www.kaggle.com/the1owl/bnp-paribas-cardif-claims-management/can-rf-challenge-xgboost/comments
