############################################################################################################
# File : stdlib_Utils.R
# Contents : Standard Library for all the utility functions explored so far
# Goal : Can be used in any R projects for speeding up the Machine learning & Data Science Process
# Author: Jiby Babu
############################################################################################################

getHighlyCorelatedNumericalColmns = function (df, maxPossibleCor=0.9,verb=F) {
  nums <- sapply(df, is.numeric)
  numerical_columns <- df[,nums]
  corelations<-cor(numerical_columns)
  hc<-findCorrelation(corelations,cutoff=maxPossibleCor,verbose=verb)
  return (hc)
  #sample usage
  # Removing Highly corelated Numerical Columns
  #hcColms = getHighlyCorelatedNumericalColmns(train_test_merged_data,.90)
  #if (!is.na(hcColms)) {
  #  cat("There are ", length(hcColms), "highly correlated columns", "and the columns are...\n", hcColms)
  #  train_test_merged_data <- train_test_merged_data[,-c(hcColms)]
  #}
}

na.roughfix2 <- function (object, value) {
  res <- lapply(object, roughfix,value)
  structure(res, class = "data.frame", row.names = seq_len(nrow(object)))
}

roughfix <- function(x,value) {
  missing <- is.na(x)
  if (!any(missing)) return(x)
  
  if (is.numeric(x)) {
    # If the NA's follows a pattern ie not random, its better not to impute
    if (value == "median") {
      x[missing] <- median.default(x[!missing])
    } else {
      x[missing] <- value
    }
  } else if (is.factor(x)) {
    cat("hello")
    freq <- table(x)
    x[missing] <- names(freq)[which.max(freq)]
  } else {
    stop("na.roughfix only works for numeric or factor")
  }
  x
}

getBestFeatures = function (df, noOfFeaturesReq) {
  ind <- sapply(df, is.integer)
  df[ind] <- lapply(df[ind], as.numeric)
  dd <- mRMR.data(data = df)
  
  feats <- mRMR.classic(data = dd, target_indices = c(ncol(df)), feature_count = noOfFeaturesReq)
  bestVars <-data.frame('features'=names(df)[solutions(feats)[[1]]], 'scores'= scores(feats)[[1]])
  print(bestVars)
  return (bestVars[,1])
}

##Hexavigesimal decoding
az_to_int <- function(az) {
  xx <- strsplit(tolower(az), "")[[1]]
  pos <- match(xx, letters[(1:26)]) 
  result <- sum( pos* 26^rev(seq_along(xx)-1))
  return(result)
}

KSTest = function(train,test,cutOff=0.007) {
  #Feature selection using KS test with 0.007-.004 as cutoff.
  ksMat = NULL
  for (j in 1:ncol(test)) {
    cat(j," ")
    ksMat = rbind(ksMat, cbind(j, ks.test(train[,j],test[,j])$statistic))
  }
  
  ksMat2 = ksMat[ksMat[,2]<cutOff,]
  return (feats <- as.numeric(ksMat2[,1])) 
}

# T distribution Stochastic neighbourhood embedding
#https://www.kaggle.com/puyokw/digit-recognizer/clustering-in-2-dimension-using-tsne/comments
#https://www.kaggle.com/cyberzhg/digit-recognizer/sklearn-pca-svm/run/12595
getTsne <- function (df,targetDf,generate,dimension=2) {
  
  if (generate == "No") {
    Y <- fread("tsne.csv") 
    targetDf$tsne1 = Y[[1]]
    targetDf$tsne2 = Y[[2]]
    return (targetDf)
  }
  
  set.seed(1234)
  nums <- sapply(df, is.numeric)
  features <- names(df[,nums])
  tsne <- Rtsne(as.matrix(df[,features]), dims = 2, perplexity=30, verbose=TRUE,check_duplicates = FALSE)
  write.csv(tsne$Y, "tsne.csv", row.names=F, quote=F)
  Y <- tsne$Y
  targetDf$tsne1 = Y[[1]]
  targetDf$tsne2 = Y[[2]]
  return (targetDf)
}

getPCA<-function(df,varianceToCatch=.99){
  nums<-sapply(df,is.numeric())
  df_num<-df[,nums] 
  
  trans = preProcess(df_num, method=c("BoxCox", "center", 
                                      "scale", "pca"),thresh=varianceToCatch)
  
  PC = predict(trans, df_num)
  return (PC)
  
}

CreateTrainTestSet <- function (df,noOfFolds,foldToUseForTesting) {
  indexr<-createFolds(MergedDeals$target,k=5, list = FALSE)
  train<-rbind(MergedDeals[indexr==1,],MergedDeals[indexr==2,],MergedDeals[indexr==3,],MergedDeals[indexr==4,])
  val<-MergedDeals[indexr==5,]
}


###############################################################################################################
#  OverSampling
# Since the data is highly imbalanced, 95% 0's and 5% 1's , We need to do overSampling
# Technique Used : Synthetic Minority Over-sampling 
# References:
# http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
# https://chandramanitiwary.wordpress.com/2013/09/10/class-imbalance-smote-and-r/
###############################################################################################################

overSampleUsingSMOTE <- function (df,targetColumnName) {
  # Moving the target column to last column to run SMOTE
  target <- df[[targetColumnName]]
  df<- df[-which( colnames(df)==targetColumnName)]
  df <- cbind(df,target)
  
  # Running the SMOTE package
  summary(df[[targetColumnName]])
  train<-SMOTE(targetColumnName ~ . , k=10, train,perc.over = 1500,perc.under=140)
  summary(df$target)
  detach(df)
}


# Compute Confusion Matrix
# Creating the Confusion Matrix function to                    
# calculate the total Accuray, Sensitivity(True Positive Rate),
# Specificity(True Negative Rate) and MissClassification Rate  
# Input to this function will be the confusion matrix.

processConfusionMatrix <- function (confusionMatrix1) {
  # Computing the Accuracy of the Prediction
  Accuracy = (confusionMatrix1[1,1]+confusionMatrix1[2,2])/(confusionMatrix1[1,1]+confusionMatrix1[1,2]+confusionMatrix1[2,1]+confusionMatrix1[2,2])
  cat(" Total Accuracy = ", Accuracy,"\n")
  
  # Computing the Recall/Sensitivity/TruePositive of the Prediction
  # TPR = (TP)/(TP+FN)
  Sensitivity = (confusionMatrix1[2,2])/(confusionMatrix1[2,2]+confusionMatrix1[1,2])
  cat("Sensitivity/TPR = ", Sensitivity,"\n")
  
  # Computing the Specificity
  # TNR = (TN) / (TN + FP) 
  Specificity = (confusionMatrix1[1,1])/(confusionMatrix1[1,1]+confusionMatrix1[2,1])
  cat("Specificity/TNR = ", Specificity,"\n")
  
  # Computing the Miss ClassificationRatio
  # Missclassification = 1- [(TP)/(TP+FN)] = FN/(FN+TP)
  MissClassificationRatio = (confusionMatrix1[1,2])/(confusionMatrix1[1,2]+confusionMatrix1[2,2])
  cat("Miss Classification Ratio = ", MissClassificationRatio,"\n")
}

fineTuneModel <- function (predictionValues,validationSet,probability) {
  p <- predictionValues
  validationSet$p <- p
  # Getting the Logistic Regression Metrics
  #predictionValues[predictionValues$target==0,"p"]
  summary(factor(val$target))
  hist(validationSet[validationSet$target==1,"p"])
  hist(validationSet[validationSet$target==0,"p"])
  prediction = ifelse(p<probability, 0, 1)
  return(prediction)
}

################################
##    Feature Engineering     ##
################################
# createFeature_CountNA
# createFeature_ZeroCount
# createFeature_Below0Count
# createFeature_Max
# createFeature_Min
# createFeature_Median
# createFeature_SD
# createFeature_SumZero
# createFeature_SumOne
# createFeature_Centroid
# createFeature_EuclidianDistance_And_EuclidianDistancePhyRatio
# createFeature_ClusterNumber

createFeature_CountNA <- function (df,targetDf) { 
  # CountNA 
  targetDf$NACount <- rowSums(is.na(df)) 
  #CountNA percentage
  targetDf$NACount_Percent <- rowSums(is.na(df)) / ncol(df)  
  
  return (targetDf)
}

createFeature_ZeroCount <- function (df,targetDf) {
  # ZeroCount Feature
  df <- as.data.frame(df)
  targetDf$ZeroCount <- rowSums(df[,names(df)]== 0) / ncol(df)
  return (targetDf)
}

createFeature_Below0Count <- function (df,targetDf) {
  # Below0Count
  df <- as.data.frame(df)
  targetDf$Below0Count <- rowSums(df[,names(df)] < 0) / ncol(df)
  
  return (targetDf)
}

createFeature_Max <- function (df,targetDf) {
  # Max
  targetDf$max <- do.call(pmax, df)
  
  return (targetDf)
}

createFeature_Min <- function (df,targetDf) {
  #Min
  targetDf$min <- do.call(pmin, df)
  
  return (targetDf)
}

createFeature_Range <- function (df,targetDf) {
  #Range
  targetDf$Range <- do.call(pmax, df) - do.call(pmin, df)
  
  return (targetDf)
}

createFeature_Median <- function (df,targetDf) {
  #Median
  targetDf$median<-apply(FUN=median,X=df,MARGIN=1)
  
  return (targetDf)
}

createFeature_SD <- function (df,targetDf) {
  #sd 
  targetDf$sd<-apply(FUN=sd,X=df,MARGIN=1)
  
  return (targetDf)
}

createFeature_UniqueValues <- function (df,targetDf) {
  #Unique Values
  targetDf$UniqueValues = apply(df, 1, function(x){length(unique(x))})
  
  return (targetDf)
}

createFeature_NoOfOutliers <- function (df,targetDf) {
  #Number of Outliers
  targetDf$Outliers <- apply(df, 1, function(x){length(boxplot(x, plot = FALSE)$out)})
  
  return (targetDf)
}

createFeature_RSD <- function (df,targetDf) {
  # Coefficent of Variation or Relative Standard Deviation
  # https://en.wikipedia.org/wiki/Coefficient_of_variation
  targetDf$RSD <- apply(df, 1, function(x){if(mean(x) == 0){0} else{sd(x) / mean(x)}})
  
  return (targetDf)
}

createFeature_Kurtosis <- function (df,targetDf) {
  # Kurtosis
  library(moments)
  targetDf$Kurtosis <- apply(df, 1, kurtosis)
  
  return (targetDf)
}

createFeature_Skewness <- function (df,targetDf) {
  # Skewness
  library(moments)
  targetDf$Kurtosis <- apply(df, 1, skewness)
  
  return (targetDf)
}

createFeature_SumZero <- function (df,targetDf) {
  #Sum Zero
  targetDf$sumzero<-apply(FUN=function(x) sum(ifelse(x==0,1,0)),X=df,MARGIN=1)
  
  return (targetDf)
}

createFeature_SumOne <- function (df,targetDf) {
  #Sum one
  targetDf$sumone<-apply(FUN=function(x) sum(ifelse(x==1,1,0)),X=df,MARGIN=1)
  
  return (targetDf)
}

createFeature_Centroid <- function (df,targetDf,generate) {
  if (generate == "No") {
    centroid <- fread("centroid.csv")
    centroid  <- as.data.frame(centroid)
    targetDf <- cbind(targetDf,centroid)
    return (targetDf)
  }
  
  # centroid = apply(df, 1, function(x){log(as.numeric(kmeans(as.numeric(df[x,]),centers=1)$centers) + 1)})
  # compute kmean centroids
  centroid=c()
  for( i in 1:nrow(df)) {
    kmeansObj=kmeans(as.numeric(df[i,]),centers=1)
    centroid=c(centroid,as.numeric(kmeansObj$centers))
  }
  targetDf$centroid=log(centroid+1)
  
  # Filling the the centroids missed to calculate with median
  if (any(is.na(targetDf$centroid))) {
    missing <- is.na(targetDf$centroid)
    targetDf$centroid[missing] <- median.default(targetDf$centroid[!missing])
  }
  
  # Writing to a file
  write.csv(targetDf$centroid, "centroid.csv", row.names=F, quote=F)
  
  return (targetDf)
}

createFeature_EuclidianDistance <- function (df,targetDf,generate) {
  
  if (generate == "No") {
    eu <- fread("euclideanDistance.csv") 
    eu  <- as.data.frame(eu)
    targetDf <- cbind(targetDf,eu)
    return (targetDf)
  }
  
  
  #compute euclidean distance
  eu_dist=c()
  for( i in 1:nrow(df))
  {
    origin=rep(0,ncol(df))
    point=as.numeric(df[i,])
    d=as.numeric(dist(rbind(origin,point)))
    eu_dist=c(eu_dist,d)
  }
  targetDf$eu_dist = eu_dist
  
  # Writing to a file
  write.csv(targetDf$eu_dist, "euclideanDistance.csv", row.names=F, quote=F)
  
  return (targetDf)
}

createFeature_EuclidianDistancePhyRatio <- function (df,targetDf,generate) {
  
  if (generate == "No") {
    euclidianDistance  <- fread("euclideanDistance.csv")
    # Value of phy is 1.618
    eu_distGoldenRatio <- ((euclidianDistance-1.618)/1.618)
    eu_distGoldenRatio <- as.data.frame(eu_distGoldenRatio)
    targetDf <- cbind(targetDf,eu_distGoldenRatio)
    return (targetDf)
  }
  
  #compute euclidean distance
  eu_dist=c()
  for( i in 1:nrow(df))
  {
    origin=rep(0,ncol(df))
    point=as.numeric(df[1,])
    d=as.numeric(dist(rbind(origin,point)))
    eu_dist=c(eu_dist,d)
  }
  
  # Value of phy is 1.618
  targetDf$eu_distGoldenRatio=((eu_dist-1.618)/1.618)
  
  return (targetDf)
}

createFeature_ClusterNumber <- function (df,targetDf,generate) {
  
  if (generate == "No") {
    ClusterNo <- fread("ClusterNo.csv") 
    ClusterNo <- as.data.frame(ClusterNo)
    targetDf  <- cbind(df,ClusterNo)
    return (targetDf)
  }
  
  # Computing the Cluster
  set.seed(1234)
  kmeansObj=kmeans(as.matrix(df),centers=3)
  kmeansObj=kmeans(as.matrix(df),centers=3)
  kmeansObj=kmeans(as.matrix(df),centers=3)
  targetDf$ClusterNo = kmeansObj$cluster
  
  # Writing to a file
  write.csv(targetDf$ClusterNo, "ClusterNo.csv", row.names=F, quote=F)
  
  return (targetDf)
}







#===============================================#
#===============================================#
#===============================================#
#===============================================#
#===============================================#

needToBeDone <- function () {
  MergedDeals$isBrokerInsuredStateSame = factor(ifelse(as.character(MergedDeals$Broker_State) == as.character(MergedDeals$Insured_State),1,0))
  
  
  
  MergedDeals$PolicyDuration <-
    as.numeric(as.Date(as.character(MergedDeals$Expiration_Date), format="%d.%m.%Y")-
                 as.Date(as.character(MergedDeals$Inception_Date), format="%d.%m.%Y"))
  
  # Function for formating the date format 
  # from "09.06.2013 02:05:42 PM" to %d.%m.%Y
  replace = function(x) {
    y = sub("([0-9][0-9].[0-9][0-9].[0-9]*) .*","\\1",x)
    return (trimws(y))
  }
  
  
  # Removing the null Broker_Segment_L3 values
  nrow(MergedDeals)
  MergedDeals<- MergedDeals[!(is.na(MergedDeals$Broker_Segment_L3)),]
  nrow(MergedDeals)
  
  # Removing the rows with Unknown Broker_Segment_L3
  nrow(MergedDeals)
  MergedDeals <- MergedDeals[(MergedDeals$Broker_Segment_L3 %nin% c('UNKNOWN')),]
  nrow(MergedDeals)
  
  
  colnames(MergedDeals)[which(names(MergedDeals) == "Insured.Region")] = "Insured_Region"
  
  # Create a new Column isBrokerInsuredStateSame;
  # If Broker and Insured States are same, fill the isBrokerInsuredStateSame with 1, else 0
  MergedDeals$isBrokerInsuredStateSame = factor(ifelse(as.character(MergedDeals$Broker_State) == as.character(MergedDeals$Insured_State),1,0))
}
