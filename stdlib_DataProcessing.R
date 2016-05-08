############################################################################################################
# File : stdlib_DataProcessing.R
# Contents : Standard Library for all data processing done explored so far
# Goal : Can be used in any R projects for speeding up the Machine learning & Data Science Process
# Author: Jiby Babu
############################################################################################################

getDataFrameAfterRemovingColumn <- function (df,column_name) {
  df <- df[-which( colnames(df)==column_name)]
  return (df)
}

getDataFrameAfterRemoveUniqueElementsInColumn <- function (df, column_name, uniqueCount) {
  df <- sqldf("SELECT * FROM consent WHERE Specialty 
              IN (SELECT Specialty FROM consent GROUP BY Specialty HAVING COUNT(*) > 2) 
              ORDER BY Specialty")
  return (df)
}

#### Dummy variable Creation: ###
# Referrence: 
# http://stackoverflow.com/questions/11952706/generate-a-dummy-variable-in-r
CreateDummies <- function (df,colName) {
  colName = factor(colName)
  colName_dummies <- as.data.frame(model.matrix(~colName))
  colName_dummies <- colName_dummies[-which(colnames(colName_dummies)=="(Intercept)")]
  colName_dummies <- as.data.frame(lapply(colName_dummies, factor))
  cat("Newly Created Dummies are following","\n")
  names(colName_dummies)
  df <- cbind(MergedDeals,colName_dummies)
  return (df)
}

isMissingValueExists <- function(df) {
  return (any(is.na(df)))
}

getColumnNumber <- function(df,columnName) {
  return (which(colnames(df)==columnName))
}

removeNAvalues <- function(df) {
  return (na.omit(df))
}

getNumericColumns <- function (df) {
  nums <- sapply(df, is.numeric)
  numerical_columns <- df[,nums]
  return (numerical_columns)
}

getCharacterColumns <- function (df) {
  char <- sapply(df, is.character)
  charecter_columns <- df[,char]
  return (charecter_columns)
}

removeColumnFromDataFrame <- function (df, columnName) {
  return (df[-which( colnames(columnName) == columnName)])
}

writeFile <- function (df,fileName) {
  write.csv(submission, fileName, row.names=F, quote=F)
}

convertCharFactsToInt = function (df) {
  cat("assuming text variables are categorical & replacing them with numeric ids\n")
  cat("re-factor categorical vars & replacing them with numeric ids\n")
  
  featureNamesSet <- names(df)
  for (eachFeature in featureNamesSet) {
    if (class(df[[eachFeature]])=="character" || class(df[[eachFeature]])=="factor") {
      df[[eachFeature]] <- as.integer(factor(df[[eachFeature]]))
    }
  }
  return (df)
}

convertInteger64ToInt = function (df) {
  
  cat("Converting numerical integer64 data type to integer data type \n")
  featureNamesSet <- names(df)
  for (eachFeature in featureNamesSet) {
    if (class(df[[eachFeature]])=="integer64") {
      df[[eachFeature]] <- as.integer(df[[eachFeature]])
    }
  }
  return (df)
}

##### Removing constant features   #####
getConstantFeatures <- function (df) {
  constantFeatures <- c()
  cat("\n## Removing the constants features.\n")
  for (f in names(df)) {
    if (length(unique(df[[f]])) == 1) {
      cat(f, "\n")
      constantFeatures <- c(constantFeatures, f)
    }
  }
  return (constantFeatures)
}


##### Removing identical features ##### 
getIdenticalFeatures <- function (df) {
  
  features_pair <- combn(names(df), 2, simplify = F)
  toRemove <- c()
  
  for(pair in features_pair) {
    f1 <- pair[1]
    f2 <- pair[2]
    
    if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
      condition <- all(df[[f1]] == df[[f2]])
      if (!is.na(condition)) {
        if(condition) {
          cat(f1, "and", f2, "are equals.\n")
          toRemove <- c(toRemove, f2)
        }
      }
    }
  }
  return (toRemove)
}

removeIdenticalRows <- function (df) {
  df <- df[!duplicated(df),]
  return (df)
}

plotHistogram <- function(df) {
  for(i in 1:length(df)) {
    if(is.numeric(df[,i])){  
      hist(as.numeric(df[,i]),main=names(df)[i])
    } else {
      cat("Non numeric Column Found while plotting histogram!!" + "Column Name:" + names(df)[i])
    }
  }
}

plotBarplot <- function(df) {
  for(i in 1:length(df)) {
    if(is.character(df[,i])){
      barplot(prop.table(table(df[,i])),main=names(df)[i])
    } else {
      cat("Non Charector Column Found while plotting barplot!!" + "Column Name:" + names(df)[i])
    }
  }
}
