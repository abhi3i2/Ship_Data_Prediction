
# Installing Packages
install.packages("psych", dependencies = TRUE, repos='http://cran.rstudio.com')
install.packages("car")
install.packages("lmtest", dependencies = TRUE, repos='http://cran.rstudio.com')
install.packages("caTools")


library(caTools)
library(lmtest)
library(forecast)
library(psych)
library(car)

##1. Opening and Processing the Dataset
getwd()

data <- read.csv("Ship.csv")

nrow(data)
str(data)
summary(data)


model1 <- lm(TEU ~ Deadweight + Depth + Displacement + Draught + Length, data = data)
model1
summary(model1)

model1$fitted.values
model1$residuals
plot(model1$fitted.values, model1$residuals)
# Example: Diagnostic plots
plot(model1)
qqPlot(model1, main = "QQ Plot")
options(scipen=999)

values = as.data.frame(cbind(data$TEU, model1$fitted.values, model1$residuals))


accuracy(model1$fitted.values, data$TEU)
# Predict values using the model
data$Predicted_TEU <- predict(model1)
# Evaluate model accuracy
accuracy(data$Predicted_TEU, data$TEU)


##2.Running Exploratory Analysis

str(data)

#Plotting Histograms
plot(density(data$TEU))
hist(data$TEU) 
hist(data$Displacement) 
hist(data$Deadweight) 
hist(data$Length) 
hist(data$Draught) 
hist(data$Depth)



#Plotting Scatterplots between quantitative variables
plot(data$Displacement, data$TEU)
plot(data$Deadweight, data$TEU)
plot(data$Depth, data$TEU)
plot(data$Length, data$TEU)
plot(data$Draught, data$TEU)


#Examining Correlations
str(data)
cor(data[c(3,4:8)])

# Check for correlations
cor_matrix <- cor(data[, c(3,4:8)])
print(cor_matrix)


##3. Test for Assumptions - Normality, Linearity, Multicollinearity, 
## Correlated Errors, Homoscedasticity, Influential Observations / Outliers

vif_values <- car::vif(lm(TEU ~ Deadweight + Displacement + Draught + Depth + Length, data = data))
print(vif_values)


# Remove "Deadweight" and "Displacement" due to high multicollinearity
data_cleaned <- data[, c("TEU", "Displacement","Draught", "Depth", "Length")]

# Rebuild the model without "Deadweight" 
lm_model_cleaned <- lm(TEU ~ Displacement + Draught + Depth + Length, data = data_cleaned)

# Calculate VIF for the new model
vif_values_cleaned <- car::vif(lm_model_cleaned)
print(vif_values_cleaned)

model2=lm_model_cleaned

# Normality of Residuals
qqnorm(data$TEU)
hist(data$TEU)

# Multicollinearity
vif(model1) #old value of VIF
vif(model2)

# Correlated Errors
durbinWatsonTest(model1)
durbinWatsonTest(model2)

# Homoscedasticity
plot(model1$fitted.values, model1$residuals)
plot(model2$fitted.values, model2$residuals)
ncvTest(model1)
ncvTest(model2)

# Testing for Linearity
crPlots(model1)
crPlots(model2)

#Testing Outliers
outlierTest(model1)
outlierTest(model2)

# Testing Influential Observations - Cook's D plot
cutoff <- 4/((nrow(data) - length(model2$coefficients) - 2))
plot(model2, which = 4, cook.levels = cutoff)


#5.Data Mining Approach

baseline = mean(data$TEU)
baseline #baseline is 4431

library(caTools)
set.seed(10000)
split = sample.split(data$TEU,SplitRatio=0.7)
train1= subset(data,split==TRUE)
test1= subset(data,split==FALSE)
str(train1)
str(test1)
hist(train1$TEU)
hist(test1$TEU)


predictTest1 = predict(model2, newdata=test1)
predictTest1
accuracy(predictTest1, test1$TEU)


modeldm <- lm(TEU ~ Depth + Displacement + Draught + Length, data = train1)
modeldm
summary(modeldm)


plot(modeldm$fitted.values, modeldm$residuals)
accuracy(modeldm$fitted.values, train1$TEU)

predictTest2 = predict(modeldm, newdata=test1)
predictTest2
accuracy(predictTest2, test1$TEU)

predictTest = predict(model2, newdata=test1)
predictTest
accuracy(predictTest, test1$TEU)

crPlots(model2) 
outlierTest(model2) 
qqPlot(model2, main="QQ Plot")
leveragePlots(model2)

crPlots(modeldm) 
outlierTest(modeldm) 
qqPlot(modeldm, main="QQ Plot")
leveragePlots(modeldm)

summary(model2)
summary(modeldm)

##6. Regularization - Lasso and Rigde Regression
#LASSO - Least absolute shrinkage and selection operator
#Training Lasso regression in k-fold cross-validation framework
#Shrinkage: Reduce overfitting, Select only most important predictor variables
#Regularization (L1)

library(caret)
set.seed(1000)

ctrlspecs = trainControl(method="cv", number=10, savePredictions = "all")

####Specify and Train Lasso regression Model

# Create vector of potential lambda values
#lambda_vector = seq(0.001, 0.1, by = 0.0005)
lambda_vector = 10^seq(5,-5, length = 500)

#Specify LASSO regression model to be estimated
str(train1)
model_Lasso = train(data = train1, TEU ~ ., preProcess = c("center", "scale"), method="glmnet", tuneGrid=expand.grid(alpha=1, lambda=lambda_vector),trControl=ctrlspecs, na.action=na.omit)
model_Lasso = train(data = train1, TEU ~ ., method="glmnet", tuneGrid=expand.grid(alpha=1, lambda=lambda_vector),trControl=ctrlspecs, na.action=na.omit)

#Best / Optimal tuning parameter (alpha, lambda)
model_Lasso$bestTune
model_Lasso$bestTune$lambda 
#RMSE of the best tuned model
min(model_Lasso$results$RMSE)

#Actual Lasso regression model coefficients (parameter estimates) which reduce the variance around these coefficients
coef(model_Lasso$finalModel, model_Lasso$bestTune$lambda)
options(scipen=999)

#Plot log(lambda) & RMSE - Using log for visualization
plot(log(model_Lasso$results$lambda), model_Lasso$results$RMSE, xlab="log(lambda)", ylab="RMSE")
plot(log(model_Lasso$results$lambda), model_Lasso$results$RMSE, xlab="log(lambda)", ylab="RMSE", xlim=c(0,5))
log(model_Lasso$bestTune$lambda)

#Plot log(lambda) & Rsquared - Using log for visualization
plot(log(model_Lasso$results$lambda), model_Lasso$results$Rsquared, xlab="log(lambda)", ylab="RSquared")
plot(log(model_Lasso$results$lambda), model_Lasso$results$Rsquared, xlab="log(lambda)", ylab="RSquared", xlim=c(0,5))
log(model_Lasso$bestTune$lambda)

