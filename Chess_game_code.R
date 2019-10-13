
setwd("C:/Users/rohit/Desktop/Files/Miscellaneous/Project help for kids/Akhil")

rm(list=ls(all=TRUE))

## Importing packages
library(tidyr)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(car)


## Reading in files
chess <- read.csv("data/games_previous_values.csv")
#chess <- read.csv("data/games.csv")

#view chess dataframe
head(chess,2)


#total_different  players
length(unique(chess$black_id)) #9331
length(unique(chess$white_id)) #9438
players = setNames(as.data.frame(unique(chess$black_id)), c('player'))
players = rbind(players,  setNames(as.data.frame(unique(chess$white_id)), c('player')))
total_unique_players = unique(players$player)
length(total_unique_players)
#15635


#In lichess.com mostly players play rating games 
#Comparing the count of players playing rating games
ggplot(chess,aes(x=toupper(rated)))+geom_bar()+xlab(label = "Rated_Games")+ylab(label = "Frequency")+theme_classic()


#top 10 openings moves based on uses
opening<-filter(summarise(group_by(chess,opening_name), count=length(opening_name)),count>200)


#plotting the moves
ggplot(opening,aes(x=opening_name,y=count))+geom_col()+coord_flip()+theme_classic()


#top 10 opening moves
open_move<-head(arrange(summarise(group_by(filter(chess,winner=="white"),opening_eco),count=length(opening_eco)),desc(count)),10)
open_move


#types of wins either by black or white
ggplot(chess,aes(x=victory_status,fill=winner))+geom_bar(position = "dodge")+theme_classic()


# 10+0 increment is most often played
increment<-filter(summarise(group_by(chess,increment_code), count=length(increment_code)),count>200)
ggplot(increment,aes(x=increment_code,y=count))+geom_col()


# range of player's rating
ggplot(chess,aes(x=white_rating,y=black_rating,color=winner))+geom_point(alpha=0.5)


#seperating by intercept
ggplot(chess,aes(x=white_rating,y=black_rating,color=winner))+geom_point(alpha=0.5)+geom_abline(slope=1,intercept=0)
rating<-rbind(chess$white_rating,chess$black_rating)
hist(rating,angle = 45, col = "grey", border = "white", main = "Player Rating", xlab = "Rating")



#Creating new variables
#Feature Engineering
chess$ratingdif <- chess$white_rating-chess$black_rating

#impact of rating difference on the win loss
ggplot(chess, aes(x= white_rating, y = ratingdif,col=winner))+geom_point(alpha = 0.5)+geom_abline(slope=0,intercept=0)
ggplot(chess, aes(x= winner, y = ratingdif)) + geom_boxplot(alpha = 0.2)+ geom_hline(yintercept = 0, col = "grey")


#Scaling the ratings 
x <- prcomp(cbind(chess$white_rating,chess$black_rating), scale = TRUE)
chess$rating <- x$x[,1]


#Work on type of openings
#New variable n_opening = the number of times played Rare openings (n_opening < 175) 
#tend to be played by higher ranked players
w_rating_by <- chess %>% group_by(chess$opening_eco)
btab <- w_rating_by %>% summarise( n_opening=length(white_rating)) %>% arrange(n_opening)

names(btab)[1]<- c("opening_eco")
chess <- chess %>% left_join(btab)

#Viewing the impact
ggplot(chess, aes(x=n_opening, y = white_rating, col = ratingdif)) + geom_point(alpha = 0.05) +  geom_smooth(method = "loess", col = 2)


#Taking out draws and fitting some predictive model(response = white wins):
#no draw
chess <- chess[chess$winner!="draw",]


chess <- droplevels(chess)
chess <- chess[, c(2,5,7,8,10,12,14,16, 17, 20:26)]

chess$winner <- as.numeric(chess$winner=="white")
head(chess,2)


#Split in train test
limit = floor(nrow(chess)*0.98)
train <- chess[(0:limit),]
test <- chess[(limit:nrow(chess)),]

#80% of the sample size
#smp_size <- floor(0.80 * nrow(chess))
### set the seed to make your partition reproducible
#set.seed(123)
#train_ind <- sample(seq_len(nrow(chess)), size = smp_size)
#train <- chess[train_ind, ]
#test <- chess[-train_ind, ]
#
test_actual <- test$winner
test =  test[, c(-3)]



#Logistic model
fit1 <- glm(winner ~  ratingdif + n_opening + white_user_previous_wins + black_user_previous_wins 
            + white_users_Previous_matches + black_users_Previous_matches, data = train, family = binomial())
summary(fit1)
pred <- predict.glm(fit1, newdata=test, type = "response")
head(pred)
pred = ifelse(pred < 0.5, 0, 1)
table(pred, test_actual, dnn=c("Prediction", "Actual"))
abc = as.data.frame(table(pred, test_actual, dnn=c("Prediction", "Actual")))
abc
acc = (abc$Freq[1] + abc$Freq[4])/ length(test_actual)
acc




#RandomForest Model
library(randomForest)
fit2 <- randomForest(winner ~ rated + ratingdif + n_opening + white_user_previous_wins + black_user_previous_wins 
            + white_users_Previous_matches + black_users_Previous_matches+
              white_rating + black_rating, data = train, family = binomial())
summary(fit2)
pred <- predict(fit2, newdata=test, type = "response")
head(pred)
pred = ifelse(pred < 0.5, 0, 1)
table(pred, test_actual, dnn=c("Prediction", "Actual"))
abc = as.data.frame(table(pred, test_actual, dnn=c("Prediction", "Actual")))
abc
acc = (abc$Freq[1] + abc$Freq[4])/ length(pred)
acc



#XGBoost
library(Matrix)
train_x <- sparse.model.matrix(~., data = as.data.frame(train[,c(1,2,5,6,8,10:14,16)]))
test_x <- sparse.model.matrix(~., data = as.data.frame(test[,c(1,2,4,5,7,9:13,15)]))


xgtrain <- xgb.DMatrix(data = train_x, label = train$winner, missing = NA)
xgtest <- xgb.DMatrix(data = test_x, missing = NA)
rm(train_x)


library(xgboost)
params <- list()
params$objective <- "reg:logistic"
params$eta <- 0.1
params$max_depth <- 5
params$subsample <- 0.9
params$colsample_bytree <- 0.9
params$min_child_weight <- 10
params$eval_metric <- "error"


nrounds = 1000
nfolds = 2
ps = list( max_depth = 7 ,eta = 0.09,objective = "reg:logistic")#,colsample_bytree <- 0.9,
#min_child_weight <- 10)
ms = list( 'error')

cvXgb = xgb.cv(params = ps, 
               xgtrain,
               nrounds = nrounds ,nfold = nfolds,print.every.n = 5,showsd = T,
               metrics = ms,stratified = T, verbose = T ,subsample = 0.7
)

Xgb_1 = xgboost(params = ps, 
                xgtrain,
                nrounds = 430 ,nfold = nfolds,print.every.n = 10,showsd = T,
                metrics = ms,stratified = T, verbose = T ,subsample = 0.7
)

imp_1 <- xgb.importance(feature_names = colnames(test_x),model = Xgb_1)
xgb.plot.importance(imp_1)
colnames(test_x)

# prediction
pred <- predict(Xgb_1, xgtest)
summary(pred)
pred <- ifelse(pred<0,0,ifelse(pred>10,10,pred))
head(pred)
pred = ifelse(pred < 0.5, 0, 1)
table(pred, test_actual, dnn=c("Prediction", "Actual"))
abc = as.data.frame(table(pred, test_actual, dnn=c("Prediction", "Actual")))
abc
acc = (abc$Freq[1] + abc$Freq[4])/ length(pred)
acc



#Enter values of new game
head(test)
new_user = data.frame(matrix(ncol = 12, nrow = 0))
x <- c("rated", "turns", "white_rating", "black_rating", "opening_ply", "white_user_previous_wins", "black_user_previous_wins"
       , "white_users_Previous_matches", "black_users_Previous_matches", "ratingdif", "n_opening")
colnames(new_user) <- x

new_value = list(TRUE, 13, 1400, 1100, 5, 11, 12, 18, 15, 100, 20)

new_user = rbind(new_user, new_value)
colnames(new_user) <- x

new_user_x <- sparse.model.matrix(~., data = as.data.frame(new_user))
xg_new_user <- xgb.DMatrix(data = new_user_x, missing = NA)
pred <- predict(Xgb_1, xg_new_user)

probability_of_white_users_win = pred
probability_of_black_users_win = 1-pred

probability_of_white_users_win
probability_of_black_users_win
