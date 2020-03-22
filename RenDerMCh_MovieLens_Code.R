################################
# Loading DATA as per project instructions
################################

# installing packages if necessary:
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>%semi_join(edx, by = "movieId") %>% semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed) #removing the unnecessary

################################
# Project itself
################################


# 1.) Preping data
################################

# Creating TRAIN and TEST data sets to avoid using validation set for testing of the trained models

set.seed(10221, sample.kind = "Rounding")            # if using R 3.5 or earlier, use `set.seed(10221)` instead - this is so that the data remains the same for all who run the code
test_index <- createDataPartition(y = edx$rating, 
  times = 1, p = 0.1, list = FALSE)                 # randomly select index of 10% of date from "edx" set 
train_set <- edx[-test_index,]                # create train_set by removing the 10% data
test_set <- edx[test_index,]                  # create test_set of only the 10% of data

train_set<- train_set %>% mutate(dates=(as.Date(as.POSIXct(train_set$timestamp, origin="1970-01-01"))))  # modifying train set to include a column that tranforms timestamp to date (excluding time)
test_set<- test_set %>% mutate(dates=(as.Date(as.POSIXct(test_set$timestamp, origin="1970-01-01"))))    # modifying train set to include a column that tranforms timestamp to date (excluding time)

test_set <- test_set %>%                            # removing users, movies & dates that do not appear in the train_set
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId") %>%
  semi_join(train_set, by = "dates")

# Creating function for measurement of the error of our models
RMSE <- function(true, predicted){sqrt(mean((true - predicted)^2))}

# 2.) Bulding models
################################

mu <- mean(train_set$rating)                      # average of all movie rating in train_set

# A.) MOVIE EFFECT MODEL

movie_avgs <- train_set %>%                       # calculating difference from average for each movie      
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + test_set %>%            # predicting rating based on movie effect model in test_set
  left_join(movie_avgs, by='movieId') %>% 
  .$b_i

model_M_rmse <- RMSE(predicted_ratings, test_set$rating)    #calculating error in test_set

rmse_results <- data_frame(method = "Movie Effect", LAMBDA = "", RMSE_train = model_M_rmse)  #creating table with results

rmse_results %>% knitr::kable(align=c('l','c','r'))                   # printing result table with column alignment

# B.) USER EFFECT MODEL

user_avgs <- train_set %>%                       # calculating difference from average for each userId      
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu))

predicted_ratings <- mu + test_set %>%            # predicting rating based on user effect model in test_set
  left_join(user_avgs, by='userId') %>% 
  .$b_u

model_U_rmse <- RMSE(predicted_ratings, test_set$rating)    #calculating error in test_set

rmse_results <- bind_rows(rmse_results,                      # adding line to the result table
                          data_frame(method="User Effects",  
                                     LAMBDA = "",
                                     RMSE_train = model_U_rmse))

rmse_results %>% knitr::kable(align=c('l','c','r'))                   # printing result table

# C.) MOVIE + USER EFFECT MODEL

movieuser_avgs <- train_set %>%                         # calculating USER effect after removing Movie effect
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>%                 # predicting rating based on movie and user effect model in test_set
  left_join(movie_avgs, by='movieId') %>%         # adding movie effect by movieID to the test_set
  left_join(user_avgs, by='userId') %>%           # adding user effect by movieID to the test_set
  mutate(pred = mu + b_i + b_u) %>%               # calculating prediction as average + movie effect + user effect
  .$pred

model_MU_rmse <- RMSE(predicted_ratings, test_set$rating)    #calculating error in test_set

rmse_results <- bind_rows(rmse_results,                      # adding line to the result table
                          data_frame(method="Movie + User Effects",  
                                     LAMBDA = "",
                                     RMSE_train = model_MU_rmse))

rmse_results %>% knitr::kable(align=c('l','c','r'))                   # printing result table

# D.) REGULARIZED MOVIE EFFECT MODEL

lambdas <- seq(0, 10, 0.25)                       # priparing parameter to optimise

just_the_sum <- train_set %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())      # preparing Movies for regularization (removing average and counting number of ratings)

rmses <- sapply(lambdas, function(l){             # testing which lamda parameter optimises the errors in test_set
  predicted_ratings <- test_set %>% 
    left_join(just_the_sum, by='movieId') %>%     # adding colums of s & n_i for each movie with their average and sum
    mutate(b_i = s/(n_i+l)) %>%                   # calculating regularized movie effect with penalization for less ratings
    mutate(pred = mu + b_i) %>%                   # creating prediction
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
lambda_m<- lambdas[which.min(rmses)]                # selecting lambda with smallest RMSE

rmse_results <- bind_rows(rmse_results,                      # adding line to the result table
                          data_frame(method="Regularized Movie Effect",  
                                     LAMBDA = as.character(lambda_m),
                                     RMSE_train = min(rmses)))

rmse_results %>% knitr::kable(align=c('l','c','r'))                   # printing result table


# E.) REGULARIZED USER EFFECT MODEL

lambdas <- seq(0, 10, 0.25)                       # priparing parameter to optimise

just_the_sum <- train_set %>% 
  group_by(userId) %>% 
  summarize(s = sum(rating - mu), n_i = n())      # preparing users for regularization (removing average and counting number of ratings)

rmses <- sapply(lambdas, function(l){            # testing which lamda parameter optimises the errors in test_set
  predicted_ratings <- test_set %>% 
    left_join(just_the_sum, by='userId') %>%     # adding colums of s & n_i for each user with their average and sum
    mutate(b_u = s/(n_i+l)) %>%                  # calculating regularized user effect with penalization for less ratings
    mutate(pred = mu + b_u) %>%                  # creating prediction
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
lambda_u<- lambdas[which.min(rmses)]                # selecting lambda with smallest RMSE

rmse_results <- bind_rows(rmse_results,                      # adding line to the result table
                          data_frame(method="Regularized User Effect", 
                                     LAMBDA = as.character(lambda_u),
                                     RMSE_train = min(rmses)
                                     ))

rmse_results %>% knitr::kable(align=c('l','c','r'))                   # printing result table


# F.) REGULARIZED MOVIE & USER EFFECT MODEL

lambdas <- seq(4, 6, 0.1)                                       # priparing parameter to optimise
rmses <- sapply(lambdas, function(l){                           # testing which lamda parameter optimises the errors in test_set
  mu <- mean(train_set$rating)                                  
  b_i <- train_set %>%                                          # calculating penalized movie effect
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>%                                          # calculating penalized user effect after taking into account the movie effect
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <-                                          # creating prediction
    test_set %>% 
    left_join(b_i, by = "movieId") %>%                          # adding in movie effect by movieId
    left_join(b_u, by = "userId") %>%                           # adding in user effect by userId
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

lamda_reg_mu <- lambdas[which.min(rmses)]                       # selecting tuning parameter with smallest error

rmse_results <- bind_rows(rmse_results,                         # adding line to results table
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     LAMBDA = as.character(lamda_reg_mu),
                                     RMSE_train = min(rmses)))

rmse_results %>% knitr::kable(align=c('l','c','r'))             # printing results table

# G.) REGULARIZED MOVIE & USER & GENRE EFFECT MODEL

lambdas <- seq(4, 6, 0.1)                                        # priparing parameter to optimise
rmses <- sapply(lambdas, function(l){                            # testing which lamda parameter optimises the errors in test_set
  mu <- mean(train_set$rating)
  b_i <- train_set %>%                                           # calculating penalized movie effect
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>%                                           # calculating penalized user effect after taking into account the movie effect
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <- train_set %>%                                           # calculating penalized genre effect after taking into account the movie and user effect
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  predicted_ratings <-                                           # calculating predictions
    test_set %>% 
    left_join(b_i, by = "movieId") %>%                           # adding in movie effect by movieId
    left_join(b_u, by = "userId") %>%                            # adding in user effect by userId
    left_join(b_g, by = "genres") %>%                            # adding in genre effect by genres
    mutate(pred = mu + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

lamda_reg_mug <- lambdas[which.min(rmses)]                       # selecting tuning parameter with smallest error

rmse_results <- bind_rows(rmse_results,                         # adding line to results table
                          data_frame(method="Regularized Movie + User + Genre Effect Model",  
                                     LAMBDA = as.character(lamda_reg_mug),
                                     RMSE_train = min(rmses)
                                     ))

rmse_results %>% knitr::kable(align=c('l','c','r'))             # printing results table

# H.) FURTHER REGULARIZATION IDENTIFICATION

train_set2<- train_set %>%                                                   # modifying train_set into a new set to explore further regularization options
  mutate(date=(as.POSIXct(train_set$timestamp, origin="1970-01-01")),        # adding date column
         year=year(as.POSIXct(train_set$timestamp, origin="1970-01-01")),    # adding year column
         month=month(as.POSIXct(train_set$timestamp, origin="1970-01-01")),  # adding month column
         week=week(as.POSIXct(train_set$timestamp, origin="1970-01-01")),    # adding week column
         day=wday(as.POSIXct(train_set$timestamp, origin="1970-01-01")),    # adding weekd day column
         hour=hour(as.POSIXct(train_set$timestamp, origin="1970-01-01")))    # adding hour column

train_set2 %>%                                      # plotting variability and count of date averages
  group_by(date) %>% 
  summarize(b = mean(rating),n=n()) %>% 
  ggplot(aes(b)) + 
  geom_histogram(bins = 20, color = "black") +
  ggtitle("Date")

train_set2 %>%                                      # plotting variability and count of year averages for years with more than 100 ratings
  group_by(year) %>% 
  summarize(b = mean(rating),n=n()) %>% 
  filter(n>=100) %>%
  ggplot(aes(b)) + 
  geom_histogram(bins = 20, color = "black") +
  ggtitle("Year")

train_set2 %>%                                       # plotting variability and count of month averages for months with more than 100 ratings
  group_by(month) %>% 
  summarize(b = mean(rating),n=n()) %>% 
  filter(n>=100) %>%
  ggplot(aes(b)) + 
  geom_histogram(bins = 20, color = "black") +
  ggtitle("Month")

train_set2 %>%                                       # plotting variability and count of week averages for weeks with more than 100 ratings
  group_by(week) %>% 
  summarize(b = mean(rating),n=n()) %>% 
  filter(n>=100) %>%
  ggplot(aes(b)) + 
  geom_histogram(bins = 20, color = "black") +
  ggtitle("Week")

train_set2 %>%                                       # plotting variability and count of week day averages for week days with more than 100 ratings
  group_by(day) %>% 
  summarize(b = mean(rating),n=n()) %>% 
  filter(n>=100) %>%
  ggplot(aes(b)) + 
  geom_histogram(bins = 20, color = "black") +
  ggtitle("Week Day")

train_set2 %>%                                      # plotting variability and count of hour for hours with more than 100 ratings
  group_by(hour) %>% 
  summarize(b = mean(rating),n=n()) %>% 
  filter(n>=100) %>%
  ggplot(aes(b)) + 
  geom_histogram(bins = 20, color = "black") +
  ggtitle("Hour")


# based on visuall inspecting the plots - date has a large (relatively) variability of the average and contains
# also week, month and day information therefore going with DATE as the next parameter for regularization
# after DATE will also use HOUR to verify if there is much change in the error

# I.) REGULARIZED MOVIE & USER & GENRE & DATE EFFECT MODEL

lambdas <- seq(4, 6, 0.1)                                         # priparing parameter to optimise

rmses <- sapply(lambdas, function(l){                             # testing which lamda parameter optimises the errors in test_set
  mu <- mean(train_set$rating)
  b_i <- train_set %>%                                            # calculating penalized movie effect
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>%                                            # calculating penalized user effect after taking into account the movie effect
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <- train_set %>%                                            # calculating penalized genre effect after taking into account the movie and user effect
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  b_d <- train_set %>%                                            # calculating penalized date effect after taking into account the movie, user  and genre effect
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    group_by(dates) %>%
    summarize(b_d = sum(rating - b_i - b_u - b_g - mu)/(n()+l))
  predicted_ratings <-                                            # calculating predictions
    test_set %>% 
    left_join(b_i, by = "movieId") %>%                            # adding in movie effect by movieId
    left_join(b_u, by = "userId") %>%                             # adding in user effect by userId
    left_join(b_g, by = "genres") %>%                             # adding in genre effect by genres
    left_join(b_d, by = "dates") %>%                              # adding in date effect by dates
    mutate(pred = mu + b_i + b_u + b_g + b_d) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

lamda_reg_mugd <- lambdas[which.min(rmses)]                       # selecting tuning parameter with smallest error

rmse_results <- bind_rows(rmse_results,                           # adding line to results table
                          data_frame(method="Regularized Movie + User + Genre + Date Effect Model",  
                                     LAMBDA = as.character(lamda_reg_mugd),
                                     RMSE_train = min(rmses)
                                     ))

rmse_results %>% knitr::kable(align=c('l','c','r'))              # printing results table

# J.) REGULARIZED MOVIE & USER & GENRE & DATE & HOUR EFFECT MODEL

lambdas <- seq(4, 6, 0.1)                                       # priparing parameter to optimise

train_set<- train_set %>% mutate(hour=hour(as.POSIXct(train_set$timestamp, origin="1970-01-01"))) # adding a column of hour to the train_set (as previously the graphs were done from a modified train_set2)
test_set<- test_set %>% mutate(hour=hour(as.POSIXct(test_set$timestamp, origin="1970-01-01")))    # adding a column of hour to the test_set

rmses <- sapply(lambdas, function(l){                            # testing which lamda parameter optimises the errors in test_set
  mu <- mean(train_set$rating)
  b_i <- train_set %>%                                           # calculating penalized movie effect
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>%                                           # calculating penalized user effect after taking into account the movie effect
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <- train_set %>%                                           # calculating penalized genre effect after taking into account the movie and user effect
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  b_d <- train_set %>%                                           # calculating penalized date effect after taking into account the movie, user  and genre effect
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    group_by(dates) %>%
    summarize(b_d = sum(rating - b_i - b_u - b_g - mu)/(n()+l))
  b_h <- train_set %>%                                           # calculating penalized hour effect after taking into account the movie, user, genre and date effect
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_d, by = "dates") %>%
    group_by(hour) %>%
    summarize(b_h = sum(rating - b_i - b_u - b_g - b_d - mu)/(n()+l))
  predicted_ratings <-                                           # calculating predictions
    test_set %>% 
    left_join(b_i, by = "movieId") %>%                           # adding in movie effect by movieId
    left_join(b_u, by = "userId") %>%                            # adding in user effect by userId
    left_join(b_g, by = "genres") %>%                            # adding in genre effect by genres
    left_join(b_d, by = "dates") %>%                             # adding in date effect by genres
    left_join(b_h, by = "hour") %>%                              # adding in hour effect by genres
    mutate(pred = mu + b_i + b_u + b_g + b_d + b_h) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

lamda_reg_mugdh <- lambdas[which.min(rmses)]                     # selecting tuning parameter with smallest error
rmse_results <- bind_rows(rmse_results,                          # adding line to results table
                          data_frame(method="Regularized Movie + User + Genre + Date + Hour Effect Model",  
                                     LAMBDA = as.character(lamda_reg_mugdh),
                                     RMSE_train = min(rmses)
                                     ))

rmse_results %>% knitr::kable(align=c('l','c','r'))              # printing results table


# As the model including HOUR regularization did not provide any further improvement we set
# the model with regularized MOVIE + USER + GENRE + DATE model as the final model and proceed to predicting on validation set

# 3.) Training selected model on whole EDX set
###########################################

edx2<- edx %>% mutate(dates=(as.Date(as.POSIXct(edx$timestamp, origin="1970-01-01")))) # modifying EDX set to include date column

l<-lamda_reg_mugd                                         # assigning the lambda from minimalized error on training set

b_i <- edx2 %>%                                           # calculating penalized movie effect on whole EDX set
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))
b_u <- edx2 %>%                                           # calculating penalized user effect on whole EDX set after taking into account the movie effect
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))
b_g <- edx2 %>%                                           # calculating penalized genre effect on whole EDX set after taking into account the movie and user effect
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
b_d <- edx2 %>%                                           # calculating penalized date effect on whole EDX set after taking into account the movie, user and genre effect
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  group_by(dates) %>%
  summarize(b_d = sum(rating - b_i - b_u - b_g - mu)/(n()+l))

# 4.) RMSE on Validation set
################################
validation2<- validation %>% mutate(dates=(as.Date(as.POSIXct(validation$timestamp, origin="1970-01-01")))) # modifying Validation set to include date column

predicted_ratings <-                                     # calculating predictions
  validation2 %>% 
  left_join(b_i, by = "movieId") %>%                     # adding in movie effect by movieId
  left_join(b_u, by = "userId") %>%                      # adding in user effect by movieId
  left_join(b_g, by = "genres") %>%                      # adding in genre effect by movieId
  left_join(b_d, by = "dates") %>%                       # adding in date effect by movieId
  mutate(pred = mu + b_i + b_u + b_g + b_d) %>%
  .$pred

final_RMSE<- RMSE(predicted_ratings, validation2$rating)     # calculating final RMSE on validation set

c<-c("","","","","","","",round(final_RMSE,digits=5),"")     # creating a vector for input to column in table
rmse_results<-rmse_results %>% mutate(RMSE_final=c)          # adding column with final result to result table
rmse_results %>% knitr::kable(align=c('l','c','r','r'))      # printing result table



