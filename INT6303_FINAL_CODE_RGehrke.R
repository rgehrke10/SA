#Ricardo Gehrke
#November 20, 2024
#LTU, INT 6303

install.packages()
#install packages: tm, SnowballC, wordcloud, wordcloud2, 
#RColorBrewer, syuzhet, RSentiment, sentimentr, ggplot2, reshape2



### CLEANING DATA ###
library(tm)
library(SnowballC)
library(quanteda)
library(syuzhet)
library(dplyr)
library(ggplot2)
library(quantmod)
library(dplyr)
library(ggplot2)
library(e1071)
library(caret)
#importing libraries

stock_tweets=read.csv('stock_tweets.csv',header=T)
#importing file stock tweets.csv

tesla_tweets <- subset(stock_tweets, Stock.Name == "TSLA")
summary(tesla_tweets)
#Utilizing only the tesla tweets from the dataframe

tesla_corpus=Corpus(VectorSource(tesla_tweets$Tweet))
#transforming it to corpus

tesla_corpus=tm_map(tesla_corpus, content_transformer(tolower))
tesla_corpus=tm_map(tesla_corpus, removeWords, stopwords('english'))
tesla_corpus=tm_map(tesla_corpus, removePunctuation)
tesla_corpus=tm_map(tesla_corpus, removeNumbers)
tesla_corpus=tm_map(tesla_corpus, stripWhitespace)
#cleaning data: all lower case, remove words/stopwords, punctuations, numbers, whitespaces


### TURNING DATA INTO CORPUS ###

tesla_dtm <- DocumentTermMatrix(tesla_corpus)
#Tokenize the cleaned corpus into a Document-Term Matrix

#View the structure of the DTM
inspect(tesla_dtm[1:5, 1:10])  
#Show the first 5 documents and 10 terms

head(tesla_dtm)

tesla_dtm <- removeSparseTerms(tesla_dtm, 0.99)  
#Keep terms appearing in at least 1% of documents


total_entries <- prod(dim(tesla_dtm))
non_zero_entries <- length(tesla_dtm$v)
sparsity <- (1 - (non_zero_entries / total_entries)) * 100
sparsity
#measuring sparsity

### BEGIN SENTIMENT ANALYSIS ###

### RULE-BASED ALGORITHM ###

#Use the Tweet column for sentiment analysis
#Convert the Tweet column to a character vector, this is required by function get_sentiment
tesla_tweets$Tweet <- as.character(tesla_tweets$Tweet)

#Calculate sentiment scores using the Tweet column and classify
tesla_tweets$sentiment<- get_sentiment(tesla_tweets$Tweet)
tesla_tweets <- tesla_tweets %>%
  mutate(sentiment_label = case_when(
    sentiment > 0 ~ "Positive",
    sentiment < 0 ~ "Negative",
    TRUE ~ "Neutral"
  ))

print(table(tesla_tweets$sentiment_label))
#Check sentiment distribution


### AGGREGATE SCORES TO THEIR RESPECTIVES DATES ###

#Ensure the Date column is in Date format
tesla_tweets$Date <- as.Date(tesla_tweets$Date)

#Aggregate sentiment scores by date
daily_sentiment <- tesla_tweets %>%
  group_by(Date) %>%
  summarise(
    average_sentiment = mean(sentiment, na.rm = TRUE),
    positive_count = sum(sentiment_label == "Positive"),
    negative_count = sum(sentiment_label == "Negative"),
    neutral_count = sum(sentiment_label == "Neutral")
  )

#View the aggregated data
print(head(daily_sentiment))


### FINANCIAL PART AND COMBINING DATABASES BY DATE ###

#Load Tesla stock prices
getSymbols("TSLA", src = "yahoo", from = "2021-09-30", to = "2022-09-28")

#Convert xts object to data.frame
TSLA_df <- fortify.zoo(TSLA)

#Rename columns for better understanding
colnames(TSLA_df) <- c("Date", "Open", "High", "Low", "Close", "Volume", "Adjusted")

#View the first few rows of the data
head(TSLA_df)

#Ensure TSLA_df and daily_sentiment date columns are in Date format
TSLA_df$Date <- as.Date(TSLA_df$Date)
daily_sentiment$Date <- as.Date(daily_sentiment$Date)

#Merge stock prices with aggregated sentiment data
combined_data <- merge(TSLA_df, daily_sentiment, by = "Date", all.x = TRUE)

#Replace any NAs in sentiment columns with 0 (optional, depending on how you want to handle missing data)
combined_data[is.na(combined_data)] <- 0

#Calculate stock price changes (percentage change) properly
TSLA_df$price_change <- c(NA, diff(TSLA_df$Close) / TSLA_df$Close[-nrow(TSLA_df)] * 100)

#View combined data
head(combined_data)

#Calculate stock price changes (percentage change) properly
TSLA_df$price_change <- c(NA, diff(TSLA_df$Close) / TSLA_df$Close[-nrow(TSLA_df)] * 100)

#Merge sentiment and stock data
sentiment_pricechange <- left_join(TSLA_df, daily_sentiment, by = "Date")
head(sentiment_pricechange)

### CORRELATION ###

#Calculate correlation
correlation <- cor(sentiment_pricechange$price_change, sentiment_pricechange$average_sentiment, use = "complete.obs")
print(correlation)

#correlation of 0.171978




### APPLYING NAIVE BAYES ###

sentiment_pricechange$price_change_label <- ifelse(sentiment_pricechange$price_change > 0, "Increase", "Decrease")
sentiment_pricechange <- sentiment_pricechange[complete.cases(sentiment_pricechange), ]  # Remove any rows with NA values

#Convert to factor
sentiment_pricechange$price_change_label <- as.factor(sentiment_pricechange$price_change_label)

#Ensure reproducibility
set.seed(123)

#Split the data (80% for training, 20% for testing)
train_index <- createDataPartition(sentiment_pricechange$price_change_label, p = 0.8, list = FALSE)
train_data <- sentiment_pricechange[train_index, ]
test_data <- sentiment_pricechange[-train_index, ]


#Train the model using average_sentiment to predict price_change_label
naive_bayes_model <- naiveBayes(price_change_label ~ average_sentiment, data = train_data)

#View model summary
print(naive_bayes_model)

#Predict on the test data
test_data$predictions <- predict(naive_bayes_model, newdata = test_data)

#View a confusion matrix
confusion_matrix <- table(Predicted = test_data$predictions, Actual = test_data$price_change_label)
print(confusion_matrix)


#Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy, 2)))

#Evaluate Precision, Recall, and F1-Score
library(caret)
conf_matrix <- confusionMatrix(as.factor(test_data$predictions), as.factor(test_data$price_change_label))
print(conf_matrix)





#Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy, 2)))

#Plot predictions vs actual
ggplot(test_data, aes(x = price_change_label, fill = predictions)) +
  geom_bar(position = "dodge") +
  labs(title = "Predicted vs Actual Price Change",
       x = "Actual Price Change",
       y = "Count") +
  theme_minimal()

### 59% accuracy

summary(tesla_tweets$sentiment)
hist(tesla_tweets$sentiment)


### FINE TUNNING THE SENTIMENT CLASSIFICATION ###

tesla_tweets <- tesla_tweets %>%
  mutate(sentiment_label = case_when(
    sentiment >= 1 ~ "Strong Positive",      # Scores 1 and above are strong positive
    sentiment <= -1 ~ "Strong Negative",     # Scores -1 and below are strong negative
    sentiment > -1 & sentiment < 1 ~ "Neutral" # Scores between -1 and 1 are neutral
  ))

#Check the distribution of sentiment labels
table(tesla_tweets$sentiment_label)

#Weighting sentiment scores
tesla_tweets <- tesla_tweets %>%
  mutate(weighted_sentiment = case_when(
    sentiment >= 1 ~ sentiment * 2,    # Double weight for strong positive sentiment
    sentiment <= -1 ~ sentiment * 2,   # Double weight for strong negative sentiment
    TRUE ~ sentiment                    # No weight for neutral sentiment
  ))

# Recalculate average weighted sentiment for each date
daily_sentiment <- tesla_tweets %>%
  group_by(Date) %>%
  summarise(
    average_weighted_sentiment = mean(weighted_sentiment, na.rm = TRUE),
    positive_count = sum(sentiment_label == "Strong Positive"),
    negative_count = sum(sentiment_label == "Strong Negative"),
    neutral_count = sum(sentiment_label == "Neutral")
  )

#Merge the weighted sentiment data with the stock data
combined_data <- left_join(TSLA_df, daily_sentiment, by = "Date")

#Calculate correlation between the weighted sentiment and price change
correlation <- cor(combined_data$average_weighted_sentiment, combined_data$price_change, use = "complete.obs")
print(correlation)

head(combined_data)


### REAPPLYING NAIVE BAYES ###

#Load required library

combined_data$price_change_label <- ifelse(combined_data$price_change > 0, "Increase", "Decrease")
combined_data <- combined_data[complete.cases(combined_data), ]  # Remove any rows with NA values

# Convert to factor
combined_data$price_change_label <- as.factor(combined_data$price_change_label)

# Ensure reproducibility
set.seed(123)

# Split the data (80% for training, 20% for testing)
train_index <- createDataPartition(combined_data$price_change_label, p = 0.8, list = FALSE)
train_data <- combined_data[train_index, ]
test_data <- combined_data[-train_index, ]


# Train the model using average_sentiment to predict price_change_label
naive_bayes_model <- naiveBayes(price_change_label ~ average_weighted_sentiment, data = train_data)

# View model summary
print(naive_bayes_model)

# Predict on the test data
test_data$predictions <- predict(naive_bayes_model, newdata = test_data)

# View a confusion matrix
confusion_matrix <- table(Predicted = test_data$predictions, Actual = test_data$price_change_label)
print(confusion_matrix)


# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy, 2)))

# Evaluate Precision, Recall, and F1-Score
library(caret)
conf_matrix <- confusionMatrix(as.factor(test_data$predictions), as.factor(test_data$price_change_label))
print(conf_matrix)



### APPLYING TIME LAG ###

# Example: Applying a 1-day lag to sentiment
daily_sentiment <- tesla_tweets %>%
  group_by(Date) %>%
  summarise(
    average_weighted_sentiment = mean(weighted_sentiment, na.rm = TRUE),
    positive_count = sum(sentiment_label == "Strong Positive"),
    negative_count = sum(sentiment_label == "Strong Negative"),
    neutral_count = sum(sentiment_label == "Neutral")
  )

# Apply a 1-day lag to the sentiment scores
daily_sentiment$lagged_sentiment <- lag(daily_sentiment$average_weighted_sentiment, 1)

# Merge the lagged sentiment data with the stock data
combined_data <- left_join(TSLA_df, daily_sentiment, by = "Date")

# Now you can calculate the correlation between lagged sentiment and stock price changes
head(combined_data)$price_change <- c(NA, diff(combined_data$Close) / head(combined_data$Close, -1) * 100)

correlation <- cor(combined_data$price_change, combined_data$lagged_sentiment, use = "complete.obs")
print(correlation)

## Correlation of 0.03
