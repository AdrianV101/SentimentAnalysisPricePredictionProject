# Files of possible interest:
Each python file has a short description at the top explaining its function. With regards to the candlestick graphs, the predicted values are those which are lighter green and pink, whilst the true values are those in darker green and red. Unfortuantely, most of the CSV data files are too large for GitHub to handle. However, you can find the Input and Output data for training the models in SentimentAnalysisPricePredictionProject/Data. The original datasets have been linked in the README. Unless specified otherwise, files are found in the SentimentAnalysisPricePredictionProject folder.

### Model outputs:
- Folder named "Output Graphs". Titles are in reference to the model used and wether the test data is in the same market conditions and value range as the training data.
### Models:
- RandomForestRegressionModel.py
- NeuralNetworkModel.py
- XGBoostModelDraft.py
### Sentiment Analysis
- TweetSentimentAnalysis.py
### Data Manipulation/Cleaning
- OHLCVandSentiment.py
- SentimentCondensing.py
- LinkFilteringTweets.py
- SourceFilteringTweets.py
- BitcoinSpecificTweets.py
- MovementIndicator.py
- OutputDataCreation.py

### Data Obtention
- TwitterScraperV1
- There are also some ultimately unused files to obtain historical crypto data, with AWS DynamoDB integrated for storage in the Folder AWS_CoinAPI


# Introduction
The "aim" of this project was to train a machine learning model into which the sentiment analysis of live tweets and recent OHLCV data can be fed, to predict the next OHLC of Bitcoin (chosen for the large amounts of Internet "chatter" regarding it). That being said, the **true aim** of this project was for me to develop my skills in Data Science and Machine Learning. Whilst there is a large amount of knowledge to gain from learning resources, and they should not be ignored, I believe the best way to do this is through real practice and implementation, even if unsuccessful.

The vision for this project, once completed, began with scraping live tweets from X, cleaning them and running sentiment analysis on them. With that done, the sentiments of tweets within a time frame can be condensed, whilst trying to retain as much information as possible. This information along with the recent historical data of Bitcoin, could then be passed into a ML model trained on historical tweet sentiment and OHLCV data, to predict the next OHLCV.

Naturally, as this is my first large self-led project of this sort, I did not succeed in predicting the price of Bitcoin with high accuracy. It is also safe to assume that if I did succeed, I would be busy relaxing whilst leaving my computer to print me money. That being said, I learnt a multitude of new skills along this journey, had many takeaways, and importantly improvements I would make if approaching this project in the future. Regretfully, I have to leave this project for a couple months in order to focus on my university examinations. The sections below cover in more detail my thought process for each component of the project, and some brief ideas for improvement.

## Twitter Scraper

This project began with me writing my own web scraper for X/Twitter. I was unfortunately too late to the party to use Twitter's API when it was free, and using X’s extortionately priced API is not financially viable for a student such as myself! The web scraper worked surprisingly well, and would obtain any information I wanted of it. 

The main challenge I ran into with the web scraper was X imposing invisible restrictions on the account used, stopping the bot from being able to load any web pages on the account for a period of time. 

The access restrictions were both temporary and limited to the account therefore, the solution to this would be simple. All that would be required would be having the bot cycle through several accounts allowing the restriction on each account to wear off whilst others were used. However, I didn’t implement this solution, as I was aware that this would be an issue only if I had a model to implement in real-time, and I was still far from achieving this goal.

## Data Collection

I began by attempting to find a dataset of historical tweets which refer to bitcoin. I eventually stumbled across [this kaggle dataset](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets?resource=download). . 

Cleaning the data effectively was a challenge which I underestimated, but learnt a large amount from. When developing the web scraper, I witnessed first hand the number of tweets which refer to bitcoin that are clearly attempting to promote a product and/or are uploaded by bots. I decided to use this dataset due to its unique provision of each tweets source. By removing all the tweets which were not uploaded from Twitter Web App or Twitter for iPhone/iPad/Android, there was a clear decrease in spam tweets.

Due to the bitcoin hashtag being used in essentially every tweet that refers to cryptocurrencies, I also decided to only include tweets that mentioned Bitcoin or BTC outside of a hashtag, hopefully resulting in the tweets remaining being of direct relevance. I also removed any tweets that contained links, as this was often an indication that someone was acting with a second agenda, and not simply voicing their opinion into the Internet void.

I then moved on to finding high-quality historical Bitcoin data, and I decided to use [this dataset](https://www.kaggle.com/datasets/jesusgraterol/bitcoin-hist-klines-all-intervals-2017-2023?resource=download), due to the broad range of time intervals it offered. I ultimately only used the 5 minute intervals, and used these to fit 4 hour periods closer to those in my tweet set (eg. 14:35-18:35). The tweet dataset had undergone a lot of thinning, and often had small pockets of tweets that didn’t fit nicely within standard 4 hour intervals.

Afer acquiring the custom interval OHLCV data, I then moved onto calculating some other relevant input features, such as recent averages. However, whilst I considered calculating some more advanced indicators (Bollinger Bands, MACD, VWAP etc.), I decided against using these as the focus for the input features was on sentiment analysis, not technical analysis.

In retrospect, I do think that including these input features would at worst have no impact, and could have potentially resulted in the models being able to acquire a stronger numerical footing, and thus more accurate predictions.

## Sentiment Analysis

The core of this project was the sentiment analysis and I began with using DistilBERT however, it failed to perform very well. As DistilBERT is made to be a "lite" version of BERT, I initially suspected that this was the reason for its low performance. I then decided to use RoBERTA, known for its high accuracy and performance, at the cost of speed. I saw little improvement with this change, and concluded that the issue was likely with the informal language and terminology used on X.

The sentiment analysis definitely could have been better, as it tended to be very optimistic. This meant that the extra information that the price prediction model received from sentiment analysis was minimal, resulting in the model essentially attempting to use only technical analysis, from a limited amount of technical information. When revisiting this project, the first thing I would do is attempt one of the following solutions to the sentiment analysis:

- The implementation of a financial tweet vocabulary to sentiment dictionary and calculating the sentiment of a given text by simply picking out certain words and finding their equivalent sentiment value. This would be much faster than sentiment analysis, and could be fine tuned quite easily to work well with most tweets.
- If that were to make insignificant improvements, I would go through the much longer process of manually labelling the sentiment of tweets (a process I began but quickly abandoned) and then train either RoBERTA or FinBERT (Financial BERT) on this dataset. This would take a lot of time however, sentiment analysis being the core of this project, it would hopefully result in a significant increase in the predictive models performance.

## Predictive Models
### Random Forest Regression
With regards to the model, Random Forest Regression generally provided the best performance however, it certainly had its faults. It was often biased towards movements in a certain direction, the bias direction changing randomly each time it was trained. The movements it predicted were often much larger than those that occurred in the test data. My suspected cause for these issues were:

- I believe that the bias was likely due to the small range of data with which the model was trained (as it was limited to the dates of the tweet dataset I was using for sentiment analysis information). An attempt to solve this issue was made, by attempting to get the model to predict the movement direction and magnitude seperately, and ignoring the high and low values. Unfortunately, this led to little improvement, indicating the model was likely incapable of finding the direction of the movement (this alone would be a massive achievement, if done with a high success rate), and so it was minimising the error by simply deciding a constant direction to move in.

- With regards to the difference in magnitude, this was clearly due to the historical data of bitcoin containing 2 different periods of market behaviour, initially being characterised by a high volatility which then transitioned into a lower volatility period with subdued fluctuations. Once the dataset was partitioned, using only the volatile market, the movements predicted were of the order of magnitude one would expect. The use of more technical indicators as input features, such as Bollinger Bands, could certainly lead to magnitudes more appropriate for each market. The change in market behaviour within the historical data was certainly an issue that applied to any model used (the neural network was a bit more robust to this) but, due to the date range of the tweets used, there would be little option but to start from scratch with a new tweet dataset to remove this issue entirely.

The Random Forest Regression model, I suspect due to its decision tree based origins, was also entirely incapable of predicting the market for values that lay outside of the range of the training data. It would instead predict the values as those at the lower bound of the training data.

The clear benefits of the Random Forest Regression model were in its ability to predict the market reasonably when the general behaviour of the market was constant, particularly with regards to volatility and bull runs etc. The major issue with this model was its inability to predict values that lay outside of the training data range, as was the random bias it would tend to have.

I then attempted an XGBoost model, which unsurprisingly, as it is also decision tree based, led to little improvement. The next model I attempted was a Support Vector Regression, but given its single output nature, it led to little progress.

### Neural network

I finally attempted to tackle the problems using a neural network. With little knowledge on the topic, progress was slow and involved a lot of trial and error. A key issue that the neural network presented was its incredibly strong bias, predicting only either positive movements or negative movements after each training, much like the Random Forest Model. 

I have no doubt that this issue likely lies in a the lack of my domain experience with neural networks, and I likely had some major oversights. I attempted to solve this issue again by trying to predict only the movement (magnitude and sign together). Unfortunately, given the roughly symmetrical nature of these movements, it unsurprisingly resulted in the model minimising error by constantly predicting a movement of magnitude 0 and random direction. 

I attempted to solve this by writing a custom loss function, which penalised predictions more if they were (incorrectly) close to 0, with little success. I then attempted to fix this by again having the model predict the magnitude of movement and the sign as separate variables, however the model was simply incapable of doing this well. With this being said, using a neural network certainly showed some promise, and it dealt with predicting values outside of the training data range much better than Random Forest. Whilst being much more difficult to implement (many architectures were trial and errored before finding one with decent performance), it would be something I would enthusiastically investigate (hopefully with more experience) if revisiting this project.

# Conclusion

The "aim" of this project was unsurprisngly not achieved. It was certainly much more expansive than I expected, and the limited time I was able to give it full attention for was not sufficient. As mentioned above, the primary areas I would choose to focus on if revisiting this project would be input feature selection, the improvement of the sentiment analysis and the use of a broader time range in training data. This project was an incredibly insightful learning experience however, it also highlighted to me which realms are challenging due to their nature as opposed to those that are challenging due to the knowledge needed. 

On the one hand, the building of models was surpringly easy, given my lack of any formal training in the field of machine learning. Of course, these models were not built from scratch, and the use of many libraries certaintly accelerated the process. That being said, there is no doubt that more domain knowledge would make a big difference here, particularly with regards to neural networks. Having little knowledge of how to design the architecture of a neural network, the development of one involved a large amount of trial and error, and by no means is the one I decided on optimal. Random Forest Regression was easy and quick to implement, whilst still being reasonably effective. Other models were also quite easy to implement, but a lack of deep understanding in some of their hyperparameters, for instance when using SVR, meant more trial and error and time inefficiencies.

On the other hand, I found that even with a strong understanding of data collection and cleaning, the sheer size and non-uniformity present in real-world data was an inherent challenge. The real-world aspect here is key, because being a student one grows complacent to problems **designed** to be presented to you and solved by following a limited range of recently introduced techniques, where even the noise present in your data is intentionally introduced and produced by some intentional function. This project opened my eyes to not only dealing with real data, which will have missing values, innacuracies and sub-optimal presentation, but just how much work needs to be put into ensuring that it can be cleaned and selected in such a way that its mathematical roots can be revealed. The selection process, and knowing its importance, was something this project taught me. I believe that a key oversight in this project was the feature engineering, which whilst derived from one's principle hypothesis and model selection, is limited by the data available to you and your manipulation of it.

Whilst the "aim" of this project was not achieved, its true aim of exposing me through first-hand experience the many facets of data science and machine learning, from start to finish, was a complete success. Wether it be when I revisit this project or start another, I have no doubt that my new-found experience will be incredibly impactful on its development and achievement.
