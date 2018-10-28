### Literacy Review

#### Selecion of Dataset

+ a well-known dataset will have fairly low alpha content, although they could be useful in a diversified risk premia portfolio

+ we would prefer processed signals and insights instead of a large amount of raw data. "The highest level of data processing happens when data is presented in the form of research reports, alerts or trade ideas". A lesser form would be a signal that be fed into a multi-signal trading model. Most commonly, we would have semi-processed dataset that needs assessment and needs sequence of cleaning (seasonality, outliers removal, etc). raw data would barely be useful to us

#### Quality of Data
+ data with longer history is more valuable
+ we should be really careful with backfilled datasets

#### Sentiment Approaches - iSentium

+ first, filter by tweet volume and realized volatility
+ second, use NLP to assign sentiment score for each tweet
+ aggregate tweet score for each stock and smooth score
+ predict log return using smoothed aggregated sentiment score with stock volume taken into consideration

#### Sentiment Approaches - Ravenpack

+ first, map each event with an entity name (currency)
+ second, use NLP to generate relevance score and filter out low relevant news.
+ third, use NLP to generate an event sentiment score that represents news sentiment for a stock (-1.0 to +1.0) and the average of the score for all filtered events is the sentiment value for the day (for the group of interested stocks). 

#### What we can do

Our news dataset has many columns and not all of them are useful. First, we should clean the raw data into a refined signal. There are several steps:

+ each tweet should be related to a specific stock (this step is like step one in ravenpack approach)
+ we can then look at stock with enough tweets associated with me (like step one in iSentium approach)
By now, we have tweets with enough volume for a group of interested stock. next. What we will do next is to build a score:

+ for each stock, we should generate an aggregated sentiment score for that specific day with enough data cleaning and seasonality considerations. 
+ by now, we should have a much refined signal and we can start playing with how does this signal interact with other types of beta or alpha signals (like volume, imbalance, etc using trade data). 

 

