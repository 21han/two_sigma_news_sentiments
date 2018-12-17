import pandas as pd
import numpy as np
from datetime import datetime
import sys
from functools import reduce
from collections import Counter

# when debug mode is on, we only take a sub-sample of total data
debug_mode = True
# when we first load this in notebook, turn reload on. afterwards, turn it off no need to reload data everytime
reload = True

if reload:
    news_train_dir = "./new_train_df.csv"
    news_train_df = pd.read_csv(news_train_dir)

    market_train_dir = "./market_train_df.csv"
    market_train_df = pd.read_csv(market_train_dir)

# globals
news_col_extractor = ["time", "assetCodes", "headline", "urgency", "takeSequence", 
                    "subjects", "audiences", "relevance", 
                    'sentimentClass','sentimentNegative', 'sentimentNeutral', 'sentimentPositive',
                    'noveltyCount12H', 'noveltyCount24H', 'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 
                    'volumeCounts12H','volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D','volumeCounts7D'
                   ]

market_col_extractor = ["time", "assetCode", "volume", "close", "open", 
                        "returnsClosePrevRaw1", "returnsOpenPrevRaw1", "returnsClosePrevRaw10", "returnsOpenPrevRaw10",
                        "returnsOpenNextMktres10", "universe"]

identity = lambda series: reduce(lambda x, y: x, series)
coalesce = lambda x: list(x)

if debug_mode:
    news = news_train_df[60000:70000]
    market = market_train_df[:5000]
else:
    news = news_train_df
    market = market_train_df

assetCode_set = set(market_train_df["assetCode"].unique())

# extract relevant columns based on their descriptions
def extract_df(news_train_df, market_train_df):
    news_df = news_train_df[news_col_extractor]
    market_df = market_train_df[market_col_extractor]
    return news_df, market_df

# given a dataframe with field time convert into datetime, month and week
def extract_time_dependent_features(df, obj = None):
    # 1. get date
    df["datetime"] = df["time"].apply(lambda ts: ts[:10])
    # 2. get month
    if obj == "news":
        return df
    df["month"] = df["datetime"].apply(lambda ts: ts[5:7])
    # 3. get week
    df["week"] = df["datetime"].apply(lambda ts: datetime.strptime(ts, '%Y-%m-%d').strftime('%a'))
    return df

# apply helper aggregator to reduce assetCode in preparation for joining
def assetCodeMapper(assetCodeSet):
    assets = list(eval(assetCodeSet).intersection(assetCode_set))
    if assets == []:
        return None
    else:
        return assets[0]
    
# join if assetCode in assetCodes and time days are the same
def mergeDframes(news_df, market_df):
    anchor = ["datetime", "assetCode"]
    mergedDF = market_df.merge(news_df, on=["datetime", "assetCode"], how="left").dropna()
    return mergedDF

# squash columns so that (datetime, assetCodes) are unique
def squash(res):
    df = res.groupby("datetime")
    df = res.groupby(["datetime", "assetCode"]).agg({'volume': identity,
                                                    'open': identity,
                                                    'close': identity,
                                                    'returnsClosePrevRaw1': identity, 
                                                    'returnsOpenPrevRaw1': identity, 
                                                    'returnsClosePrevRaw10': identity,
                                                    'universe': identity,
                                                    'month': identity,
                                                    'week': identity,
                                                    'headline': coalesce,
                                                    'urgency': coalesce,
                                                    'takeSequence': coalesce,
                                                    'subjects': coalesce,
                                                    'audiences': coalesce,
                                                    'relevance': coalesce,
                                                    'sentimentClass': coalesce, 
                                                    'sentimentNegative': coalesce, 
                                                    'sentimentNeutral': coalesce,
                                                    'sentimentPositive': coalesce, 
                                                    'noveltyCount12H': coalesce, 
                                                    'noveltyCount24H': coalesce,
                                                    'noveltyCount3D': coalesce, 
                                                    'noveltyCount5D': coalesce, 
                                                    'noveltyCount7D': coalesce, 
                                                    'volumeCounts12H': coalesce,
                                                    'volumeCounts24H': coalesce, 
                                                    'volumeCounts3D': coalesce, 
                                                    'volumeCounts5D': coalesce, 
                                                    'volumeCounts7D': coalesce,
                                                    'returnsOpenNextMktres10': identity
                                                   })
    return df

# helper functiuons for urgency related partition calculation
def urgency_helper(x, column, urgency_type):
    relevance = [0 if i == urgency_type else i for i in x.relevance]
    return np.multiply(relevance, column).sum() / sum(relevance)

def urgency_dist_helper(x):
    d = Counter(x)
    if 1 not in d:
        d[1] = 0
    if 3 not in d:
        d[3] = 0
    return d[1], d[3]

def urgency_time_helper(x, column, urgency_type):
    # as indicator function
    relevance = [0 if i == urgency_type else 1 for i in x.relevance]
    return np.multiply(relevance, column).sum()

# generate relevance weighted features
def generate_relevance_weighted_sentiment(squashedDf):
    # we are removing urgency = 2 type because there are too few of them for learning
    squashedDf = squashedDf[squashedDf["urgency"] != 2]
    
    # for article and alert, let's compute different values
    urgency_ls = [1, 3]
    urgency_name = ["alert", "article"]
    time_ls = ["12H", "24H", "3D", "5D", "7D"]
    for i in range(len(urgency_ls)):
        name = urgency_name[i]
        u = urgency_ls[i]
        squashedDf[name+"_relevance_weighted_sentiment"] = squashedDf.apply(lambda x: urgency_helper(x, x.sentimentClass, u), axis=1)
        squashedDf[name+"_relevance_weighted_negative_sentiment"] = squashedDf.apply(lambda x: urgency_helper(x, x.sentimentNegative, u), axis=1)
        squashedDf[name+"_relevance_weighted_positive_sentiment"] = squashedDf.apply(lambda x: urgency_helper(x, x.sentimentPositive, u), axis=1)
        squashedDf[name+"_relevance_weighted_neutral_sentiment"] = squashedDf.apply(lambda x: urgency_helper(x, x.sentimentNeutral, u), axis=1)
        for time in time_ls:
            squashedDf[name+"_news_volume_sum_"+time] = squashedDf.apply(lambda x: urgency_time_helper(x, x["volumeCounts"+time], u), axis=1)
            squashedDf[name+"_news_novelty_sum_"+time] = squashedDf.apply(lambda x: urgency_time_helper(x, x["noveltyCount"+time], u), axis=1)
    squashedDf["relevance_weighted_sentiment"] = squashedDf.apply(lambda x: np.multiply(x.relevance, x.sentimentClass).sum() / sum(x.relevance), axis=1)
    squashedDf["relevance_weighted_negative_sentiment"] = squashedDf.apply(lambda x: np.multiply(x.relevance, x.sentimentNegative).sum() / sum(x.relevance), axis=1)
    squashedDf["relevance_weighted_positive_sentiment"] = squashedDf.apply(lambda x: np.multiply(x.relevance, x.sentimentPositive).sum() / sum(x.relevance), axis=1)
    squashedDf["relevance_weighted_neutral_sentiment"] = squashedDf.apply(lambda x: np.multiply(x.relevance, x.sentimentNeutral).sum(), axis=1)
    for time in time_ls:
        squashedDf["news_volume_sum_"+time] = squashedDf.apply(lambda x: sum(x["volumeCounts"+time]), axis=1)
        squashedDf["news_novelty_sum_"+time] = squashedDf.apply(lambda x: sum(x["noveltyCount"+time]), axis=1)
    squashedDf["alert"] = squashedDf.urgency.apply(lambda x: urgency_dist_helper(x)[0])
    squashedDf["article"] = squashedDf.urgency.apply(lambda x: urgency_dist_helper(x)[1])
    return squashedDf

def extract_features(df):
    extract_ls = ['month','week', 'alert', 'article',
              
                'alert_relevance_weighted_sentiment',
                'alert_relevance_weighted_negative_sentiment',
                'alert_relevance_weighted_positive_sentiment',
                'alert_relevance_weighted_neutral_sentiment',
                'alert_news_volume_sum_12H', 'alert_news_novelty_sum_12H',
                'alert_news_volume_sum_24H', 'alert_news_novelty_sum_24H',
                'alert_news_volume_sum_3D', 'alert_news_novelty_sum_3D',
                'alert_news_volume_sum_5D', 'alert_news_novelty_sum_5D',
                'alert_news_volume_sum_7D', 'alert_news_novelty_sum_7D',
              
                'article_relevance_weighted_sentiment',
                'article_relevance_weighted_negative_sentiment',
                'article_relevance_weighted_positive_sentiment',
                'article_relevance_weighted_neutral_sentiment',
                'article_news_volume_sum_12H', 'article_news_novelty_sum_12H',
                'article_news_volume_sum_24H', 'article_news_novelty_sum_24H',
                'article_news_volume_sum_3D', 'article_news_novelty_sum_3D',
                'article_news_volume_sum_5D', 'article_news_novelty_sum_5D',
                'article_news_volume_sum_7D', 'article_news_novelty_sum_7D',
              
                'relevance_weighted_sentiment', 'relevance_weighted_negative_sentiment',
                'relevance_weighted_positive_sentiment',
                'relevance_weighted_neutral_sentiment', 'news_volume_sum_12H',
                'news_novelty_sum_12H', 'news_volume_sum_24H', 'news_novelty_sum_24H',
                'news_volume_sum_3D', 'news_novelty_sum_3D', 'news_volume_sum_5D',
                'news_novelty_sum_5D', 'news_volume_sum_7D', 'news_novelty_sum_7D',
              
              'returnsOpenNextMktres10']
    df = df[df.universe == 1.0][extract_ls].dropna()
    return df
    
# orchestration
def main():
    news_df, market_df = extract_df(news, market)
    market_df = extract_time_dependent_features(market_df)
    news_df = extract_time_dependent_features(news_df, "news")
    news_df["assetCode"] = news_df["assetCodes"].apply(assetCodeMapper)
    mergedDF = mergeDframes(news_df, market_df)
    squashedDf = squash(mergedDF)
    featureDf = generate_relevance_weighted_sentiment(squashedDf)
    df = extract_features(featureDf)
    return df
    

# execution
df = main()
