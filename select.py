import pandas
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
import pymysql
from tqdm import tqdm
import numpy as np
import random
import re
from snownlp import SnowNLP
import hashlib

"""
step1：Limit the number of users to 1000, extract the information from their watch list, and obtain the adjacency matrix social network representation.
"""

db=pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='Password123!',
)
cursor=db.cursor()
cursor.execute("use douban")
cursor.execute("select name, following_id, rates from user")
users=cursor.fetchall()
cursor.execute("select id from movie")
movies=cursor.fetchall()
cursor.close()
user_name=[0 for _ in range(1000)]
movie_id=[0 for _ in range(1000)]
label=np.zeros((1000,1000))
social_network=np.zeros((1000,1000))
userItem=np.zeros((1000,1000))
for n,i in enumerate(movies):
    movie_id[n]=i[0]
for n,i in enumerate(users):
    user_name[n]=i[0]
for n,i in enumerate(users):
    for following in eval(i[1]):
        if following in user_name:
            social_network[n][user_name.index(following)]=1
    rate=eval(i[2])
    for movie in rate:
        if movie in movie_id:
            userItem[n][movie_id.index(movie)]=rate[movie]


np.save("social_network.npy",social_network)
np.save("user_item_with_social.npy",userItem)


"""
step2：Record the user's ratings for each movie in the matrix, divide the training and test data
"""

# reviews=pd.read_csv("reviews_item_based.csv")
# userID=[]
# itemID=[]
# userIndex={}
# itemIndex={}
# j=0
# k=0
# for i in range(len(reviews)):
#     if reviews["USER_MD5"][i] not in userIndex:
#         userIndex[reviews["USER_MD5"][i]]=j
#         j+=1
#         userID.append(reviews["USER_MD5"][i])
#     if reviews["MOVIE_ID"][i] not in itemIndex:
#         itemIndex[reviews["MOVIE_ID"][i]]=k
#         k+=1
#         itemID.append(reviews["MOVIE_ID"][i])
# table=np.zeros((j,k))
# label=np.zeros((j,k)) # 建立label矩阵，确定该处的记录是训练集还是测试集。
# for i in range(len(reviews)):
#     user=userIndex[reviews["USER_MD5"][i]]
#     item=itemIndex[reviews["MOVIE_ID"][i]]
#     table[user][item]=reviews["RATING"][i]
#     p=random.random()
#     # if p<0.8:
#     label[user][item]=1  # 1表示训练集
#     else:
#         label[user][item]=2  # 2表示测试集，0表示此处没有记录
# np.save("item_user.npy",table)
# np.save("label_item_full.npy",label)
# file=open("user_index_item_based.pkl",'wb')
# pickle.dump(userID,file)
# file.close()
# file=open("item_index_item_based.pkl",'wb')
# pickle.dump(itemID,file)
# file.close()

"""
step3：Delete some movie entries that have not yet been released and have incomplete records, and delete entries that do not contain ratings in the review table, and combine the two to provide semantic information. To deal with memory allocation, the csv file is read, processed and saved in segments.
"""

# df_tmp = []
# select_cols1 = ["MOVIE_ID", "DIRECTORS", "GENRES", "TAGS", "DOUBAN_SCORE", "YEAR"]
# select_cols3 = ["USER_MD5", "MOVIE_ID", "CONTENT", "RATING","VOTES"]
# dtypes1={"DOUBAN_SCORE":np.float16,"YEAR":np.int16}
# dtypes2={"RATING":np.float16,"VOTES":np.int16}
# movies = pandas.read_csv("movies_clean.csv",usecols=select_cols1,dtype=dtypes1)
# movie_id=list(movies['MOVIE_ID'])
# for chunk in tqdm(pandas.read_csv("comments.csv",chunksize=200000,usecols=select_cols3,dtype=dtypes2)):
#     chunk=chunk[chunk['MOVIE_ID'].isin(movie_id) & chunk['RATING'].notnull()]
#     movie_review = pandas.merge(left=chunk, right=movies, on=['MOVIE_ID'])
#     df_tmp.append(movie_review)
#     del chunk
#     del movie_review
# movie_review=pd.DataFrame(index=["0"],dtype=np.float32)
# for i in range(len(df_tmp)):
#     tmp=pandas.DataFrame(data=df_tmp[0])
#     movie_review=pandas.concat([movie_review,tmp],ignore_index=True)
#     del df_tmp[0],tmp
# movie_review.to_csv("reviews_clean.csv")


"""
step4: generate pseudo explanation, Screening of highly relevant sentences in reviews, based on sentiment analysis and relevance to movie topics
"""

# reviews=pandas.read_csv("reviews_selected.csv")
# movies=pandas.read_csv("movies.csv")
#
# reviews["SUMMARY"]=""
# reviews["SELECTED"]=""
# old_movie=None
# for i in tqdm(range(len(reviews))):
#     new_movie=reviews["MOVIE_ID"][i]
#     s=SnowNLP(reviews["CONTENT"][i])
#     sent = s.sentences
#     if len(sent)==0:
#         continue
#     summ = s.summary(3)
#     score=reviews["RATING"][i]
#     if not new_movie == old_movie:
#         brief = movies[movies["MOVIE_ID"].isin([new_movie])]["STORYLINE"]
#         if not pandas.isna(brief.values[0]):
#             b=SnowNLP(brief.values[0])
#             bsplit=b.words
#     selected=[(float("-inf"),'') for i in range(3)]
#     for sentence in sent:
#         temp=SnowNLP(sentence)
#         sen=temp.sentiments
#         if not pandas.isna(brief.values[0]):
#             sim=SnowNLP([temp.words]).sim(bsplit)[0]
#         else:
#             sim=0
#         value=-abs(5*sen-score)+0.1*abs(sim)
#         for j in range(3):
#             if selected[j][0] < value:
#                 selected.pop()
#                 selected.insert(j, (value, sentence))
#                 break
#     choice=[]
#     for j in selected:
#         if j[0]>float("-inf"):
#             choice.append(j[1])
#     reviews["SUMMARY"][i] = " ".join(summ)
#     reviews["SELECTED"][i] = " ".join(choice)
#     old_movie=new_movie
# reviews.to_csv("reviews_s.csv")


