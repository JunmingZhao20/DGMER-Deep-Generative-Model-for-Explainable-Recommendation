# DGMER-----Deep Generative Model for Explainable Recommendation





[TOC]

## background

Given user and item , the explainable recommendation task aims at predicting a rating which indicates user's preference toward item and generating a reasonable explanation. In DGMER, we use the reviews from douban website to generate pseudo explanation and complete an unsupervised learning task for explanation generation. Based on GPT2, the model will make the next-token prediction based on preceding tokens in an auto-regressive method.







## Usage
```python
python -u improved.py \
--data_path Dataset/reviews_s.csv \
--cuda \
--checkpoint ./douban/ >> douban.log
```







## Dataset

[Moviedata-10M](http://moviedata.csuldw.com/)

we cleaned it and generated pseudo explanation from its review. The processed data is in the Dataset/reviews_s.csv. The process program is  select.py





## Code dependencis

python 3.9

pytorch 1.12







## Code reference

[Personalized Transformer for Explainable Recommendation](https://github.com/lileipisces/PETER)

[Personalized Prompt Learning for Explainable Recommendation](https://github.com/lileipisces/PEPLER)

## 

