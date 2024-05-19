<p align="left">
<!--   <img src="https://github.com/konnir/x_grammar_spelling/assets/119952960/f415aef0-dd6b-4223-81be-9ce5d677b53a" alt="anyword_logo" width="150" style="margin-left: 50px;"/> -->
  <img src="https://github.com/konnir/x_grammar_spelling/assets/119952960/aaae3161-5d93-4e82-87bf-1ac468f1817a" alt="Nir_Kon_Logo_inverted (1)" width="100"/>
</p>

# Amazon Reviews Stars Ratings: 
Compare ML (TF-IDF + Random Forest) and DL (BertForSequenceClassification) for Amazon Reviews Stars Ratings

- Simple UI base on the rest server to test on real amazon reviews:
- https://amazon-reviews-rater-iu5vx2gsqa-uc.a.run.app

![image](https://github.com/konnir/amazon_reviews_chat_LLM_ner/assets/119952960/ac9c45ce-e6a2-4a3b-b573-051f7c2c324d)

## DataSet:
Amazon Reviews 2018 (Full Dataset) - https://www.kaggle.com/datasets/rogate16/amazon-reviews-2018-full-dataset?resource=download

## Models Review:
- TF-IDF + Random Forest
- BertForSequenceClassification

# Content
1. models:
-  Data exploration (see notebooks/am_data_explore.ipynb)
-  Logistic Regression - Tested, 68% accuracy
-  Random Forest (see model_factory/am_random_forest.ipynb)- Tested, 73% accuracy - SELECTED
-  BertForSequenceClassification, "bert-base-uncased on 10% of the data (see notebooks/bert_predictor.ipynb), 71% accuracy

2. Server:
- Full rest server (see ratings_server.py) - simple rest server on the random forest model

3. Setup:
- requirements (main folder)- based on 3.9, tried to split to base and additional
- Randome Forest Model, take from drive (2.1GB) and set in  (/models) folder
- Bert Model (bert_3.pth), take from drive (438MB) and set in  (/models) folder
- DS, a clean DS with 'revieText' and 'rating' is in the drive, name amazon_reviews_reviewText_ratings.csv, put it in /data
- https://drive.google.com/drive/folders/1MJHU91-3o6TMMg3cLppXnotbb_J7BDwi?usp=drive_link (your email is provisioned)

4. Rest Server (Random Forest + BERT):
- simple Server, below is image to help with the sending and receiving from postman.
- pay attention that req_id is expected in params (to link request to response and debug).

## /predict/random_forest

![image](https://github.com/konnir/velotix_ex/assets/119952960/bb24ec02-9251-43d3-9431-76486f127788)

## /predict/bert

![image](https://github.com/konnir/velotix_ex/assets/119952960/ea605915-709d-478d-983b-5afd7df481e7)

