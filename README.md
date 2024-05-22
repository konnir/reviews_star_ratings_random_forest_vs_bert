<p align="left">
<!--   <img src="https://github.com/konnir/x_grammar_spelling/assets/119952960/f415aef0-dd6b-4223-81be-9ce5d677b53a" alt="anyword_logo" width="150" style="margin-left: 50px;"/> -->
  <img src="https://github.com/konnir/x_grammar_spelling/assets/119952960/aaae3161-5d93-4e82-87bf-1ac468f1817a" alt="Nir_Kon_Logo_inverted (1)" width="100"/>
</p>

# Reviews Stars Ratings - Random Forest Vs Bert: 
Fine tuning 50K star reviews (1-5) on ML (TF-IDF + Random Forest) vs DL (BertForSequenceClassification) to compare results.

- Demo UI: [http://review.nir-kon.com](https://loading-proxy-iu5vx2gsqa-uc.a.run.app/get?url=https://reviews-rater-iu5vx2gsqa-uc.a.run.app/)
- <a href="https://loading-proxy-iu5vx2gsqa-uc.a.run.app/get?url=https://x-grammar-spelling-cpu-gcp-iu5vx2gsqa-uc.a.run.app/" target="_blank">http://review.nir-kon.com</a>
- It's a sleeping machine that load 2GB Random Forest model and another 0.5 GB of Bert model - Allow 2-3 minuites to wake up!!!
- Allow few seconds for first hit (it's working)!

![image](https://github.com/konnir/reviews_star_ratings_random_forest_vs_bert/assets/119952960/c7ccbae0-3bfc-4634-87e5-3808abca5ed3)

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

4. Rest Server (Random Forest + BERT):
- simple Server, below is image to help with the sending and receiving from postman.
- pay attention that req_id is expected in params (to link request to response and debug).

Licence:

- All work done by the author is free to use without any warranty.
- Any third parties code, models, data sets and others belong to their authors and legal entities, it's user responsibility to check and get any needed approval.
- The purpose of this code and demo is for reaserch only.
