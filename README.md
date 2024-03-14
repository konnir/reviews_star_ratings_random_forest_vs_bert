
![Nir_Kon_Logo_inverted (1)](https://github.com/konnir/amazon_reviews_chat_LLM_ner/assets/119952960/3884a586-832a-4e6a-904b-dfcc17b6b027)

# Amazon Reviews Stars Ratings: 
Compare ML (TF-IDF % Random Forest) and DL (Bert)

## DataSet:
Amazon Reviews 2018 (Full Dataset) - https://www.kaggle.com/datasets/rogate16/amazon-reviews-2018-full-dataset?resource=download

## Models Review:
- TF-IDF + Rarndom Forest
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

4. UI:
- Simple UI base on the rest server to test on real amazon reviews

![image](https://github.com/konnir/velotix_ex/assets/119952960/2cca05b3-f8d7-4cd1-9e9b-031c06b4e240)


5. Rest Server (Random Forest + BERT):
- simple Server, below is image to help with the sending and receiving from postman.
- pay attention that req_id is expected in params (to link request to response and debug).

## /predict/random_forest

![image](https://github.com/konnir/velotix_ex/assets/119952960/bb24ec02-9251-43d3-9431-76486f127788)

## /predict/bert

![image](https://github.com/konnir/velotix_ex/assets/119952960/ea605915-709d-478d-983b-5afd7df481e7)

