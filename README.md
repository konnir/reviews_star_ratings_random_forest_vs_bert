![image](https://github.com/konnir/velotix_ex/assets/119952960/e4c6bb61-80ba-48a4-8b8c-4b8ab8818c74)

# velotix_ex

Assignment Outline:
_Data Analysis:
Download Amazon Reviews dataset.
Perform data exploration, cleaning, and preparation.

_Model Development:
Build and compare at least two predictive models.
Select and justify the best performing model.

_Application Development:
Create a Flask web application utilizing your model.
The application should recommend the top 5 products based on predicted ratings.

_Deliverables:
A detailed report of your analysis and model development.
Source code for the model and the Flask application.
Documentation for application setup and usage.

# Submission:

1. models:
-  Data exploration (see notebooks/am_data_explore.ipynb)
-  Logistic Regression - Tested, 68% accuracy
-  Random Forest (see model_factory/am_random_forest.ipynb)- Tested, 73% accuracy - SELECTED
-  BertForSequenceClassification, "bert-base-uncased on 10% of the data (see notebooks/bert_predictor.ipynb), 71% accuracy

2. Server:
- Full rest server (see velotix_server.py) - simple rest server on the random forest model

3. Setup:
- requirements (main folder)- based on 3.9, tried to split to base and additional
- Model, take from drive (2.1GB) and set in  (/models) folder - https://drive.google.com/drive/folders/1MJHU91-3o6TMMg3cLppXnotbb_J7BDwi?usp=drive_link (your email is provisioned)

4. UI:
- Simple UI base on the rest server to test on real amazon reviews
  
![image](https://github.com/konnir/velotix_ex/assets/119952960/789c614e-0bac-48a4-83a7-cd48ad4bde95)

5. Rest Server:
- simple Server, below is image to help with the sending and receiving from postman.
- pay attention that req_id is expected in params (to link request to response and debug).
  
![image](https://github.com/konnir/velotix_ex/assets/119952960/bb24ec02-9251-43d3-9431-76486f127788)


