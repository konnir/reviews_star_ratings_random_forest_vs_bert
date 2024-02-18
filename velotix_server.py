from flask import Flask, request, jsonify, render_template
import numpy as np

from inference_models.random_forest_inference import RandomForestAmazonModel

# Assuming logistic_regression_model is your pre-loaded model
# from logistic_regression_model import logistic_regression_model

random_forest_model = RandomForestAmazonModel()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict/random_forest', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Initialize an empty dictionary for the ratings
            ratings = {}
            # Iterate through each review ID and review text in the form data
            for review_id, review_text in request.form.items():
                # Assuming your model's inference method takes the review text and returns a rating
                rating = random_forest_model.inference(review_text)
                # Storing the prediction using review_id as key in the ratings dictionary
                ratings[review_id] = rating

            # Capture the req_id from the query string
            req_id = request.args.get('req_id', 'default_req_id')  # Provide a default value if req_id is missing

            # Include the req_id and ratings in the response
            response = {
                "req_id": req_id,
                "ratings": ratings
            }
            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8081)
