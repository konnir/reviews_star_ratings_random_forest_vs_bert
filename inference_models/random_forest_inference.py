
import joblib  # For saving and loading the model

from inference_models import inference_utils


class RandomForestAmazonModel:
    def __init__(self, model_path: str = '/home/user/IdeaProjects/velotix_ex/models/random_forest_review_model.joblib'):
        from joblib import load
        model_filename = model_path
        self.random_forest_model = joblib.load(model_filename)

    def inference(self, review_text: str) -> int:
        """ Predict the rating of a review based on the model. """
        review_text_clean = inference_utils.review_text_to_clean(review_text)

        # Make prediction
        reviews = [review_text_clean]
        predictions = self.random_forest_model.predict(reviews)
        return int(predictions[0])

# sanity
# if __name__ == '__main__':




