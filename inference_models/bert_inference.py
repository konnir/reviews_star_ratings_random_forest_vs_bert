import numpy as np
import pandas as pd

from inference_models import inference_utils
import torch
from transformers import AutoTokenizer, BertModel, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader


class BertSentimentModel:
    def __init__(self, model_path: str = 'models/Model_full_data/2/'):
        self.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device {self.device}')
        # self.model = BertForSequenceClassification.from_pretrained(
        #     "bert-base-uncased",
        #     num_labels = 5, # Number of unique labels for our multi-class classification problem.
        #     output_attentions = False,
        #     output_hidden_states = False,
        # )
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=5,  # Make sure this matches the original number of labels
            output_attentions=False,
            output_hidden_states=False,
        )
        self.model.to(device)

        # Load the trained model.
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()


    def inference(self, review_text: str) -> int:
        """ Predict the rating of a review based on the model. """
        review_text_clean = inference_utils.review_text_to_clean(review_text)

        # Make prediction
        predictions = self.get_single_prediction(review_text_clean)
        return int(predictions)


    def get_single_prediction(self, comment):
        """
        Predict a star rating from a review comment.
        :comment: the string containing the review comment.
        :model: the model to be used for the prediction.
        """

        df = pd.DataFrame()
        df['reviewText'] = [comment]
        df['rating'] = ['0']

        dataset = ReviewsDataset(df)

        TEST_BATCH_SIZE = 1
        NUM_WORKERS = 1

        test_params = {'batch_size': TEST_BATCH_SIZE,
                       'shuffle': True,
                       'num_workers': NUM_WORKERS}

        data_loader = DataLoader(dataset, **test_params)

        total_examples = len(df)
        predictions = np.zeros([total_examples], dtype=object)

        for batch, data in enumerate(data_loader):

            # Get the tokenization values.
            input_ids = data['input_ids'].to(self.device)
            mask = data['attn_mask'].to(self.device)

            # Make the prediction with the trained model.
            outputs = self.model(input_ids, mask)

            # Get the star rating.
            big_val, big_idx = torch.max(outputs[0].data, dim=1)
            star_predictions = (big_idx + 1).cpu().numpy()

        return star_predictions[0]

class ReviewsDataset(Dataset):
    def __init__(self, df, max_length=512):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # input=review, label=stars
        review = self.df.loc[idx, 'reviewText']
        # labels are 0-indexed
        label = int(self.df.loc[idx, 'rating']) - 1

        encoded = self.tokenizer(
            review,                      # Review to encode.
            add_special_tokens=True,
            max_length=self.max_length,  # Truncate all segments to max_length.
            padding='max_length',        # Pad all reviews with the [PAD] token to the max_length.
            return_attention_mask=True,  # Construct attention masks.
            truncation=True
        )

        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids),
            'attn_mask': torch.tensor(attn_mask),
            'label': torch.tensor(label)
        }

# sanity
if __name__ == '__main__':
    handler = BertSentimentModel()
    prediction = handler.inference("Purchase this based on some Amazon recommendations, and I was a little disappointed. I found the product had, reasonable hold, but deteriorated to be very pasty after a short period of time.")
    print(prediction)

