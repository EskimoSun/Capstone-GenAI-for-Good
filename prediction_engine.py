import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from collections import Counter

class PredictionEngine:
    def __init__(self):
        # Initialize class attributes
        self.tokenizer = None
        self.model = None
        self.device = None
        self.models_dict = {}
        self.pca_dict = {}
        self.party_distribution = None
        self.datasets = {}
        self.train_embeddings = None
        self.test_embeddings = None
        
        # Constants
        self.factor_scores = [
            'location_score', 'education_score', 'event_coverage_score',
            'echo_chamber_score', 'news_coverage_score', 'malicious_account_score'
        ]
        self.count_columns = [
            "barely_true_counts", "false_counts", "half_true_counts",
            "mostly_true_counts", "pants_on_fire_counts"
        ]
        
        # Initialize NLP model
        self.nlp = spacy.load("en_core_web_sm")

    def calculate_location_score(self, statement):
        doc = self.nlp(statement)
        location_tokens = sum([len(ent) for ent in doc.ents if ent.label_ in ["GPE", "LOC"]])
        total_tokens = len(doc)
        return location_tokens / total_tokens if total_tokens > 0 else 0

    def calculate_education_score(self, row):
        total_counts = row[['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']].sum()
        return row['mostly_true_counts'] / total_counts if total_counts > 0 else 0

    def calculate_event_coverage(self, statement):
        event_keywords = ['conference', 'summit', 'meeting', 'election', 'protest', 'war', 'tournament', 'concert', 'festival']
        words = statement.lower().split()
        event_count = sum(word in event_keywords for word in words)
        return event_count / len(words) if words else 0

    def calculate_echo_chamber(self, party_affiliation):
        return 1 - self.party_distribution.get(party_affiliation.lower(), 0)

    def calculate_news_coverage(self, subjects):
        from collections import Counter
        if not subjects:
            return 0
        subject_list = subjects.split(',')
        count = Counter(subject_list)
        n = len(subject_list)
        sum_of_squares = sum(freq**2 for freq in count.values())
        diversity_index = 1 - (sum_of_squares / (n * n))
        return diversity_index

    def calculate_malicious_account(self, row):
        total_counts = row[['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']].sum()
        return row['pants_on_fire_counts'] / total_counts if total_counts > 0 else 0

    def parallel_apply_scores(self, df):
        df['location_score'] = df['statement'].apply(self.calculate_location_score)
        df['education_score'] = df.apply(self.calculate_education_score, axis=1)
        df['event_coverage_score'] = df['statement'].apply(self.calculate_event_coverage)
        df['echo_chamber_score'] = df['party_affiliation'].apply(
            lambda x: self.calculate_echo_chamber(x)
        )
        df['news_coverage_score'] = df['subjects'].apply(self.calculate_news_coverage)
        df['malicious_account_score'] = df.apply(self.calculate_malicious_account, axis=1)
        return df

    def load_dataset_and_prepare_models(self):
        print("Loading dataset...")
        scores = []
        
        # Load datasets
        for dtype in ['train', 'val', 'test']:
            df = pd.read_csv(f"data/{dtype}2.tsv", sep="\t", 
                           header=None, dtype=str).drop(columns=[0])
            df.columns = ["ID", "label", "statement", "subjects", "speaker", "speaker_job_title", "state_info", "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context", "justification"]
            df[self.count_columns] = df[self.count_columns].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
            df.dropna(inplace=True)
            self.datasets[dtype] = df
        
        party_counts = self.datasets['train']['party_affiliation'].value_counts(normalize=True)
        self.party_distribution = party_counts.to_dict()

        # Calculate scores for each dataset
        for dtype, data in self.datasets.items():
            print(f"Calculating scores for {dtype} data...")
            self.datasets[dtype] = self.parallel_apply_scores(data)

        # Export datasets to TSV files with all features
        for dtype, data in self.datasets.items():
            data.to_csv(f"PredictiveAI/{dtype}_data_full.tsv", sep='\t', index=False)
            print(f"{dtype.capitalize()} dataset saved to {dtype}_data_full.tsv.")

        # Calculate and save average scores to a TSV file
        for factor in self.factor_scores:
            for dtype, data in self.datasets.items():
                average_score = data[factor].mean()
                scores.append([factor, dtype, average_score])

        scores_df = pd.DataFrame(scores, columns=['factor', 'source', 'score'])
        scores_df.to_csv('PredictiveAI/average_scores.tsv', sep='\t', index=False)
        print("Average scores saved to 'average_scores.tsv'.")
        
        # Initialize BERT models
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Train models
        self.train_and_prepare_models()

    def get_bert_embeddings(self, statements):
        """
        Extracts BERT embeddings for a list of statements.
        """ 
        self.model.to(self.device)
        self.model.eval()
        embeddings = []
        batch_size = 32
        with torch.no_grad():
            for i in tqdm(range(0, len(statements), batch_size), desc="Extracting BERT embeddings"):
                batch_statements = statements[i:i+batch_size]
                encoded_input = self.tokenizer(batch_statements, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
                outputs = self.model(**encoded_input)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

    def train_model(self, target_score='overall'):
        """
        Train model for either overall veracity or individual factor scores
        target_score: 'overall' or one of the factor scores
        """

        # Select target variable based on what we're predicting
        if target_score == 'overall':
            y_train = self.datasets['train']['label'].values
            y_test = self.datasets['test']['label'].values
            # Use classifier for overall prediction (discrete classes)
            rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, 
                                            class_weight='balanced', random_state=42)
        else:
            # Use regressor for factor scores (continuous values)
            y_train = self.datasets['train'][target_score].values
            y_test = self.datasets['test'][target_score].values
            rf_model = RandomForestRegressor(n_estimators=150, max_depth=10, 
                                           random_state=42)

        # Dimensionality reduction
        pca = PCA(n_components=0.95)
        
        # Include all factor scores except the target one when predicting individual scores
        if target_score == 'overall':
            additional_features = self.datasets['train'][self.factor_scores].values
            test_additional_features = self.datasets['test'][self.factor_scores].values
        else:
            other_scores = [f for f in self.factor_scores if f != target_score]
            additional_features = self.datasets['train'][other_scores].values
            test_additional_features = self.datasets['test'][other_scores].values

        X_train = pca.fit_transform(np.hstack([self.train_embeddings, additional_features]))
        X_test = pca.transform(np.hstack([self.test_embeddings, test_additional_features]))

        # Train model
        rf_model.fit(X_train, y_train)

        # Make predictions and evaluate
        y_test_pred = rf_model.predict(X_test)
        if target_score == 'overall':
            test_accuracy = accuracy_score(y_test, y_test_pred)
            print(f"Test Accuracy for {target_score}: {test_accuracy:.4f}")
            print(classification_report(y_test, y_test_pred))
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_test, y_test_pred)
            r2 = r2_score(y_test, y_test_pred)
            print(f"MSE for {target_score}: {mse:.4f}")
            print(f"R2 score for {target_score}: {r2:.4f}")
            
        return rf_model, pca

    def train_and_prepare_models(self):
        # Calculate party distribution
        party_counts = self.datasets['train']['party_affiliation'].value_counts(normalize=True)
        self.party_distribution = party_counts.to_dict()
        # Process embeddings for training and testing
        self.train_embeddings = self.get_bert_embeddings(self.datasets['train']['statement'].tolist())
        self.test_embeddings = self.get_bert_embeddings(self.datasets['test']['statement'].tolist())
        
        # Train models for each factor
        # for factor in self.factor_scores + ['overall']:
        print(f"\nTraining model for {'overall'}...")
        rf_model, pca = self.train_model('overall')
        self.models_dict['overall'] = rf_model
        self.pca_dict['overall'] = pca

    def process_new_datapoint(self, df):
        # Ensure count columns are numeric
        df[self.count_columns] = df[self.count_columns].apply(
            pd.to_numeric, errors='coerce'
        ).fillna(0).astype(int)
        
        return self.parallel_apply_scores(df)

    def predict_scores(self, df):
        """
        Predict all factor scores and overall veracity for new datapoints
        """
        # First process the datapoint to get factor scores
        processed_df = self.process_new_datapoint(df)
        
        # Get BERT embeddings
        embeddings = self.get_bert_embeddings(processed_df['statement'].tolist())
        
        predictions = {}
        
        # Predict each factor score
        # for factor in self.factor_scores:
        #     # Get other scores as features
        #     other_scores = [f for f in self.factor_scores if f != factor]
        #     additional_features = processed_df[other_scores].values
            
            # Transform features using stored PCA
        #     X = self.pca_dict[factor].transform(np.hstack([embeddings, additional_features]))
            
            # Make prediction
        #     predictions[factor] = self.models_dict[factor].predict(X)
        
        # Predict overall veracity
        additional_features = processed_df[self.factor_scores].values
        X = self.pca_dict['overall'].transform(np.hstack([embeddings, additional_features]))
        predictions['overall'] = self.models_dict['overall'].predict(X)

        return predictions

    def predict_new_example(self, new_data):
        """
        Predict scores for a new example, new_data is a pandas Series of length 15
        """
        if isinstance(new_data, pd.Series):
            new_data = pd.DataFrame([new_data])
            new_data.columns = ["ID", "label", "statement", "subjects", "speaker", "speaker_job_title", "state_info", "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context", "justification"]
        
        predictions = self.predict_scores(new_data)
        return pd.DataFrame(predictions)