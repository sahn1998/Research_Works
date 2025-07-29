import gc
import os
import psutil
import traceback
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model

from QHETI_eval_pipeline.feature_extractor import FeatureExtractor
from QHETI_eval_pipeline.classifier_trainer import ClassifierTrainer

class ModelProcessor:
    """
    Proprietary class for processing models:
    - Loads deep learning models
    - Extracts features
    - Trains conventional classifiers
    - Evaluates performance across K-fold splits
    """

    def __init__(self, batch_size, num_splits, evaluation_metrics, evaluate_fn):
        self.batch_size = batch_size
        self.num_splits = num_splits
        self.evaluation_metrics = evaluation_metrics
        self.evaluate = evaluate_fn

    def process(self, target_id, model_filename, model_filepath, model_type, dataset_list):
        try:
            # --- Step 1: Load pretrained deep learning model ---
            model = load_model(model_filepath, compile=False)
            feature_extractor = FeatureExtractor(
                feature_extractor_model=Model(inputs=model.input, outputs=model.layers[-2].output),
                batch_size=self.batch_size
            )

            kfold_results, kfold_confusion_matrices = [], []

            # --- Step 2: K-fold evaluation loop (details hidden) ---
            for i in range(self.num_splits):
                # Get train/test sets
                X_train, X_test, Y_train, Y_test = dataset_list[i]

                # Extract deep features
                X_train_feat = feature_extractor.extract(X_train)
                classifier = ClassifierTrainer(model_type).train(X_train_feat, Y_train)

                # Test set features + prediction
                X_test_feat = feature_extractor.extract(X_test)
                y_pred = classifier.predict(X_test_feat)

                # Proprietary evaluation logic
                results, matrix = self.evaluate(y_pred.reshape(-1, 1), Y_test)
                kfold_results.append(results)
                kfold_confusion_matrices.append(matrix)

                # Memory management
                mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                print(f"Memory usage after split {i+1}: {mem:.2f} MB")
                K.clear_session()
                gc.collect()

            # --- Step 3: Average results across folds ---
            version_tag = f"{model_filename[:-3]}_{model_type}"
            avg_results = (
                pd.DataFrame(kfold_results, columns=self.evaluation_metrics)
                .mean()
                .values
                .reshape(1, -1)
            )
            avg_results_df = pd.DataFrame(avg_results, columns=self.evaluation_metrics)
            avg_results_df.insert(0, "version_tag", version_tag)

            # Clear memory
            K.clear_session()
            gc.collect()
            return version_tag, avg_results_df

        except Exception as e:
            # Proprietary exception handling
            print(f"❌ Error processing {model_filename}: {e}")
            print(traceback.format_exc())
            return None, None
