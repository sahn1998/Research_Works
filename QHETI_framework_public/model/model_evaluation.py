import numpy as np

class Evaluation:
    def __init__(self, nn_model):
        """
        Initializes the PopulationEvaluation class with a trained model.

        Parameters:
        nn_pop (object): A trained model that has a .predict(X) method.
        """
        self.nn_model = nn_model

    def model_predict(self, X):
        """
        Predicts output using the trained model.

        Parameters:
        X (np.ndarray): Input features.

        Returns:
        np.ndarray: Predicted values.
        """
        return self.nn_model.predict(X)

    def model_test(self, X_test, Y_test):
        """
        Generates class predictions and evaluates them using the evaluate method.

        Parameters:
        X_test (np.ndarray): Test features.
        Y_test (np.ndarray): True class labels.

        Returns:
        tuple: (evaluation_metrics, confusion_matrix)
        """
        # print("X_test", self.X_test)
        # print("Y_test", self.Y_test)
        y_hat = self.model_predict(X_test)
        y_hat_class = (y_hat + 0.5).astype(int)
        # y_hat_class = np.where(y_hat.cpu()<0.5, 0, 1)
        # print("y_hat", self.y_hat)
        # print("y_hat_class", self.y_hat_class)
        # print("Y_test", self.Y_test)
        return self.evaluate(y_hat_class, Y_test)

    def evaluate(self, y_hat_class, Y):  # Y = real value, Y_hat = expected
        """
        Evaluates prediction performance and computes various classification metrics.

        Parameters:
        y_hat_class (np.ndarray): Predicted class labels.
        Y (np.ndarray): True class labels.

        Returns:
        tuple: (metrics array, confusion matrix)
        """
        cm = np.zeros(
            (2, 2), dtype=int
        )  # Initialize the confusion matrix as a 2x2 matrix of zeros
        y_hat_class = y_hat_class.reshape(-1)  # Ensure y_hat_class is a 1D array
        
        # Calculate
        # True Positives (TP) - T0, 
        # True Negatives (TN) - T1, 
        # False Positives (FP) - F0, 
        # False Negatives (FN) - F1

        for y_hat, y in zip(y_hat_class, Y):
            if y == 0:  # Actual class is 0 (positive)
                if y_hat == 0:
                    cm[0, 0] += 1  # True Positive (TP)
                else:
                    cm[0, 1] += 1  # False Negative (FN)
            elif y == 1:  # Actual class is 1 (negative)
                if y_hat == 1:
                    cm[1, 1] += 1  # True Negative (TN)
                else:
                    cm[1, 0] += 1  # False Positive (FP)
        tp, fn, fp, tn = cm.ravel()
        assert (tp + tn + fp + fn) != 0.0

        wa1 = (
            (tn / (2 * (tn + fp))) if (tn + fp) != 0.0 else 0.0
        )  ## Local Var - Weighted_Accuracy_class1
        wa0 = (
            (tp / (2 * (tp + fn))) if (tp + fn) != 0.0 else 0.0
        )  ## Local Var - Weighted_Accuracy_class0
        wa = wa0 + wa1  ## Weighted Accuracy


        r = tp / (tp + fn) if (tp + fn) != 0.0 else 0.0  ## Sensitivity/Recall

        p1 = (
            tn / (tn + fn) if (tn + fn) != 0.0 else 0.0
        )  ## Precision_class0 # negative predictive value
        p0 = (
            tp / (tp + fp) if (tp + fp) != 0.0 else 0.0
        )  ## Precision_class1 # positive predictive value

        s = tn / (tn + fp) if (tn + fp) != 0.0 else 0.0  ## Specificity

        fscore0 = 2 * p0 * r / (p0 + r) if p0 + r != 0.0 else 0.0  ## F1_class0
        fscore1 = 2 * p1 * s / (p1 + s) if p1 + s != 0.0 else 0.0  ## F1_class1

        # d = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) != 0.0 else 0.0     ## Removed - Accuracy
        j = tp / (tp + fp + fn) if (tp + fp + fn) != 0.0 else 0.0  ## Jaccard
        tpr = (
            tp / (fn + tp) if (fn + tp) != 0 else 0.0

        )  ## True Positive Rate - Local Var
        fpr = fp / (tn + fp) if (tn + fp) != 0 else 0.0  ## False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0  ## True Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0  ## False Negative Rate
        fdr = fp / (tp + fp) if (tp + fp) != 0 else 0.0  ## False Discovery Rate
        fo_rate = fn / (fn + tn) if (fn + tn) != 0 else 0.0  ## False Omission Rate
        # Formula for AUC_Roc score without using function - https://stackoverflow.com/questions/50848163/manually-calculate-auc
        auc_roc = (1 / 2) - (fpr / 2) + (tpr / 2)  ## auc_roc score
        pavg = (p0 + p1) / 2.0  ## Precision_avg
        f1avg = (fscore0 + fscore1) / 2.0  ## F1_avg
        return (
            np.array(
                [
                    # tp,
                    # fn,
                    # fp,
                    # tn,
                    wa,
                    r,
                    s,
                    p0,
                    p1,
                    pavg,
                    fscore0,
                    fscore1,
                    f1avg,
                    auc_roc,
                    fpr,
                    fdr,
                    fnr,
                    fo_rate,
                    j,
                ]
            ),
            cm,
        )