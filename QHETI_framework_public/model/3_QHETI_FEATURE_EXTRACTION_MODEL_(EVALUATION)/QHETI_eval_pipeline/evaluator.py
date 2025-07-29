import numpy as np

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, y_hat_class, Y):
        """
        Computes classification metrics and confusion matrix.

        Args:
            y_hat_class (np.ndarray): Predicted labels.
            Y (np.ndarray): Ground-truth labels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (metrics vector, 2x2 confusion matrix)
        """
        cm = np.zeros((2, 2), dtype=int)
        y_hat_class = y_hat_class.reshape(-1)

        for y_hat, y in zip(y_hat_class, Y):
            if y == 0:
                if y_hat == 0:
                    cm[0, 0] += 1
                else:
                    cm[0, 1] += 1
            elif y == 1:
                if y_hat == 1:
                    cm[1, 1] += 1
                else:
                    cm[1, 0] += 1

        tp, fn, fp, tn = cm.ravel()
        assert (tp + tn + fp + fn) != 0.0

        wa0 = tp / (2 * (tp + fn)) if (tp + fn) else 0.0
        wa1 = tn / (2 * (tn + fp)) if (tn + fp) else 0.0
        wa = wa0 + wa1

        r = tp / (tp + fn) if (tp + fn) else 0.0
        p0 = tp / (tp + fp) if (tp + fp) else 0.0
        p1 = tn / (tn + fn) if (tn + fn) else 0.0
        pavg = (p0 + p1) / 2.0

        s = tn / (tn + fp) if (tn + fp) else 0.0
        fscore0 = 2 * p0 * r / (p0 + r) if (p0 + r) else 0.0
        fscore1 = 2 * p1 * s / (p1 + s) if (p1 + s) else 0.0
        f1avg = (fscore0 + fscore1) / 2.0

        tpr = tp / (fn + tp) if (fn + tp) else 0.0
        fpr = fp / (tn + fp) if (tn + fp) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        fdr = fp / (tp + fp) if (tp + fp) else 0.0
        fo_rate = fn / (fn + tn) if (fn + tn) else 0.0

        auc_roc = 0.5 * (1 + tpr - fpr)
        j = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

        return (
            np.array([
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
            ]),
            cm,
        )

