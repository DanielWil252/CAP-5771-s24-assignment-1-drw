# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c

# from sklearn.base import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from part_1_template_solution import Section1 as Part1
from part_2_template_solution import Section2 as Part2

# Fill in the appropriate import statements from sklearn to solve the homework
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, top_k_accuracy_score

# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold  # , KFold
from sklearn.svm import SVC  # , LinearSVC

# from collections import defaultdict

# import logistic regresssion module
# from sklearn.linear_model import LogisticRegression
import numpy as np
from numpy.typing import NDArray
from typing import Any

import new_utils as nu

# import matplotlib.pyplot as plt
# import warnings

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize
        self.part1 = Part1(seed=seed, frac_train=frac_train)
        self.part2 = Part2(seed=seed, frac_train=frac_train)

    def remove_nines_convert_to_01(self, X, y, frac):
        """
        frac: fraction of 9s to remove
        """
        # Count the number of 9s in the array
        num_nines = np.sum(y == 9)

        # Calculate the number of 9s to remove (90% of the total number of 9s)
        num_nines_to_remove = int(frac * num_nines)

        # Identifying indices of 9s in y
        indices_of_nines = np.where(y == 9)[0]

        # Randomly selecting 30% of these indices
        num_nines_to_remove = int(np.ceil(len(indices_of_nines) * frac))
        indices_to_remove = np.random.choice(
            indices_of_nines, size=num_nines_to_remove, replace=False
        )

        # Removing the selected indices from X and y
        X = np.delete(X, indices_to_remove, axis=0)
        y = np.delete(y, indices_to_remove)

        y[y == 7] = 0
        y[y == 9] = 1
        return X, y

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        return {
            "class_counts": {},  # Replace with actual class counts
            "num_classes": 0,  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        Xtest: NDArray[np.int32],
        ytest: NDArray[np.int32],
    ):
        """ """
        answer = {}

        counts_train = np.unique(y, return_counts=True)
        counts_test = np.unique(ytest, return_counts=True)
        print("===> counts_train: ", counts_train)
        print("===> counts_test: ", counts_test)
        print()

        self.is_int = nu.check_labels(y)
        self.is_int = nu.check_labels(y)
        self.dist_dict = self.analyze_class_distribution(
            y.astype(np.int32)
        )  # Convert y to NDArray[int32]

        clf = RandomForestClassifier(random_state=self.seed)
        clf.fit(X, y)
        ytrain_pred = clf.predict_proba(X)
        ytest_pred = clf.predict_proba(Xtest)
        # nb_unique, counts = np.unique(y, return_counts=True)
        # nb_unique, counts = np.unique(ytest, return_counts=True)
        # nb_unique, counts = np.unique(ytrain_pred, return_counts=True)
        # nb_unique, counts = np.unique(ytest_pred, return_counts=True)

        # top-k accuracy score
        # Return the fraction of correctly classified samples (float), i.e. the accuracy.
        topk = [k for k in range(1, 6)]
        plot_scores_test = []
        plot_scores_train = []
        for k in topk:
            topk_dict = {}
            nb_unique, counts = np.unique(ytest, return_counts=True)
            score_train = top_k_accuracy_score(y, ytrain_pred, normalize=True, k=k)
            score_test = top_k_accuracy_score(ytest, ytest_pred, normalize=True, k=k)
            # What does normalize argument do?
            topk_dict["score_train"] = score_train
            topk_dict["score_test"] = score_test
            answer[k] = topk_dict
            plot_scores_test.append((k, score_test))
            plot_scores_train.append((k, score_train))

        answer["clf"] = clf
        # Plot k vs. score
        # (1, score1), (2, score2), ...
        answer["plot_k_vs_score_train"] = plot_scores_train
        answer["plot_k_vs_score_test"] = plot_scores_test

        # Keys with "text_" require an textual explanation with double or triple quote delimiters
        answer["text_rate_accuracy_change"] = None
        answer["text_is_topk_useful_and_why"] = None

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    # How to make sure the seed propagates. Perhaps specify in the class constructor.
    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        Xtest: NDArray[np.int32],
        ytest: NDArray[np.int32],
    ):
        # Only Keep 7 and 9's
        seven_nine_idx = (y == 7) | (y == 9)
        X = X[seven_nine_idx, :]
        y = y[seven_nine_idx]
        frac_to_remove = 0.8
        X, y = self.remove_nines_convert_to_01(X, y, frac=frac_to_remove)
        Xtest, ytest = self.remove_nines_convert_to_01(
            Xtest, ytest, frac=frac_to_remove
        )
        answer = {}
        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ):
        n_splits = 5
        clf = SVC(random_state=self.seed)
        # Shuffling is fine because of seed
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        # score = ["accuracy", "recall", "precision", "f1"]

        def cross_validate_metric(score_type: str):
            score = ["accuracy", "recall", "precision", "f1"]
            """
            scoring = {
                "accuracy": make_scorer(accuracy_score),
                "recall": make_scorer(recall_score, average=score_type),
                "precision": make_scorer(precision_score, average=score_type),
                "f1": make_scorer(f1_score, average=score_type),
            }
            print("===> scoring: ", scoring)
            """
            cv_scores = cross_validate(
                clf, X, y, scoring=score, cv=cv, return_train_score=False
            )

            scores = {
                "mean_accuracy": cv_scores["test_accuracy"].mean(),
                "mean_recall": cv_scores["test_recall"].mean(),
                "mean_precision": cv_scores["test_precision"].mean(),
                "mean_f1": cv_scores["test_f1"].mean(),
                "std_accuracy": cv_scores["test_accuracy"].std(),
                "std_recall": cv_scores["test_recall"].std(),
                "std_precision": cv_scores["test_precision"].std(),
                "std_f1": cv_scores["test_f1"].std(),
            }
            return scores

        # scores_macro = cross_validate_metric(score_type="macro")
        scores = cross_validate_metric(score_type="macro")

        # Train on all the data
        clf.fit(X, y)

        # rows: actual, columns: True labels
        # cols: actual, columns: predicted labels
        # Return confusion matrix (DO NOT USE plot_confusion_matrix, which is deprecated) (CHECK via test that it is not used)
        # Return confusion matrix (no need to plot it)
        ytrain_pred = clf.predict(X)
        ytest_pred = clf.predict(Xtest)
        conf_mat_train = confusion_matrix(y, ytrain_pred)
        conf_mat_test = confusion_matrix(ytest, ytest_pred)

        answer = {}
        answer["scores"] = scores
        answer["cv"] = cv
        answer["clf"] = clf
        answer["is_precision_higher_than_recall"] = (
            scores["mean_precision"] > scores["mean_recall"]
        )
        answer["explain_is_precision_higher_than_recall"] = "Explanatory text"
        answer["confusion_matrix_train"] = conf_mat_train  # 10 x 10 matrix
        answer["confusion_matrix_test"] = conf_mat_test  # 10 x 10 matrix

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use compute_class_weight to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ):
        n_splits = 5
        clf = SVC(random_state=self.seed, class_weight="balanced")
        # Shuffling is fine because of seed
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        # score = ["accuracy", "recall", "precision", "f1"]

        def cross_validate_metric(score_type: str):
            score = ["accuracy", "recall", "precision", "f1"]
            """
            scoring = {
                "accuracy": make_scorer(accuracy_score),
                "recall": make_scorer(recall_score, average=score_type),
                "precision": make_scorer(precision_score, average=score_type),
                "f1": make_scorer(f1_score, average=score_type),
            }
            print("===> scoring: ", scoring)
            """
            cv_scores = cross_validate(
                clf, X, y, scoring=score, cv=cv, return_train_score=False
            )

            scores = {
                "mean_accuracy": cv_scores["test_accuracy"].mean(),
                "mean_recall": cv_scores["test_recall"].mean(),
                "mean_precision": cv_scores["test_precision"].mean(),
                "mean_f1": cv_scores["test_f1"].mean(),
                "std_accuracy": cv_scores["test_accuracy"].std(),
                "std_recall": cv_scores["test_recall"].std(),
                "std_precision": cv_scores["test_precision"].std(),
                "std_f1": cv_scores["test_f1"].std(),
            }
            return scores

        # scores_macro = cross_validate_metric(score_type="macro")
        scores = cross_validate_metric(score_type="macro")

        # Train on all the data
        clf.fit(X, y)

        # rows: actual, columns: True labels
        # cols: actual, columns: predicted labels
        # Return confusion matrix (DO NOT USE plot_confusion_matrix, which is deprecated) (CHECK via test that it is not used)
        # Return confusion matrix (no need to plot it)
        ytrain_pred = clf.predict(X)
        ytest_pred = clf.predict(Xtest)
        conf_mat_train = confusion_matrix(y, ytrain_pred)
        conf_mat_test = confusion_matrix(ytest, ytest_pred)

        answer = {}
        answer["scores"] = scores
        answer["cv"] = cv
        answer["clf"] = clf
        answer["is_precision_higher_than_recall"] = (
            scores["mean_precision"] > scores["mean_recall"]
        )
        answer["explain_is_precision_higher_than_recall"] = "Explanatory text"
        answer["confusion_matrix_train"] = conf_mat_train  # 10 x 10 matrix
        answer["confusion_matrix_test"] = conf_mat_test  # 10 x 10 matrix
        """
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(ytrain), y=ytrain
        )
        weight_dict = {
            label: weight for label, weight in zip(np.unique(ytrain), class_weights)
        }
        """

        return answer
