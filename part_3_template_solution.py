import numpy as np
from numpy.typing import NDArray
from typing import Any

from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score, make_scorer, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import utils as u
import new_utils as nu
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
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """
        # Enter code and return the `answer`` dictionary

        answer = {}

        clf = LogisticRegression(random_state=self.seed, max_iter=300)
        clf.fit(Xtrain, ytrain)
        training_scores = []
        testing_scores = []

        # Calculate top_k_accuracy_score for k=1,2,3,4,5
        ks = [1,2,3,4,5]
        for k in ks:
            training_score = top_k_accuracy_score(ytrain, clf.predict_proba(Xtrain), k=k)
            testing_score = top_k_accuracy_score(ytest, clf.predict_proba(Xtest), k=k)
            training_scores.append(training_score)
            testing_scores.append(testing_score)

        # Plot k vs. score for both training and testing data
        plt.figure(figsize=(10, 6))
        plt.plot(ks, training_scores, label='Training Score', marker='o')
        plt.plot(ks, testing_scores, label='Testing Score', marker='s')
        plt.xlabel('k')
        plt.ylabel('Top-k Accuracy Score')
        plt.title('Top-k Accuracy Score vs. k for Training and Testing Data')
        plt.legend()
        plt.grid(True)
        plt.show()

        answer = {}
        answer["clf"] = clf
        answer["plot_k_vs_score_train"] = []
        answer["plot_k_vs_score_test"] = []
    
        for k in ks:
            # Compute top-k accuracy scores for training and testing sets
            score_train = top_k_accuracy_score(ytrain, clf.predict_proba(Xtrain), k=k)
            score_test = top_k_accuracy_score(ytest, clf.predict_proba(Xtest), k=k)
        
            # Populate the dictionary with scores
            answer[k] = {"score_train": score_train, "score_test": score_test}
        
            # Add plot data
            answer["plot_k_vs_score_train"].append((k, score_train))
            answer["plot_k_vs_score_test"].append((k, score_test))
    
        # Add commentary
        answer["text_rate_accuracy_change"] = "The accuracy typically increases with k, showing the model's ability to include the correct label in the top-k predictions."
        answer["text_is_topk_useful_and_why"] = "Top-k accuracy is a useful metric for evaluating classifiers, especially in multi-class scenarios or when a range of acceptable predictions exists. It provides insight into the model's prediction confidence and can help in understanding performance beyond simple accuracy."

        
        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """

        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}


        X, y, Xtest, ytest = u.prepare_data()
        X, y = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        y = np.where(y == 7, 0, np.where(y == 9, 1, y))
        ytest = np.where(ytest == 7, 0, np.where(ytest == 9, 1, ytest))

        # Function to remove 90% 
        def remove_ninety_percent_ones(X, y):
            mask = np.random.rand(len(y)) > 0.9  # True for about 10% of the time
            return X[(y == 0) | ((y == 1) & mask)], y[(y == 0) | ((y == 1) & mask)]

        # Apply the filtering function
        Xtrain, ytrain = remove_ninety_percent_ones(X, y)
        Xtest, ytest = remove_ninety_percent_ones(Xtest, ytest)

        # Scale the data
        
        #print(f"is x scaled? {nu.scale_data(X)}")
        #print(f"Is y all integers? {np.all(y[isinstance(y,int)])}")
    # Fill in the answer dictionary
        answer = {
            "length_Xtrain": np.size(Xtrain),
            "length_Xtest": np.size(Xtest),
            "length_ytrain": np.size(ytrain),
            "length_ytest": np.size(ytest),
            "max_Xtrain": np.max(Xtrain),
            "max_Xtest": np.max(Xtest)
        }

        #print(answer)
        # Answer is a dictionary with the same keys as part 1.B

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
    ) -> dict[str, Any]:
        """"""
        
        #X, y = u.filter_out_7_9s(X, y)
        #Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        # Enter your code and fill the `answer` dictionary
        answer = {}
        scaler = StandardScaler()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        clf = SVC(random_state=self.seed, probability=True)
        


        clf.fit(scaler.fit_transform(X), y)
        # Defining scorers

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro')
        }

        # Running cross-validation
        
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)

        # Calculating mean and std for the scores




        

        y_pred_train = clf.predict(X)
        y_pred_test = clf.predict(scaler.transform(Xtest))

        # Confusion matricies

        confusion_matrix_train = confusion_matrix(y, y_pred_train)
        confusion_matrix_test = confusion_matrix(ytest,y_pred_test)

        

        # Plotting with only matplotlib
        fig, ax = plt.subplots(figsize=(10, 7))
        cax = ax.matshow(confusion_matrix_train, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        # Adding annotations to each cell
        for (i, j), val in np.ndenumerate(confusion_matrix_train):

           ax.text(j, i, f'{val}', ha='center', va='center', color='black')


        ax.set_xticklabels([''] + ["Predicted 0", "Predicted 1", "Predicted 2"], rotation=45)
        ax.set_yticklabels([''] + ["True 0", "True 1", "True 2"])

        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.title('Confusion Matrix')
        plt.show()

        answer = {
            "scores": {
                "mean_accuracy": scores['test_accuracy'].mean(),
                "std_accuracy": scores['test_accuracy'].std(),
                "mean_precision": scores['test_precision'].mean(),
                "std_precision": scores['test_precision'].std(),
                "mean_recall": scores['test_recall'].mean(),
                "std_recall": scores['test_recall'].std(),
                "mean_f1": scores['test_f1'].mean(),
                "std_f1": scores['test_f1'].std()
            },
            "cv": cv,
            "clf": clf,
            "is_precision_higher_than_recall": scores['test_precision'].mean()> scores['test_recall'].mean(),
            "explain_is_precision_higher_than_recall": "Precision is higher than recall, indicating that the model is more conservative in predicting positive classes; it prefers to be more confident when it does so. This can be due to the model prioritizing the avoidance of false positives over the capture of all positives." if scores['test_precision'].mean()> scores['test_recall'].mean() else "Recall is higher than precision, indicating that the model prioritizes capturing as many positives as possible, even at the risk of increasing false positives.",
            "confusion_matrix_train": confusion_matrix_train,
            "confusion_matrix_test": confusion_matrix_test
        }

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """
        #print(answer)
        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        

        # Compute class weights
        classes = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weights_dict = dict(zip(classes, class_weights))
        #print(class_weights_dict)
        scaler = StandardScaler()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        clf = SVC(random_state=self.seed, probability=True,class_weight=class_weights_dict)
        


        clf.fit(scaler.fit_transform(X), y)
        # Defining scorers

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro')
        }

        # Running cross-validation
        
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)

        # Calculating mean and std for the scores




        

        y_pred_train = clf.predict(X)
        y_pred_test = clf.predict(scaler.transform(Xtest))

        # Confusion matricies

        confusion_matrix_train = confusion_matrix(y, y_pred_train)
        confusion_matrix_test = confusion_matrix(ytest,y_pred_test)

        

        # Plotting with only matplotlib
        fig, ax = plt.subplots(figsize=(10, 7))
        cax = ax.matshow(confusion_matrix_train, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        # Adding annotations to each cell
        for (i, j), val in np.ndenumerate(confusion_matrix_train):

           ax.text(j, i, f'{val}', ha='center', va='center', color='black')


        ax.set_xticklabels([''] + ["Predicted 0", "Predicted 1", "Predicted 2"], rotation=45)
        ax.set_yticklabels([''] + ["True 0", "True 1", "True 2"])

        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.title('Confusion Matrix')
        plt.show()

        answer = {
            "scores": {
                "mean_accuracy": scores['test_accuracy'].mean(),
                "std_accuracy": scores['test_accuracy'].std(),
                "mean_precision": scores['test_precision'].mean(),
                "std_precision": scores['test_precision'].std(),
                "mean_recall": scores['test_recall'].mean(),
                "std_recall": scores['test_recall'].std(),
                "mean_f1": scores['test_f1'].mean(),
                "std_f1": scores['test_f1'].std()
            },
            "cv": cv,
            "clf": clf,
            "class_weights": class_weights_dict,
            "confusion_matrix_train": confusion_matrix_train,
            "confusion_matrix_test": confusion_matrix_test,
            "explain_purpose_of_class_weights": "Class weights are used to address the imbalance in the training data by giving more importance to underrepresented classes. This helps in improving the classifier's performance on these classes.",
            "explain_performance_difference": "Using class weights likely improved the model's performance on underrepresented classes, potentially leading to higher recall or precision for these classes compared to Part C, where class weights were not used."
        }

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
