# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u

# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        answer = u.starter_code()
        return answer

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

    def partB(
        self,
    ):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain_test = nu.scale_data(Xtrain)
        Xtest_test = nu.scale_data(Xtest)
        answer = {}
        # Enter your code and fill the `answer` dictionary
        #print(f"Xtest: {Xtest}\nXtest Length: {np.size(Xtest)}\nXtrain: {Xtrain}\nXtrain Length: {np.size(Xtrain)}\nYtest: {ytest}\nYtest length: {np.size(ytest)}\nYtrain:{ytrain}\nYtrain length: {np.size(ytrain)}")
        #print(f"Is Y only integers? {np.all(y[isinstance(y,int)])}\nIs X normalized? {nu.scale_data(X)}")
        #np.set_printoptions(threshold=np.inf)
        answer["length_Xtrain"] = np.size(Xtrain)  # Number of samples
        answer["length_Xtest"] = np.size(Xtest)
        answer["length_ytrain"] = np.size(ytrain)
        answer["length_ytest"] = np.size(ytest)
        answer["max_Xtrain"] = np.max(Xtrain)
        answer["max_Xtest"] = np.max(Xtest)
        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # scores, clf, cv = self.partC(X, y)
        # Enter your code and fill the `answer` dictionary
        cv = KFold(n_splits=5,shuffle=True,random_state=self.seed)
        clf = DecisionTreeClassifier(random_state=self.seed)
        clf.fit(X,y)
        scores = cross_validate(clf,X,y,cv=cv)
        accuracy = scores['test_score']
        #print(f"mean accuracy: {accuracy.mean()}\nstd accuracy: {accuracy.std()}")
        answer = {}
        answer["clf"] = clf  # the estimator (classifier instance)
        answer["cv"] = cv  # the cross validator instance
        # the dictionary with the scores  (a dictionary with
        # keys: 'mean_fit_time', 'std_fit_time', 'mean_accuracy', 'std_accuracy'.
        answer["scores"] = scores
        return answer

    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary
        cv = ShuffleSplit(n_splits=5,random_state=self.seed)
        clf = DecisionTreeClassifier(random_state=self.seed)
        clf.fit(X,y)
        scores = cross_validate(clf,X,y,cv=cv)
        accuracy = scores['test_score']
        #print(f"mean accuracy: {accuracy.mean()}\nstd accuracy: {accuracy.std()}")
        answer = {}
        # Answer: same structure as partC, except for the key 'explain_kfold_vs_shuffle_split'

        answer = {}
        answer["clf"] = clf
        answer["cv"] = cv
        answer["scores"] = scores
        answer["explain_kfold_vs_shuffle_split"] = "Shuffle split can be more helpful in examining the variablilty of models compared to Kfold, but can be worse and less stable when trying to do a basic model evaluation."
        return answer

    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        answer = {}
        for k in [2,5,8,16]:
            #print(f"k = {k}")
            cv = ShuffleSplit(n_splits=k,random_state=self.seed)
            clf = DecisionTreeClassifier(random_state=self.seed)
            clf.fit(X,y)
            scores = cross_validate(clf,X,y,cv=cv)
            accuracy = scores['test_score']
            #print(f"mean accuracy: {accuracy.mean()}\nstd accuracy: {accuracy.std()}")
            answer[k] = {'scores':scores, 'cv':cv, 'clf':clf}
        # Answer: built on the structure of partC
        # `answer` is a dictionary with keys set to each split, in this case: 2, 5, 8, 16
        # Therefore, `answer[k]` is a dictionary with keys: 'scores', 'cv', 'clf`

        # Enter your code, construct the `answer` dictionary, and return it.

        return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ) -> dict[str, Any]:
        """ """
        answer = {}

        # Enter your code, construct the `answer` dictionary, and return it.
        cv = ShuffleSplit(n_splits=5,random_state=self.seed)
        clf_RF = RandomForestClassifier(random_state=self.seed)
        clf_RF.fit(X,y)
        clf_DT = DecisionTreeClassifier(random_state=self.seed)
        clf_DT.fit(X,y)
        scores_RF = cross_validate(clf_RF,X,y,cv=cv)
        scores_DT = cross_validate(clf_DT,X,y,cv=cv)
        accuracy_RF = scores_RF['test_score']
        accuracy_DT = scores_DT['test_score']
        #print(f"mean accuracy: {accuracy.mean()}\nstd accuracy: {accuracy.std()}")
        answer["clf_RF"] = clf_RF
        answer["clf_DT"] = clf_DT
        answer["cv"] = cv
        answer["scores_RF"] = {"mean_fit_time":scores_RF['fit_time'].mean(),"std_fit_time":scores_RF['fit_time'].std(),"mean_accuracy":accuracy_RF.mean(),"std_accuracy":accuracy_RF.std()}
        answer["scores_DT"] = {"mean_fit_time":scores_DT['fit_time'].mean(),"std_fit_time":scores_DT['fit_time'].std(),"mean_accuracy":accuracy_DT.mean(),"std_accuracy":accuracy_DT.std()}
        answer["model_highest_accuracy"] = "RF"
        answer["model_lowest_variance"] = np.min([accuracy_RF.std()**2,accuracy_DT.std()**2])
        answer["model_fastest"] = np.min([scores_RF['fit_time'],scores_DT['fit_time']])
        """
         Answer is a dictionary with the following keys: 
            "clf_RF",  # Random Forest class instance
            "clf_DT",  # Decision Tree class instance
            "cv",  # Cross validator class instance
            "scores_RF",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "scores_DT",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
            "model_highest_accuracy" (string)
            "model_lowest_variance" (float)
            "model_fastest" (float)
        """

        return answer

    # ----------------------------------------------------------------------
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """

    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - clf: The classifier instance to use for classification.
        - cv: The cross validator instance to use for cross validation.
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """

        # refit=True: fit with the best parameters when complete
        # A test should look at best_index_, best_score_ and best_params_
        """
        List of parameters you are allowed to vary. Choose among them.
         1) criterion,
         2) max_depth,
         3) min_samples_split, 
         4) min_samples_leaf,
         5) max_features 
         5) n_estimators
        """

        answer = {}

        # Enter your code, construct the `answer` dictionary, and return it.

        #asked chatgpt how to use parameter grids.
        param_grid = {
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 4, 6],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2']
                    }
        #get original mean accuracy.
        clf = RandomForestClassifier(random_state=self.seed)
        clf.fit(X,y)
        cv = ShuffleSplit(n_splits=5,random_state=self.seed)
        cm_original_train = confusion_matrix(y,clf.predict(X))
        cm_original_test = confusion_matrix(ytest,clf.predict(Xtest))

        TP = cm_original_test[1, 1]
        TN = cm_original_test[0, 0]
        FP = cm_original_test[0, 1]
        FN = cm_original_test[1, 0]
        accuracy_original_test = (TP + TN)/(TP+TN+FP+FN)

        TP = cm_original_train[1, 1]
        TN = cm_original_train[0, 0]
        FP = cm_original_train[0, 1]
        FN = cm_original_train[1, 0]
        accuracy_original_train = (TP + TN)/(TP+TN+FP+FN)
        #scores = cross_validate(clf,X,y,cv=cv)
        #original_mean_accuracy_train = scores['test_score'].mean()
        

        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=self.seed), param_grid=param_grid, cv=cv,scoring='accuracy',n_jobs=-1)
        grid_search.fit(X,y)
        best_parameters = grid_search.best_params_
        best_clf = RandomForestClassifier(**best_parameters, random_state=self.seed)
        best_clf.fit(X,y)
        cm_best_train = confusion_matrix(y,best_clf.predict(X))
        cm_best_test = confusion_matrix(ytest,best_clf.predict(Xtest))

        TP = cm_best_test[1, 1]
        TN = cm_best_test[0, 0]
        FP = cm_best_test[0, 1]
        FN = cm_best_test[1, 0]
        accuracy_best_test = (TP + TN)/(TP+TN+FP+FN)

        TP = cm_best_train[1, 1]
        TN = cm_best_train[0, 0]
        FP = cm_best_train[0, 1]
        FN = cm_best_train[1, 0]
        accuracy_best_train = (TP + TN)/(TP+TN+FP+FN)


        #new_scores = cross-validate(best_clf,X,y,cv=cv)
        #new_mean_accuracy = new_scores['test_score'].mean()

        answer = {
                    "clf": clf,
                    "default_parameters": clf.get_params(),
                    "best_estimator": best_clf,
                    "grid_search": grid_search,
                    "mean_accuracy_cv": grid_search.best_score_,
                    "confusion_matrix_train_orig": cm_original_train,
                    "confusion_matrix_train_best": cm_best_train,
                    "confusion_matrix_test_orig": cm_original_test,
                    "confusion_matrix_test_best": cm_best_test,
                    "accuracy_orig_full_training": accuracy_original_train,
                    "accuracy_best_full_training": accuracy_best_train,
                    "accuracy_orig_full_testing": accuracy_original_test,
                    "accuracy_best_full_testing": accuracy_best_test,
                }
        """
           `answer`` is a dictionary with the following keys: 
            
            "clf", base estimator (classifier model) class instance
            "default_parameters",  dictionary with default parameters 
                                   of the base estimator
            "best_estimator",  classifier class instance with the best
                               parameters (read documentation)
            "grid_search",  class instance of GridSearchCV, 
                            used for hyperparameter search
            "mean_accuracy_cv",  mean accuracy score from cross 
                                 validation (which is used by GridSearchCV)
            "confusion_matrix_train_orig", confusion matrix of training 
                                           data with initial estimator 
                                (rows: true values, cols: predicted values)
            "confusion_matrix_train_best", confusion matrix of training data 
                                           with best estimator
            "confusion_matrix_test_orig", confusion matrix of test data
                                          with initial estimator
            "confusion_matrix_test_best", confusion matrix of test data
                                            with best estimator
            "accuracy_orig_full_training", accuracy computed from `confusion_matrix_train_orig'
            "accuracy_best_full_training"
            "accuracy_orig_full_testing"
            "accuracy_best_full_testing"
               
        """
        #print(answer)
        return answer
