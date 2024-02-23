# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.
import utils as u
import numpy as np
from numpy.typing import NDArray
from typing import Any
from part_1_template_solution import Section1 as Part1
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
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
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        
        answer = {}
        # Enter your code and fill the `answer`` dictionary
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        #Xtrain_test = nu.scale_data(Xtrain)
        #Xtest_test = nu.scale_data(Xtest)
        print(f"Xtest: {Xtest}\nXtest Length: {np.size(Xtest)}\nXtrain: {Xtrain}\nXtrain Length: {np.size(Xtrain)}\nYtest: {ytest}\nYtest length: {np.size(ytest)}\nYtrain:{ytrain}\nYtrain length: {np.size(ytrain)}")
        answer['nb_classes_train'] = np.size(set(ytrain))
        answer['nb_classes_test'] = np.size(set(ytest))
        _,class_count_train = np.unique(ytrain,return_counts=True)
        _,class_count_test = np.unique(ytest,return_counts=True)
        print(class_count_test)
        print(class_count_train)
        answer['class_count_train'] = class_count_train
        answer['class_count_test'] = class_count_test
        answer['length_Xtrain'] = np.size(Xtrain)
        answer['length_Xtest'] = np.size(Xtest)
        answer['length_ytrain'] = np.size(ytrain)
        answer['length_ytest'] = np.size(ytest)
        answer['max_Xtrain'] = np.max(Xtrain)
        answer['max_Xtest'] = np.max(Xtest)
        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

        #Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        #ytrain = ytest = np.zeros([1], dtype="int")

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        # Enter your code and fill the `answer`` dictionary
        answer = {}
        

        def partF(
                    self,
                    X: NDArray[np.floating],
                    y: NDArray[np.int32],
                    Xtest: NDArray[np.floating],
                    ytest: NDArray[np.int32],
                ) -> dict[str, Any]:

                    answer = {}

                    # Enter your code, construct the `answer` dictionary, and return it.
                    cv = ShuffleSplit(n_splits=5,random_state=self.seed)
                    clf_LR = LogisticRegression(random_state=self.seed,max_iter=300)
                    clf_LR.fit(X,y)
                    cm_original_train = confusion_matrix(y,clf_LR.predict(X))
                    cm_original_test = confusion_matrix(ytest,clf_LR.predict(Xtest))

                    #TP = cm_original_test[1, 1]
                    #TN = cm_original_test[0, 0]
                    #FP = cm_original_test[0, 1]
                    #FN = cm_original_test[1, 0]
                    #accuracy_test = (TP + TN)/(TP+TN+FP+FN)

                    #TP = cm_original_train[1, 1]
                    #TN = cm_original_train[0, 0]
                    #FP = cm_original_train[0, 1]
                    #FN = cm_original_train[1, 0]
                    #accuracy_train = (TP + TN)/(TP+TN+FP+FN)

                    scores_train = cross_validate(clf_LR,X,y,cv=cv)
                    scores_test = cross_validate(clf_LR,Xtest,ytest,cv=cv)
                    
                    #print(f"mean accuracy: {accuracy.mean()}\nstd accuracy: {accuracy.std()}")
                    
                    answer["scores_train_F"] = scores_train
                    answer["scores_test_F"] = scores_test
                    answer["mean_cv_accuracy_F"] = scores_train['test_score'].mean()
                    answer["clf"] = clf_LR
                    answer["cv"] = cv
                    answer["conf_mat_train"] = cm_original_train
                    answer["conf_mat_test"] = cm_original_test
                    return answer

        for ind,n in enumerate(ntrain_list):
            Xtrain = X[0:n, :]
            ytrain = y[0:n]
            Xtest = X[n:n+ntest_list[ind]]
            ytest = y[n:n+ntest_list[ind]]
            _,class_count_train = np.unique(ytrain,return_counts=True)
            _,class_count_test = np.unique(ytest,return_counts=True)
            answer[n] = {
            "partC" : Part1.partC(self,Xtrain,ytrain),
            "partD" : Part1.partD(self,Xtrain,ytrain),
            "partF" : partF(self,Xtrain,ytrain,Xtest,ytest),
            "ntrain" : n,
            "ntest" : ntest_list[ind],
            "class_count_train" : class_count_train.tolist(),
            "class_count_test" : class_count_test.tolist()
            }
            
        """
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """
        print(answer)
        return answer
