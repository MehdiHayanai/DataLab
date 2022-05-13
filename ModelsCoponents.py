# from time import sleep
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split






class DataLabModel:

    def __init__(self, model_option):
        self.option = model_option
        self.model = None
        self.params =  {}
        self.best_score = 0


    def set_model(self):
    # 'SVC', 'RandomForestClassifier', 'Gaussian Naive Bayes', 'Knn', 'LogisticRegression'
        if self.option == "SVC":
            model = SVC()
        
        elif self.option == "RandomForestClassifier":
            model = RandomForestClassifier()

        elif self.option == "Gaussian Naive Bayes":
            model = GaussianNB()

        elif self.option == "Knn":
            model = KNeighborsClassifier()

        elif self.option == "LogisticRegression":
            model = LogisticRegression()

        else :
            return None
        return model

    def _set_params(self, key, argument):
        if key in self.model.model.get_params():
            try :
                self.params[key] = argument
                # success
                return True
            except:
                return False

    def get_model(self):
        return self.model

    def fit(self, X, y, trainsize):
        # sets model to the best model of the current and old one 
        # returns last model evaluation

        # train test spliting from user input
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=trainsize)

        # create a model
        model = self.set_model()
        
        # for first time current model is equal to last model
        if  self.model == None:
            self.model = model

        # fit and get score 
        model.fit(X_train, y_train)
        evaluation = model.score(X_test, y_test)

        # model is equal to the best model
        if evaluation > self.best_score:
            self.best_score = evaluation
            self.model = model

        return evaluation


    def update_params(self):
        try:
            self.model.set_params(self.params)
            return True
        except:
            return False

