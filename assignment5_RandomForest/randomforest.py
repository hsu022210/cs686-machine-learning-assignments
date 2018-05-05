from classifier import classifier
from decision_tree import decision_tree
from sklearn.model_selection import train_test_split
import pprint
import random

class randomforest(classifier):

    def __init__(self, trees=10, max_depth=-1):
        self.pp = pprint.PrettyPrinter(indent=4)
        self.num_trees = trees
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.tree_list = self.create_list(self.num_trees)  # decision_tree list
        for t in self.tree_list:
            # subsample_x, subsample_y = self.subsample(X.values.tolist(), Y.values.tolist()) # Bagging
            subsample_x, subsample_y = self.subsample(X, Y) # Bagging
            feature_list = self.sample_of_features(X) # Random features
            # print("feature_list:", feature_list)
            t.fit(subsample_x, subsample_y, feature_list)


    def predict(self, X):
        from collections import defaultdict
        hypothesis_list = [t.predict(X) for t in self.tree_list]
        result = []
        for i in range(len(hypothesis_list[0])):
            counts = defaultdict(int)
            for each_tree_hypo in hypothesis_list:
                counts[each_tree_hypo[i]] += 1
            result.append(sorted(counts.items(), reverse=True, key=lambda tup: tup[1])[:len(self.tree_list)][0][0])
        return result


    def create_list(self, num_trees):
        return [decision_tree(self.max_depth) for i in range(num_trees)]


    def subsample(self, X, Y):
        # seed = random.randint(0, 123)
        # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
        # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
        # x = X.sample(frac=1.0, replace=True)
        x = X.sample(frac=1.0)
        y = Y[x.index]
        return x, y


    def sample_of_features(self, X):
        result = []
        for i in range(len(X.columns)//3):
            rand_i = random.randint(0, len(X.columns)-1)
            while X.columns[rand_i] in result:
                rand_i = random.randint(0, len(X.columns)-1)
            result.append(X.columns[rand_i])
        return result
