from classifier import classifier
import pprint
import pandas as pd

class decision_tree(classifier):

    def __init__(self, max_depth, criterion="entropy"):
        self.pp = pprint.PrettyPrinter(indent=4)
        self.max_depth = max_depth

        self.criterion_dict = {
            "entropy": self.entropy,
            "gini": self.gini
        }

        if criterion not in self.criterion_dict:
            print("no such criterion")
            return
        self.criterion = criterion


    def gini(self, Y):
        size = len(Y)
        counts = dict()
        for y in Y:
            if y not in counts:
                counts[y] = 0.
            counts[y] += 1.
        gini = 0.
        for key in counts:
            prob = counts[key] / size
            gini += prob * (1-prob)
        return gini


    def entropy(self, Y):
        from math import log

        size = len(Y)
        counts = dict()
        for y in Y:
            if y not in counts:
                counts[y] = 0.
            counts[y] += 1.
        entropy = 0.
        for key in counts:
            prob = counts[key] / size
            entropy -= prob * log(prob,2)
        return entropy


    def split_data_raw(self, X, Y, axis, value):
        return_x = []
        return_y = []
        axisl_i = X.columns.values.tolist().index(axis)
        X = X.values.tolist()
        Y = Y.values.tolist()

        for x, y in (zip(X, Y)):
            if x[axisl_i] == value:
                reduced_x = x[:axisl_i]
                reduced_x.extend(x[axisl_i+1:])
                return_x.append(reduced_x)
                return_y.append(y)
        return pd.DataFrame(return_x), pd.Series(return_y)


    def split_data(self, X, Y, col, value):
        return_x = X[X[col] == value]
        return_x = return_x.drop(col, axis=1)
        # return_x = return_x.drop_duplicates()
        # return_x = return_x[~return_x.index.duplicated()]
        return_y = Y[return_x.index]
        return return_x, return_y


    def choose_feature(self, X, Y):
        entropy = self.criterion_dict[self.criterion](Y.values.tolist())
        # entropy = self.criterion_dict[self.criterion](Y)

        best_information_gain = 0.
        best_feature = -1

        for i in self.feature_list:  # For each feature
            # feature_list = [x[i] for x in X.values.tolist()]
            feature_list = X.loc[:, i].unique().tolist()
            values = set(feature_list)
            entropy_i = 0.
            for value in values:
                sub_x, sub_y = self.split_data(X, Y, i, value)
                prob = len(sub_x.values.tolist()) / float(len(X.values.tolist()))
                # prob = len(sub_x) / float(len(X))
                entropy_i += prob * self.criterion_dict[self.criterion](sub_y)
            info_gain = entropy - entropy_i
            if info_gain > best_information_gain:
                best_information_gain = info_gain
                best_feature = i
        if best_feature in self.feature_list:
            self.feature_list.remove(best_feature)
        return best_feature


    def class_dict(self, Y):
        classes = dict()
        for y in Y:
            if y not in classes:
                classes[y] = 0
            classes[y] += 1
        return classes


    def majority(self, Y):
        from operator import itemgetter
        # Use this function if a leaf cannot be split further and
        # ... the node is not pure

        classcount = self.class_dict(Y)
        # sorted_classcount = sorted(classcount.iteritems(), key=itemgetter(1), reverse=True)
        sorted_classcount = sorted(classcount.items(), key=itemgetter(1), reverse=True)
        return sorted_classcount[0][0]


    def build_tree(self, X, Y, depth=1):
        # IF there's only one instance or one class, don't continue to split
        # if len(Y) <= 1 or len(self.class_dict(Y)) == 1:
        #     return Y[0]

        if self.max_depth != -1 and depth >= self.max_depth:
            return self.majority(Y.values.tolist())

        # # IF there's only one instance or one class, don't continue to split
        if len(Y.values.tolist()) <= 1 or len(self.class_dict(Y.values.tolist())) == 1:
            return Y.values.tolist()[0]

        # if len(X[0]) == 1:
        #     return self.majority(Y)   # TODO: Fix this

        if len(X.values.tolist()[0]) == 1:
            return self.majority(Y.values.tolist())   # TODO: Fix this

        best_feature = self.choose_feature(X, Y)

        if isinstance(best_feature, int):
            # if best_feature < 0 or best_feature >= len(X.values.tolist()[0]):
            if best_feature < 0:
                return self.majority(Y.values.tolist())
                # return None

        this_tree = dict()
        # feature_values = [example[best_feature] for example in X]
        feature_values = X[best_feature].values.tolist()
        unique_values = set(feature_values)
        for value in unique_values:
            # Build a node with each unique value:
            subtree_x, subtree_y = self.split_data(X, Y, best_feature, value)
            if best_feature not in this_tree:
                this_tree[best_feature] = dict()
            if value not in this_tree[best_feature]:
                this_tree[best_feature][value] = 0
            this_tree[best_feature][value] = self.build_tree(subtree_x, subtree_y, depth=depth+1)
        return this_tree


    def fit(self, X, Y, feature_list):
        self.feature_list = feature_list
        self.tree = self.build_tree(X, Y)
        # print("----------printing tree-----------------")
        # self.pp.pprint(self.tree)
        # self.pp.pprint("====================================")

    def predict(self, X):
        pred = []

        for x in X:
            pred.append(self.walk_tree(self.tree, x))
        # pred = X.apply(lambda row: self.walk_tree(self.tree, row), axis=1)
        return pred


    def walk_tree(self, tree, x):
        if isinstance(tree, (int, str)):
            return tree

        keys = list(tree.keys())
        feature_i = keys[0]
        sub_tree = tree[feature_i]
        result = None
        if x[feature_i] not in sub_tree:
            leaves = []
            for key in sub_tree.keys():
                leaves.append(self.walk_tree(sub_tree[key], x))
            result = self.majority(leaves)
        else:
            result = self.walk_tree(sub_tree[x[feature_i]], x)
        return result
