from classifier import classifier
from scipy.spatial import distance

class knn(classifier):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, Y):
        self.x = X
        self.y = Y

    def predict(self, X):
        hypothesis = []
        for i, row_i in X.iterrows():
            points = self.get_sorted_points(row_i)
            neighbors = self.get_top(self.k, points)
            hyp = self.majority_class(neighbors)
            hypothesis.append(hyp)
        return hypothesis

    def get_sorted_points(self, test_x_row):
        points = []
        for j, row_j in self.x.iterrows():
            i_vals = [int(v) for v in test_x_row.values]
            j_vals = [int(v) for v in row_j.values]
            dst = distance.euclidean(i_vals, j_vals)
            points.append((j, dst))
        points.sort(key=lambda tup: tup[1])
        return points

    def get_top(self, n, arr):
        return arr[:n]

    def majority_class(self, arr):
        classes = []
        for tup in arr:
            classes.append(self.y[tup[0]])

        most = classes[0]
        curr_most_count = 0
        for val in classes:
            if classes.count(val) > curr_most_count:
                most = val
                curr_most_count = classes.count(val)
        return most
