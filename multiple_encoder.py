from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder


class MultiColumnLabelEncoder(TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode
        self.encoders = None

    def fit(self, X, y=None):
        self.encoders = {}
        for col in self.columns:
            self.encoders[col] = LabelEncoder().fit(X[col])

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        x_out = X.copy(deep=False)
        for col_name in self.columns:
            x_out[col_name] = self.encoders[col_name].transform(x_out[col_name])

        return x_out
