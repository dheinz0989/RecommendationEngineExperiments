
import pandas as pd

from pathlib import Path, WindowsPath
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from deepctr.inputs import SparseFeat
from deepctr.inputs import get_feature_names


class Data:
    def __init__(self, sparse_features, target, data_format, test_size):
        self.data_format = data_format

        # ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
        self.sparse_features = sparse_features
        # ['rating']
        self.target = target

        self.test_size = test_size

        self.encoders = {}

        self.feature_names = None
        self.linear_feature_columns = None
        self.dnn_feature_columns = None

        self.X_train = None
        self.y_train = None

        self.X_test = None
        self.y_test = None

    def ingest(self,data_source,**options):
        if isinstance(data_source,WindowsPath):
            data_source = str(data_source)
        if isinstance(data_source,str):
            suffix = Path(data_source).suffix
            reader_mapping = {
                '.csv':pd.read_csv,
                '.json':pd.read_json,
                '.parquet':pd.read_parquet,
                '.pl': pd.read_pickle,
                '.pickle': pd.read_pickle
            }
            print(f'Data format was found in the "{suffix}" format. The respective reader function is {reader_mapping[suffix]}')
            reader = reader_mapping[suffix]
            self.input = reader(data_source, **options)

        elif isinstance(data_source,pd.core.frame.DataFrame):
            self.input = pd.DataFrame(data_source)
        return self

    def prepare(self):
        if self.data_format == "deepctr":
            # 1.Label Encoding for sparse features, and
            # simple Transformation for dense features
            for feat in self.sparse_features:
                lbe = LabelEncoder()
                self.input[feat] = lbe.fit_transform(self.input[feat])
                self.encoders[feat] = lbe

            # 2.count #unique features for each sparse field
            fixlen_feature_columns = [
                SparseFeat(feat, self.input[feat].nunique(), embedding_dim=4)
                for feat in self.sparse_features
            ]

            self.linear_feature_columns = fixlen_feature_columns
            self.dnn_feature_columns = fixlen_feature_columns

            self.feature_names = get_feature_names(
                self.linear_feature_columns + self.dnn_feature_columns
            )

            # 3.generate input data for model
            train, test = train_test_split(self.input, test_size=self.test_size)

            self.X_train = {name: train[name].values for name in self.feature_names}
            self.y_train = train[self.target].values

            self.X_test = {name: test[name].values for name in self.feature_names}
            self.y_test = test[self.target].values
        else:
            raise ("Not supported dataset:" + self.data_format)