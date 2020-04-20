
from sklearn.utils import class_weight
from ..preprocess import Data
from deepctr.models import DeepFM


class DeepModel:
    def __init__(self, model_name, model_architecture="DeepFM"):
        self.model_name = model_name
        self.model_architecture = model_architecture

        self.model = None
        self.history = None
        self.data = None
        self.callbacks = []

    # requires tf2
    # def set_notebook_mode(self):
    #    progress_bar_cb = tfa.callbacks.TQDMProgressBar() #TQDMNotebookCallback(leave_inner=True, leave_outer=True)
    #    self.callbacks.append(progress_bar_cb)

    def prepare_data(self, data_source, sparse_features, target, test_size=0.1):
        self.data = Data(
            sparse_features, target, data_format="deepctr", test_size=test_size
        )
        self.data.ingest(data_source)
        self.data.prepare()

    def build(self, task):
        assert task in ['regression','binary']
        if self.model_architecture == "DeepFM":
            self.model = DeepFM(
                self.data.linear_feature_columns,
                self.data.dnn_feature_columns,
                task=task,
            )
        else:
            raise NotImplementedError('At the current stage of the development, only a DeepFM is supported')

        task_attr =  {
            'regression': {
                'loss' : 'mse',
                'metrics' : 'mse'
            },
            'binary': {
                'loss':'binary_crossentropy',
                'metrics': 'accuracy'
            }
        }
        if task == "regression":
            loss = "mse"
            metrics = "mse"
        elif task == "binary":
            loss = "binary_crossentropy"
            metrics = "accuracy"

        self.model.compile(optimizer="adam", loss=task_attr[task]['loss'], metrics=task_attr[task]['metrics'])

    def train(self, batch_size=256, epochs=10, validation_split=0.1):
        #class_weights = class_weight.compute_class_weight(
        #    "balanced", np.unique(self.data.y_train[:, 0]), self.data.y_train[:, 0]
        #)
        self.history = self.model.fit(
            self.data.X_train,
            self.data.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=2,
            #class_weight=class_weights,
            callbacks=self.callbacks,
        )

    def evaluate(self):
        self.model.evaluate(self.data.X_test, self.data.y_test, batch_size=4096)

    def prepare_input(self, df):
        df = df.copy()
        for feat in self.data.sparse_features:
            lbe = self.data.encoders[feat]
            df[feat] = lbe.transform(df[feat])

        X = {name: df[name].values for name in self.data.feature_names}
        return X

    def predict(self, X, batch_size=256):
        return self.model.predict(X, batch_size=batch_size)
