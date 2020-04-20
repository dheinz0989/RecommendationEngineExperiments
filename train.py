from src.model import DeepModel

import argparse

def pipe_data_to_model(
    data_source,
    feature_columns,
    target_columns,
    model_name,
    model_backend,
    model_architecture,
    model_task,
    epochs,
    batch_size,
    valid_size=0.1,
    test_size=0.1,
):
    assert model_backend in ['deepctr','lightfm']
    if model_backend == "deepctr":
        model = DeepModel(model_name=model_name, model_architecture=model_architecture)
    elif model_backend == "lightfm":
        raise ("Not supported model backend:" + model_backend)

    # if mode == 'notebook':
    #    model.set_notebook_mode()

    model.prepare_data(
        data_source=data_source,
        sparse_features=feature_columns,
        target=target_columns,
        test_size=test_size,
    )

    model.build(task=model_task)

    print("TRAINING>>>>>>>>>>>>>>>>>>>>>>>>>")
    model.train(epochs=epochs, batch_size=batch_size, validation_split=valid_size)

    print("EVALUATING>>>>>>>>>>>>>>>>>>>>>>>")
    model.evaluate()

    return model


#def test_confusion_matrix(model, threshold=0.5, normalize=True):
#    plot_confusion_matrix(
#        model.data.y_test[:, 0],
#        model.predict(model.data.X_test)[:, 0] > threshold,
#        target_names=[0, 1],
#        normalize=normalize,
#        figsize=(7, 7),
#    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_source",
        type=str,
        required=True,
        help="Path to the data input file (currently supported format is csv). See test/data/movielens for example.",
    )

    parser.add_argument(
        "--feature_columns",
        type=str,
        nargs='+',
        required=True,
        help='Comma separated list of feature column names from the data source. E.g., "userId,movieId".',
    )

    parser.add_argument(
        "--target_columns",
        type=str,
        required=True,
        help='Comma separated list of target column names from the data source. E.g., "click".',
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="String specifying model name to be used for model metadata and for saving result in binary files.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="String specifying path to save the model binary files.",
    )

    parser.add_argument(
        "--model_backend",
        type=str,
        required=False,
        default="deepctr",
        help='String specifying model backend we want to use ["deepctr" | "lightfm"]. Currently only "deepctr" is supported, which is also a default value.',
    )

    parser.add_argument(
        "--model_architecture",
        type=str,
        required=False,
        default="DeepFM",
        help='String specifying model architecture to be used with model backend. Default is "DeepFM".',
    )

    parser.add_argument(
        "--model_task",
        type=str,
        required=False,
        default="binary",
        help='String specifying model task type ["binary" | "regression"]. Default is "binary".',
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=3,
        help="Integer number of epochs to be used for training. Default is 3.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=256,
        help="Integer number specifying batch size to be used for training. Default is 256.",
    )

    parser.add_argument(
        "--valid_size",
        type=float,
        required=False,
        default=0.1,
        help="Float number specifying data fraction to be used for validation purposes. Default is 0.1.",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        required=False,
        default=0.1,
        help="Float number specifying data fraction to be used for testing purposes. Default is 0.1.",
    )

    parser.add_argument(
        "--keep_data",
        type=bool,
        required=False,
        default=False,
        help="Boolean value specifying if the trained model should keep data when saved. Default is False.",
    )

    args = vars(parser.parse_args())

    # model training
    model = pipe_data_to_model(
        data_source=args["data_source"],
        feature_columns=args["feature_columns"],
        target_columns=args["target_columns"].split(","),
        model_name=args["model_name"],
        model_backend=args["model_backend"],
        model_architecture=args["model_architecture"],
        model_task=args["model_task"],
        epochs=args["epochs"],
        batch_size=args["batch_size"],
        valid_size=args["valid_size"],
        test_size=args["test_size"],
    )