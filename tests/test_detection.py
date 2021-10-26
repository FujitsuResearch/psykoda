from os import path
from shutil import rmtree

import numpy as np
from scipy.sparse import csr_matrix
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense

from psykoda.detection import LAYERNAME_ENCODER_OUTPUT, DeepSAD

rsrc_dir = path.join(
    path.dirname(path.abspath(__file__)),
    "rsrc",
    __name__.replace("tests.", ""),
)


def create_toy_data(num_normal=20, num_anomaly=2):
    input_shape = [2, 3, 4, 5]
    x_normal = np.zeros(shape=input_shape)
    x_anomaly = np.zeros_like(x_normal)
    x_normal[1, 1, 1:3, 1:3] = 1
    x_anomaly[1, 1, 1:3, 0:3] = 1

    X = np.vstack(
        [
            np.repeat(x_normal[None, :], num_normal, axis=0),
            np.repeat(x_anomaly[None, :], num_anomaly, axis=0),
        ]
    )
    # X = X + np.random.normal( 0, 0.1, input_shape )
    y = np.append(np.repeat(1.0, num_normal), np.repeat(-1.0, num_anomaly))
    y[0] = 0.0

    return X, y


def test_detection():
    """Basic properties of anomaly scores and SHAP values.

    Warnings
    --------
    Fails randomly at low probability since this test compares random variables
    (score, shap_values) with magic number (2).  Retry in that case.  May be
    workarounded by fixing random number generator seeds.
    """
    X, y = create_toy_data()
    X_2d = X.reshape([len(X), -1])
    X_sparse = csr_matrix(X_2d)

    path_model = path.join(rsrc_dir, "tmp_weights_best.h5")
    model = DeepSAD(DeepSAD.Config())
    model.train(X_sparse, y, path_model=path_model, config=DeepSAD.TrainConfig())
    score = model.compute_anomaly_score(X_sparse)
    shap_values = model.explain_anomaly(X_sparse, background_samples=X_sparse).reshape(
        (-1, 2, 3, 4, 5)
    )

    # normal samples have score 0
    assert (score[:-2] == 0).all()
    # anomaly samples have large scores
    assert (score[-2:] > 2).all()

    assert shap_values.shape == (22, 2, 3, 4, 5)
    with np.printoptions(threshold=np.inf):
        print(shap_values)

    # non-zero no-contribution features have SHAP 1
    assert (shap_values[:, 1, 1, 1:3, 1:3] == 1).all()
    # zero features have SHAP 0
    assert (shap_values[:, 0, :, :, :] == 0).all()
    assert (shap_values[:, 2:, :, :, :] == 0).all()
    assert (shap_values[:, 1, 0, :, :] == 0).all()
    assert (shap_values[:, 1, 2:, :, :] == 0).all()
    assert (shap_values[:, 1, 1, 0, :] == 0).all()
    assert (shap_values[:, 1, 1, 3:, :] == 0).all()
    assert (shap_values[:-2, 1, 1, 1:3, 0] == 0).all()
    assert (shap_values[:, 1, 1, 1:3, 3:] == 0).all()
    # anomaly features of anomaly samples have large SHAP
    assert (shap_values[-2:, 1, 1, 1:3, 0] > 2).all()

    rmtree(rsrc_dir)


def nearly_equal(actual, expected):
    epsilon = expected / 10
    return (expected - epsilon) <= actual <= (expected + epsilon)


def test_DSAD___init__():
    config = DeepSAD.Config(path_pretrained_model="hoge")

    dsad = DeepSAD(config)

    assert dsad.dim_hidden == config.dim_hidden
    assert dsad.eta == config.eta
    assert dsad.lam == config.lam
    assert dsad.path_pretrained_model == config.path_pretrained_model
    assert dsad.dim_input is None
    assert dsad.history is None
    assert dsad.detector is None


def test_DSAD__build_encoder():
    config = DeepSAD.Config()
    dsad = DeepSAD(config)
    dsad.dim_input = 5

    actual_encoder = dsad._build_encoder()

    assert actual_encoder.input_shape == (None, dsad.dim_input)
    actual_dense_count = 0
    for layer in actual_encoder.layers:
        if isinstance(layer, Dense):
            assert nearly_equal(layer.kernel_regularizer.get_config()["l2"], config.lam)
            assert layer.output_shape[1] == config.dim_hidden[actual_dense_count]
            actual_dense_count += 1
    assert actual_dense_count == len(config.dim_hidden)


def test_DSAD__build_autoencoder():
    config = DeepSAD.Config()
    dsad = DeepSAD(config)
    dsad.dim_input = 5
    encoder = dsad._build_encoder()

    actual_autoencoder = dsad._build_autoencoder(
        encoder.input, encoder.layers[-1].output
    )

    actual_dense_count = 0
    for layer in actual_autoencoder.layers[len(encoder.layers) : -1]:
        if isinstance(layer, Dense):
            assert nearly_equal(layer.kernel_regularizer.get_config()["l2"], config.lam)
            assert (
                layer.output_shape[1] == config.dim_hidden[::-1][1:][actual_dense_count]
            )
            actual_dense_count += 1
    actual_dense_count += 1
    assert actual_autoencoder.layers[-1].output_shape == (None, dsad.dim_input)
    assert nearly_equal(
        actual_autoencoder.layers[-1].kernel_regularizer.get_config()["l2"], config.lam
    )
    assert actual_dense_count == len(config.dim_hidden)


def test_DSAD__build_detector():
    config = DeepSAD.Config()
    dsad = DeepSAD(config)
    dsad.dim_input = 5
    encoder = dsad._build_encoder()
    center = np.array([0.1] * encoder.output_shape[1], dtype=np.float32)

    actual_detector = dsad._build_detector(encoder, center)

    assert isinstance(actual_detector, Model)


def test_compute_embeddings_01():
    config = DeepSAD.Config()
    dsad = DeepSAD(config)
    empty_X = np.array([])

    actual_ret = dsad.compute_embeddings(empty_X)

    assert actual_ret is None


def test_compute_embeddings_02():
    config = DeepSAD.Config()
    dsad = DeepSAD(config)
    dummy_X = np.array([[0, 1, 2]])
    dummy_Y = np.array([[1, 1, 1]])
    dummy_detector = Sequential(
        [Dense(5, input_shape=(3,)), Dense(1, name=LAYERNAME_ENCODER_OUTPUT)]
    )
    dummy_detector.compile(optimizer="adam", loss="binary_crossentropy")

    dsad.detector = dummy_detector
    actual_pred = dsad.compute_embeddings(dummy_X)

    assert actual_pred.shape == (1, 1)


def test_load_detector():
    config = DeepSAD.Config()
    dsad = DeepSAD(config)
    path_model = path.join(rsrc_dir, "dummy_detector")
    dummy_detector = Sequential([Dense(2, input_shape=(3,)), Dense(1)])
    dummy_detector.compile(optimizer="adam", loss="binary_crossentropy")
    dummy_detector.save(path_model)
    expected_weights_biases = dummy_detector.get_weights()

    dsad.load_detector(path_model)
    actual_weights_biases = dsad.detector.get_weights()

    try:
        assert len(dsad.detector.layers) == len(dummy_detector.layers)
        assert len(actual_weights_biases) == len(expected_weights_biases)
        for actual, expected in zip(actual_weights_biases, expected_weights_biases):
            assert np.all(actual == expected)
    finally:
        rmtree(rsrc_dir)
