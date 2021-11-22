"""Anomaly Detection and Explanation."""

import dataclasses
import os
import random
from logging import getLogger
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, Layer, LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence

logger = getLogger(__name__)

KERNEL_INITIALIZER = "he_normal"
LAYERNAME_ENCODER_OUTPUT = "encoder_output"
REGULARIZER_L2 = (
    tf.keras.regularizers.L2 if tf.__version__ == "2.3.0" else tf.keras.regularizers.l2
)


class generator_autoencoder_training(Sequence):
    """Sparse matrix as batches of dense arrays"""

    def __init__(self, X: csr_matrix, batch_size: int):
        self.X = X
        self.batch_size = batch_size
        self.num_samples: int = X.shape[0]
        self.steps_per_epoch = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].toarray()
        # autoencoder
        return batch_X, batch_X

    def on_epoch_end(self):
        # add ops of shuffling all samples, if available
        pass


def loss_sad(c, eta=1.0):
    """Loss function for Deep SAD

    References
    ----------
    [1] L. Ruff, R. A. Vandermeulen, N. Görnitz, A. Binder, E. Müller, K.-R. Müller, M. Kloft,
    "Deep Semi-Supervised Anomaly Detection", https://arxiv.org/abs/1906.02694
    """

    def loss_function(labels: tf.Tensor, embeddings: tf.Tensor):
        """Loss function for Deep SAD

        Parameters
        ----------
        labels
            ground truth labels

            :shape: (batch_size, 1)

        embeddings
            outputs of encoder phi

            :shape: (batch_size, dim_embedding)
        """
        labels = tf.reshape(labels, [-1])  # flatten GT

        loss_nolabeled = tf.reduce_sum(
            (embeddings - c) ** 2 + 1e-6, axis=1
        )  # loss for not labeled samples
        loss_labeled = eta * tf.pow(loss_nolabeled, labels)  # loss for labeled samples

        mask = tf.equal(labels, tf.zeros_like(labels))

        loss_total = tf.where(mask, loss_nolabeled, loss_labeled)
        loss = tf.reduce_mean(loss_total)

        return loss

    return loss_function


def dense_block(inputs: tf.Tensor, units: int, lam: float, name: str) -> Layer:
    """Basic block (Dense-LeakyReLU layers) of multi layer perceptron.

    Parameters
    ----------
    input
        input of block
    units
        number of the units in the Dense layer
    lam
        regularization parameter on the weights in Dense layer
    name
        name of block; "_dense" and "_LeakyReLU" are appended for the layers

    Returns
    -------
    output
        Dense-LeakyReLu layers
    """
    output = Dense(
        units=units,
        use_bias=False,
        activation="linear",
        name=name + "_dense",
        kernel_regularizer=REGULARIZER_L2(lam),
        kernel_initializer=KERNEL_INITIALIZER,
    )(inputs)
    return LeakyReLU(name=name + "_LeakyReLU")(output)


class DeepSAD:
    """Deep SAD Semi-supervised Anomaly Detector.

    Translated from `paper author Lukas Ruff's PyTorch implementation <https://github.com/lukasruff/Deep-SAD-PyTorch>`_
    into TensorFlow.

    .. todo:: more detailed description, including comparison with PyTorch version.

    Attributes
    ----------
    dim_hidden
        from Config
    eta
        from Config
    lam
        from Config
    path_pretrained_model
        from Config
    dim_input
        number of features
    history
    detector

    Original License
    ----------------
    MIT License

    Copyright (c) 2019 lukasruff

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    @dataclasses.dataclass
    class Config:
        """Configuration for DeepSAD model

        Parameters
        ----------
        dim_hidden
            number of units in hidden layers
        eta
            Deep SAD regularization hyperparameter eta (must be 0 < eta)
            balancing the loss for labeled and unlabeled samples
        lam
            regularization parameter on L2-norm of weights
        path_pretrained_model
            path to pretrained model (currently unused)
        """

        dim_hidden: List[int] = dataclasses.field(
            default_factory=lambda: [128, 128, 64, 64, 32]
        )
        eta: float = 1.0
        lam: float = 1e-6
        path_pretrained_model: Optional[str] = None

    def __init__(
        self,
        config: Config,
    ):
        self.dim_hidden = config.dim_hidden
        self.eta = config.eta
        self.lam = config.lam
        self.path_pretrained_model = config.path_pretrained_model

        self.dim_input: Optional[int] = None
        self.history = None
        self.detector: Optional[tf.keras.Model] = None

    def _build_encoder(self) -> tf.keras.Model:
        """Build encoder model (Multi Layer Perceptron)

        Returns
        -------
        encoder
            input: (dim_input, ), output: (dim_hidden[-1], )
        """

        inputs = Input(shape=(self.dim_input,), sparse=False, name="encoder_input")
        outputs = inputs
        for i, dim in enumerate(self.dim_hidden[:-1]):
            outputs = dense_block(
                outputs, units=dim, lam=self.lam, name="encoder_block" + str(i + 1)
            )

        outputs = Dense(
            units=self.dim_hidden[-1],
            use_bias=False,
            activation="linear",
            name=LAYERNAME_ENCODER_OUTPUT,
            kernel_regularizer=REGULARIZER_L2(self.lam),
            kernel_initializer=KERNEL_INITIALIZER,
        )(outputs)

        encoder = tf.keras.Model(inputs=inputs, outputs=outputs)

        return encoder

    def _build_autoencoder(
        self, encoder_input: tf.Tensor, encoder_output: tf.Tensor
    ) -> tf.keras.Model:
        """Build autoencoder model

        Parameters
        ----------
        encoder_input
            input of the first layer of the encoder
        encoder_ouput
            output of the last layer of the encoder (bottleneck layer)

        Returns
        -------
        autoencoder
            input: (dim_input, ), output: (self.dim_input, )
        """

        outputs = LeakyReLU(name="decoder_1st_LeakyReLU")(encoder_output)
        for i, dim in enumerate(self.dim_hidden[::-1][1:]):
            outputs = dense_block(
                outputs, units=dim, lam=self.lam, name="decoder_block" + str(i + 1)
            )

        outputs = Dense(
            units=self.dim_input,
            use_bias=False,
            activation="linear",
            name="decoder_output",
            kernel_regularizer=REGULARIZER_L2(self.lam),
            kernel_initializer=KERNEL_INITIALIZER,
        )(outputs)

        autoencoder = tf.keras.Model(inputs=encoder_input, outputs=outputs)

        return autoencoder

    def _build_detector(
        self, encoder: tf.keras.Model, center: np.ndarray
    ) -> tf.keras.Model:
        r"""Build anomaly detector model

        Parameters
        ----------
        encoder
        center
            center of embeddings (encoded feature) ("c" in the paper)

            :shape: (dim_embedding, )

        Returns
        -------
        detector
            anomaly detector
            detector(x) = \| encoder(x) - c \|^2
            the higher, the more anomalous
        """
        assert center.ndim == 1

        score = tf.reduce_sum((encoder.output - tf.constant(center)) ** 2, axis=1)
        detector = tf.keras.Model(inputs=encoder.input, outputs=score)

        return detector

    @dataclasses.dataclass
    class TrainConfig:
        """Configuration of training process.

        Parameters
        ----------
        epochs_pretrain
            epochs for pretraining (center initialization)
        epochs_train
            epochs for training of detector
        learning_rate
            learning rate of optimizer
        batch_size
            batch size
        """

        epochs_pretrain: int = 10
        epochs_train: int = 20
        learning_rate: float = 1e-3
        batch_size: int = 64

    def train(
        self,
        X: Union[np.ndarray, csr_matrix],
        y: np.ndarray,
        path_model: str,
        config: TrainConfig,
        verbose: int = 1,
    ):
        r"""Train anomaly detector (self.detector) with encoder (local variable).

        Set self.detector, self.dim_input and self.history.
        Save encoder to path_model and loss-epoch plot next to it.

        Parameters
        ----------
        X
            feature matrix

            :shape: (n_samples, n_features)
        y
            label
                0
                    not labeled as normal
                1
                    labeled as normal

            :shape: (n_samples, )
        path_model
            path '\*\*.h5' to save trained model
        verbose
            verbosity of logging/output
        """
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

        os.makedirs(os.path.dirname(path_model), exist_ok=True)

        self.dim_input = X.shape[-1]
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

        # training autoencoder for the weight and center initialization
        if self.path_pretrained_model is None:
            logger.info(
                "start detector weight initialization with epochs %s",
                config.epochs_pretrain,
            )
            encoder = self._build_encoder()
            autoencoder = self._build_autoencoder(
                encoder.input, encoder.layers[-1].output
            )
            autoencoder.compile(optimizer=optimizer, loss="mse")

            if isinstance(X, csr_matrix):
                generator = generator_autoencoder_training(X, config.batch_size)
                autoencoder.fit(
                    generator, epochs=config.epochs_pretrain, verbose=verbose
                )
            elif isinstance(X, np.ndarray):
                autoencoder.fit(X, X, epochs=config.epochs_pretrain, verbose=verbose)

        else:
            logger.info("load pre-trained detector from %s", self.path_pretrained_model)
            detector = tf.keras.models.load_model(self.path_pretrained_model)
            encoder = tf.keras.Model(
                inputs=detector.input,
                outputs=detector.get_layer(LAYERNAME_ENCODER_OUTPUT).output,
            )

        center = encoder.predict(X).mean(axis=0)
        loss = loss_sad(c=center, eta=self.eta)
        callbacks = [
            ModelCheckpoint(
                filepath=path_model,
                monitor="loss",
                verbose=0,
                save_best_only=True,
                mode="auto",
            )
        ]

        # encoder training
        logger.info("start detector training with epochs %s", config.epochs_train)
        encoder.compile(optimizer=optimizer, loss=loss)
        self.history = encoder.fit(
            X,
            y,
            epochs=config.epochs_train,
            initial_epoch=0,
            callbacks=callbacks,
            verbose=verbose,
        )
        encoder.load_weights(path_model)

        # build anomaly detector from the trained encoder and center
        self.detector = self._build_detector(encoder, center)
        self.detector.save(path_model)
        logger.info("save detector on %s", path_model)

        # plot and save training process for debugging
        plt.plot(self.history.history["loss"], marker="o", label="training loss")
        # plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
        plt.title("detector training process")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(path_model), "training_process.png"))
        plt.close()

    def load_detector(self, path_model: str):
        """Load pre-trained anomaly detector"""

        self.detector = load_model(path_model)

    def compute_anomaly_score(
        self, X: Union[np.ndarray, csr_matrix], scale=True
    ) -> np.ndarray:
        """Compute anomaly score

        Parameters
        ----------
        X
            :shape: (n_samples, n_features)
        scale
            scale anomaly scores

        Returns
        -------
        score : ndarray
            anomaly scores

            :shape: (n_samples, )
        """
        # Without type annotation ": ndarray" after score, sphinx treats "score" as type.
        # some text and a blank line is needed before :shape: too.

        if self.detector is None:
            raise AttributeError("detector is not set")
        score = self.detector.predict(X)
        if not scale:
            return score

        # scale anomaly score
        med = np.median(score)
        var = np.median(np.abs(score - med))
        if var == 0:
            var = med
        return (score - med) / var

    def compute_embeddings(
        self, X: Union[np.ndarray, csr_matrix]
    ) -> Optional[np.ndarray]:
        """Compute input embeddings (latent representation/output of bottleneck layer)

        Parameters
        ----------
        X
            :shape: (n_samples, n_features)

        Returns
        -------
        feature : ndarray
            embedding for each input

            :shape: (n_samples, dim_embedding)
        """

        detector = self.detector
        if detector is None:
            raise AttributeError("detector is not set")
        if X.shape[0] == 0:
            return None

        encoder = tf.keras.Model(
            inputs=detector.input,
            outputs=detector.get_layer(LAYERNAME_ENCODER_OUTPUT).output,
        )
        return encoder.predict(X)

    def explain_anomaly(
        self,
        X_anomaly: Union[np.ndarray, csr_matrix],
        background_samples: Union[np.ndarray, csr_matrix],
        zero_correction=True,
        shapvalue_scale=True,
    ):
        """Compute Shapley values (degree of contribution to anomaly) of each feature for anomaly samples


        Parameters
        ----------
        X_anomaly
            feature matrix of anomaly samples

            :shape: (n_anomaly_samples, n_features)
        background_samples
            background samples used to compute Shapley values,
            typically randomly sampled from training set

            :shape: (n_background_samples, n_features)
        zero_correction: bool
            set Shapley value to zero if the corresponding feature is zero
        shapvalue_scale: bool
            scale Shapley values into [1,Inf) (just for simplicity)

        Returns
        -------
        Shapley values
            :shape: (n_anomaly_samples, n_features)

        Notes
        -----
        Uses `SHAP by Scott Lundberg <https://github.com/slundberg/shap>`_.
        """

        if isinstance(X_anomaly, csr_matrix):
            X_anomaly = X_anomaly.toarray()

        num_background_samples = background_samples.shape[0]

        if num_background_samples >= 100:
            idx = random.sample(range(num_background_samples), 100)
            background_samples = background_samples[idx]

        if isinstance(background_samples, csr_matrix):
            background_samples = background_samples.toarray()

        explainer = shap.GradientExplainer(self.detector, background_samples)
        # explainer = shap.DeepExplainer( self.detector, background_samples )
        # DeepExplainer is not available

        shap_values = explainer.shap_values(X_anomaly)

        if zero_correction:
            shap_values[X_anomaly == 0] = 0

        if shapvalue_scale:
            zero_mask = X_anomaly != 0
            shap_mins = (shap_values * zero_mask).min(axis=-1, keepdims=True)
            shap_values = shap_values - shap_mins + 1
            shap_values[~zero_mask] = 0

        return shap_values


def detection_report(
    score_sorted: Series,
    shap_value_idx_sorted: DataFrame,
    shap_top_k: int = 5,
) -> DataFrame:

    """detection report

    Parameters
    ----------
    score_sorted
        anomaly score, sorted in descending order

        :index:
            (datetime_rounded, src_ip)
    shap_value_idx_sorted
        Shapley values of anomaly samples, sorted in descending order by anomaly score

        :index:
            (datetime_rounded, src_ip), top-n of score_sorted
        :columns:
            features
    shap_top_k
        number of Shapley values to include per (datetime_rounded, src_ip)

    Returns
    -------
    detection_report

        :index:
            (datetime_rounded, src_ip)
        :columns:
            anomaly_score, shap_top_{i}, top_{i}_shap_value
            for 0 < i <= shap_top_k
    """
    logger.info(score_sorted.index)
    logger.info(shap_value_idx_sorted.index)

    shap_top_k = min(shap_top_k, shap_value_idx_sorted.shape[-1])

    columns = [
        ["shap_top_" + str(k + 1), "top_" + str(k + 1) + "_shap_value"]
        for k in range(shap_top_k)
    ]
    columns = sum(columns, [])
    dtypes = dict(zip(columns, ["str", "float"] * shap_top_k))
    df_shap = pd.DataFrame(
        0,
        index=score_sorted.index,
        columns=columns,
    ).astype(dtypes)

    for i, sample in enumerate(shap_value_idx_sorted.index):
        shap_values = shap_value_idx_sorted.loc[sample].sort_values(ascending=False)
        fe: List[Union[int, str]] = [
            "__".join(l) for l in list(shap_values.index[:shap_top_k])
        ]
        value = list(shap_values.iloc[:shap_top_k])
        for k in range(shap_top_k):
            if value[k] == 0:
                fe[k] = 0

        df_shap.iloc[i, np.arange(0, shap_top_k * 2, 2)] = fe
        df_shap.iloc[i, np.arange(1, shap_top_k * 2, 2)] = value

    df_shap.insert(0, "anomaly_score", score_sorted)
    return df_shap
