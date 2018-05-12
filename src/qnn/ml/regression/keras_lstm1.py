from typing import TYPE_CHECKING, Optional, List, Dict
from abc import ABC, ABCMeta, abstractmethod

import pandas as pd
import numpy as np

from keras.layers import Input, LSTM, Dense, BatchNormalization, Activation, Dropout, Flatten, concatenate
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import initializers

from qnn.core.parameters import ParametersNode, IntParameter, FloatParameter
from qnn.core import EventHook
from .base import IRegressionMLModel
from ..data_provider import IMLDataProvider


class KerasLSTM1(IRegressionMLModel):
    @staticmethod
    def get_parameters_template() -> ParametersNode:
        return ParametersNode({
            'lstm_units': IntParameter(64),
            'dense_dim': IntParameter(1024),
            'num_dense_layers': IntParameter(3),
            'epochs': IntParameter(10),
            'batch_size': IntParameter(32),
            'learning_rate': FloatParameter(0.001),
            'clipnorm': FloatParameter(1.0),
        })

    def __init__(self, parameters: ParametersNode):
        super().__init__(parameters)

    def _fit(self, training_data_provider: IMLDataProvider = None):
        # Parameters
        lstm_units = self.parameters['lstm_units'].v
        dense_dim = self.parameters['dense_dim'].v
        num_dense_layers = self.parameters['num_dense_layers'].v
        epochs = self.parameters['epochs'].v
        batch_size = self.parameters['batch_size'].v
        learning_rate = self.parameters['learning_rate'].v
        clipnorm = self.parameters['clipnorm'].v

        # Create input placeholders and layers
        inputs_phs = {}
        inputs_x = {}
        for k, v in training_data_provider.input_shapes.items():
            inputs_phs[str(k)] = Input(shape=v, dtype='float32', name='in_' + str(k))
            inputs_x[str(k)] = LSTM(lstm_units, return_sequences=True, activation='relu', kernel_initializer='he_normal')(inputs_phs[str(k)])
            inputs_x[str(k)] = Dropout(0.2)(inputs_x[str(k)])
            inputs_x[str(k)] = LSTM(lstm_units, return_sequences=True, activation='relu', kernel_initializer='he_normal')(inputs_x[str(k)])
            inputs_x[str(k)] = Flatten()(inputs_x[str(k)])

        # Concatenate output of input layers
        if len(inputs_x) > 1:
            x = concatenate([v for v in inputs_x.values()])
        else:
            x = list(inputs_x.values())[0]

        x = Dropout(0.2)(x)

        for i in range(num_dense_layers):
            x = Dense(dense_dim, use_bias=False, kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            if i != num_dense_layers - 1:
                x = Dropout(0.5)(x)

        # Create output layers
        outputs_x = {}
        self.__outputs_list = []
        self.__output_keys_list = []
        for k, v in training_data_provider.target_shapes.items():
            outputs_x[str(k)] = Dense(1, name='out_' + str(k), kernel_initializer='he_normal')(x)

            self.__outputs_list.append(outputs_x[str(k)])
            self.__output_keys_list.append(str(k))

        # Create and compile the model
        self.model = Model(inputs=list(inputs_phs.values()), outputs=self.__outputs_list)
        self.model.compile(RMSprop(lr=learning_rate, clipnorm=clipnorm), loss={'out_' + str(k): 'mean_squared_error' for k in training_data_provider.target_shapes.keys()})

        # Fit the model
        self.model.fit({'in_' + str(k): v for k, v in training_data_provider.get_all_training_samples()[0].items()},
                       {'out_' + str(k): v for k, v in training_data_provider.get_all_training_samples()[1].items()},
                       validation_data=({'in_' + str(k): v for k, v in training_data_provider.get_all_val_samples()[0].items()},
                                        {'out_' + str(k): v for k, v in training_data_provider.get_all_val_samples()[1].items()}),
                       epochs=epochs,
                       batch_size=batch_size)

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
        preds = self.model.predict({'in_' + k: v for k, v in inputs.items()})
        if len(self.__output_keys_list) > 1:
            return {k: [v for v in values] for k, values in zip(self.__output_keys_list, preds)}
        else:
            return {self.__output_keys_list[0]: [v[0] for v in preds]}
