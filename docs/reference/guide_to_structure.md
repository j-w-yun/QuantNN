# Guide to QNN structure
## Parameter nodes
Many objects/models have hyperparameters, I refer to those as just parameters.

It makes sense to explicity specify what kind of parameters (with default values) are expected for some classes (such as machine learning models).

To do this there is the `ParametersNode` class. All models that have hyperparameters should have a `get_parameters_template` method to return the default hyperparameters (as a `ParametersNode` object).

For example:
  - ML models like regressors or classifiers can have (hyper)parameters specifying learning-rate and number of layers etc..
  - The KerasLSTM1 regression ML model has hyperparameters for number of lstm unit, number of layers, etc...

Example of using `ParametersNode`:
```py
class KerasLSTM1(IRegressionMLModel):
    # Use a static method get_parameters_template to return the default parameters.
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

    # The class constructor takes a ParametersNode as argument. This can be the defaults or an altered version of the defaults.
    def __init__(self, parameters: ParametersNode):
        super().__init__(parameters)
```

`ParameterNode` objects have to_dict() and from_dict(dict) methods for serializing and deserializing parameters.
These two methods will also serialize/deserialize what type the parameter is and optionally the maximum and minimum values.
This can be useful, but often you will want to only serialize the values of the parameters. Then you can use to_values_dict().
To restore a ParametersNode with only a values dict you need to call a class its get_parameters_template() method and update it, like so:
```py
parameters = SomeClass.get_parameters_template()
parameters.update_from_values_dict(my_values_dict)
```

It is possible to have a ParameterNode contain parameters of another ParameterNode object.
Especially useful is `ModelChoiceParameter`:
```py
@staticmethod
def get_parameters_template() -> ParametersNode:
  return ParametersNode({
      'ml_model': ModelChoiceParameter.from_models_map(REGRESSION_ML_MODELS_MAP),
      'input_sequence_length': IntParameter(31),
      'val_split': FloatParameter(0.05),
  })
```
In this example the `ml_model` parameter allows specifying which regression ML model to use and with what parameters.
Note that `REGRESSION_ML_MODELS_MAP` is a dict with string as key and class types (not instances, just the classes) as values; of course, all classes in that map should have a static `get_parameters_template()` method.

This explicit specification of (hyper)parameters becomes especially useful when creating a GUI dialog for configuring models. It is also useful when building an algorithm to search over hyperparameters automatically (grid search, bayeseian optimization, genetic optimization, etc... can all work with this structure/hiearchy of parameters).

## Regression targets
Regression targets specify what *regression models* should predict. There can be all types of regression targets, and each one is one class. Each regression target should have a *get_parameters_template* method and a *generate_target* method.

As we'll see later, **regression models** take regression targets as a constructor argument.

## Regression problems
Regression problems are what regression models as benchmarked on in the `traintest_regression_model` program. This program also takes a regression model as argument, more about that later.

Regression problems specify the following:
  - timeframe
  - symbols (to use as input)
  - target symbols (to predict)
  - train range
  - test range
  - regression target name (what regression target to use)
  - regression target parameters

With this information we can construct a problem that a regression model then can solve; and this is done in the `traintest_regression_model` program.

## Regression models
Regression models (not to be confused with regression ML models (see below)) are given a dict of regression targets and market data, as well as what timeframe to use for predictions and a list of symbols to be used as input to the model.

All regression models have a `get_parameters_template` method, a fit method and a predict method. The fit and predict methods with with market data tables as input data. Regression models can internally use regression ML models; these take a dict of numpy matrices as input (and as target in their fit() method).

So the difference between regression models and regression ML models is that regression models operate on market data tables whereas regression ML models are more low-level and work on numpy matrices.  A regression model usually processes data from the market data tables to a type is that digestible for a regression ML model. The advantage of having separate ML models is that we can now create a regression model and use a `ModelChoiceParameter` to have it work with *any* regression ML model. This makes testing with different kinds of ML models more straightforward, as we'd just change the parameter value in the regression model.

## Regression ML models
*TODO*

## Putting it all together
*TODO*
