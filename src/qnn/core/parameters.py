from typing import Dict, List, Optional, Type
from abc import ABC, ABCMeta, abstractmethod
import copy

PARAMETER_TYPES: Dict[str, Type['IParameter']] = {}


def register_parameter_type(class_: Type['IParameter']):
    PARAMETER_TYPES[class_.__name__] = class_


class ParametersNode(object):
    def __init__(self, parameters: Dict[str, 'IParameter']=None):
        self.__parameters: Dict[str, 'IParameter'] = parameters or {}

    @property
    def parameters(self) -> Dict[str, 'IParameter']:
        return self.__parameters

    def __len__(self):
        return len(self.__parameters)

    def __getitem__(self, item):
        return self.__parameters[item]

    def __setitem__(self, key, value):
        assert isinstance(key, str)
        assert isinstance(value, IParameter)

        self.__parameters[key] = value

    def to_values_dict(self) -> dict:
        return {
            'parameters': {k: v.to_value() for k, v in self.__parameters.items()},
        }

    def update_from_values_dict(self, d):
        for k, v in d['parameters'].items():
            if k in self.__parameters:
                self.__parameters[k].update_from_value(v)

    def to_dict(self) -> dict:
        return {
            'parameters': {k: {'_type': v.__class__.__name__, **v.to_dict()} for k, v in self.__parameters.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> 'ParametersNode':
        ret = ParametersNode()
        for k, v in d['parameters'].items():
            ret[k] = PARAMETER_TYPES[v['_type']].from_dict(v)
        return ret


class IParameter(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def to_value(self):
        raise NotImplementedError

    @abstractmethod
    def update_from_value(self, v):
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_dict(d: dict) -> 'IParameter':
        raise NotImplementedError


class IntParameter(IParameter):
    def __init__(self, v: int):
        super().__init__()

        self.v = v

    def to_value(self):
        return self.v

    def update_from_value(self, v):
        self.v = int(v)

    def to_dict(self) -> dict:
        return {
            'v': self.v,
        }

    @staticmethod
    def from_dict(d: dict) -> 'IParameter':
        return IntParameter(int(d['v']))


class FloatParameter(IParameter):
    def __init__(self, v: float):
        super().__init__()

        self.v = v

    def to_value(self):
        return self.v

    def update_from_value(self, v):
        self.v = float(v)

    def to_dict(self) -> dict:
        return {
            'v': self.v,
        }

    @staticmethod
    def from_dict(d: dict) -> 'IParameter':
        return FloatParameter(float(d['v']))


class ModelChoiceParameter(IParameter):
    def __init__(self, v: str, choices: Dict[str, ParametersNode], allow_empty: bool=False):
        super().__init__()

        self.v = v
        self.choices = choices
        self.allow_empty = allow_empty

    @property
    def parameters(self):
        return self.choices[self.v]

    def to_value(self):
        return {
            'v': self.v,
            'choices': {k: v.to_values_dict() for k, v in self.choices.items()},
        }

    def update_from_value(self, v):
        self.v = v['v']

        for k, d in v['choices'].items():
            if k in self.choices:
                self.choices[k].update_from_values_dict(d)

    def to_dict(self) -> dict:
        return {
            'v': self.v,
            'choices': {k: v.to_dict() for k, v in self.choices.items()},
            'allow_empty': self.allow_empty,
        }

    @staticmethod
    def from_dict(d: dict) -> 'IParameter':
        return ModelChoiceParameter(d['v'], {k: ParametersNode.from_dict(v) for k, v in d['choices'].items()}, bool(d['allow_empty']))

    @staticmethod
    def from_models_map(map: Dict[str, any], default_v: str=None, allow_empty: bool=False):
        if allow_empty:
            initialv = ''
        else:
            initialv = list(map.keys())[0] if len(map.keys()) != 0 else ''
        return ModelChoiceParameter(initialv, {k: v.get_parameters_template() for k, v in map.items()}, allow_empty)



__PARAMETERS_TEMPLATE_CACHE = {}


def get_parameters_template(Class):
    global __PARAMETERS_TEMPLATE_CACHE

    try:
        return copy.deepcopy(__PARAMETERS_TEMPLATE_CACHE[Class])

    except KeyError:
        pt = Class.get_parameters_template()
        __PARAMETERS_TEMPLATE_CACHE = pt
        return copy.deepcopy(pt)


register_parameter_type(IntParameter)
register_parameter_type(FloatParameter)
register_parameter_type(ModelChoiceParameter)
