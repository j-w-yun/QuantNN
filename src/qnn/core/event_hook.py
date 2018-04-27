from typing import TypeVar, Generic

HandlerCallableType = TypeVar('T')


class EventHook(Generic[HandlerCallableType]):
    def __init__(self):
        self.__handlers = []

    def __iadd__(self, handler: HandlerCallableType):
        self.__handlers.append(handler)
        return self

    def __isub__(self, handler: HandlerCallableType):
        self.__handlers.remove(handler)
        return self

    def __call__(self, *args, **kwargs):
        for handler in self.__handlers:
            handler(*args, **kwargs)
