from abc import ABC, abstractmethod


class Module(ABC):

    def __init__(self):
        self.callbacks = []

    def on_upstream(self, callback):
        self.callbacks.append(callback)

    def emit_upstream(self, *data):
        [callback(*data) for callback in self.callbacks]

    def set_goal_downstream(self):
        pass

    @abstractmethod
    def on_goal(self, *args):
        """
            Translate goal, plan ahead/offline, set and or/pass the (sub)goals downstream, etc.
        """
        pass

    @abstractmethod
    def on_observation(self, *args):
        """
            Call actions downstream and/or pass observations upstream
        """
        pass

    @abstractmethod
    def on_feedback(self, *args):
        """
            Update internal models
        """

