from abc import ABC, abstractmethod

class _DataGenerator(ABC):


    @abstractmethod
    def _check_input_data(self):
        pass

    @abstractmethod
    def _generate_data(self):
        pass

    def _check_generated_data(self):
        """
        want to be able to use the DataLoaders method for this since they are the same...
        maybe make that a function?
        :return:
        """
        pass

class Twiss_data_generator(_DataGenerator):

    def __init__(self, x_twiss, y_twiss, E, input_data):
        super().__init__(input_data)

    def _import_data(self):
        pass

    def _check_input_data(self):
        pass
