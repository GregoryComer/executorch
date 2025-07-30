from executorch.backends.test.harness.error_statistics import ErrorStatistics

class OutputMismatchError(Exception):
    """ An error that is raised when the output of the model does not match the reference output. """
    error_statistics: ErrorStatistics
    
    def __init__(self, message: str, error_statistics: ErrorStatistics):
        super().__init__(message)
        self.error_statistics = error_statistics

class LoadModelError(Exception):
    """ An error that is raised when the model fails to load. """
    pass

class RunModelError(Exception):
    """ An error that is raised when the model fails to run. """
    pass
