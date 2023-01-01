class InvalidInputForcing(Exception):
    """
    Exception raised when input forcing not applicable with other given input parameters.

    Parameters
    ----------
    error_text: str
        error message which is printed when the error is raised.
    """

    def __init__(self, error_text):
        self.error_text = error_text
