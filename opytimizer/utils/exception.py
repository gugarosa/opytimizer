class ArgumentException(Exception):
    def __init__(self, argument):
        super().__init__("Missing input argument. Expects: " + "'" + argument + "'.")

class ParameterException(Exception):
    def __init__(self, parameter):
        super().__init__("Missing optimizer parameter. Expects: " + "'" + parameter + "'.")
