
class ForceError:
    def __init__(self, message):
        self.message = message
    
    def force(self):
        raise ValueError(self.message)
