class SpamFilterError(Exception):
    pass


class ExtractionError(SpamFilterError):
    def __init__(self, folder: str, message: str = None):
        self.message = message
        self.folder = folder

    def __str__(self):
        if self.message:
            return f"{self.message} Папка - {self.folder}"