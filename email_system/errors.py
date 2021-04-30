class EmailSystemError(Exception):
    pass


class ConnectionError(EmailSystemError):
    pass


class LoginError(EmailSystemError):
    pass


class SelectFolderError(EmailSystemError):
    def __init__(self, folder: str, message: str = None):
        self.message = message
        self.folder = folder

    def __str__(self):
        if self.message:
            return f"{self.message} Папка - {self.folder}"


class EmailUidError(EmailSystemError):
    def __init__(self,message:str,folder:str,uid:str):
        self.uid = uid
        self.folder = folder
        self.message = message

    def __str__(self):
        return f'{self.message} Папка - {self.folder}. Uid - {self.uid}.'
        
