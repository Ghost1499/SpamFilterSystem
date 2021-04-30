import smtplib
import imaplib
from abc import ABCMeta, abstractmethod

from email_system.errors import *


class MailSystem(object):
    """description of class"""

    def __init__(self, server, port, login, password):
        self.server = server
        self.port = port
        self.login = login
        self.password = password
        self._connect(self.server, self.port)
        self._login(self.mail, self.login, self.password)

    def __del__(self):
        self._delete()

    @abstractmethod
    def _delete(self):
        pass

    def _connect(self, server, port):
        try:
            self.mail = self._connect_to_server(server, port)
        except:
            raise ConnectionError("Ошибка подключения к серверу")

    @abstractmethod
    def _connect_to_server(self, server, port):
        pass

    def _login(self, mail, login, password):
        try:
            mail.login(login, password)
        except:
            raise LoginError("Ошибка авторизации на сервере")
