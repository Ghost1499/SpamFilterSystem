import imaplib
import email
import shlex
from email.parser import Parser
from email.message import EmailMessage
from typing import List

import html2text

from email_system.NecessaryEmail import NecessaryEmail

from email_system.MailSystem import MailSystem
from email_system.utils.imap_encoding import imaputf7decode
from email_system.errors import *

class MailReceiver(MailSystem):
    """description of class"""
    mail: imaplib.IMAP4_SSL

    @property
    def folder(self):
        return self._folder


    def __init__(self, server, port, login, password):
        MailSystem.__init__(self, server, port, login, password)
        self._text_maker = html2text.HTML2Text()
        self._text_maker.ignore_links = True
        self._text_maker.bypass_tables = False
        self._text_maker.skip_internal_links = True
        self._text_maker.ignore_images = True
        self._text_maker.ignore_tables = True
        self._text_maker.ignore_emphasis = True

        self._folder=None
        self._parser = email.parser.BytesParser(EmailMessage)

    def _delete(self):
        self.mail.logout()

    def _connect_to_server(self, server, port) -> imaplib.IMAP4_SSL:
        return imaplib.IMAP4_SSL(server, port)

    def select(self,folder):
        try:
            # _,folders=self.mail.list()
            # folders=[shlex.split(imaputf7decode(folder.decode()))[-1] for folder in folders]
            # if folder in folders:
            if self._folder!=None:
                self.mail.close()
            res,_= self.mail.select(folder)
            if res=="OK":
                self._folder=folder
            else:
                raise SelectFolderError(folder, "Данной папки не существует.")
        except SelectFolderError:
            raise
        except imaplib.IMAP4.error:
            raise SelectFolderError("Ошибка выбора папки.")

    def get_all(self):
        return self.by_uid(self.get_uids())

    def by_uid(self, uids):
        """
        Извлекает письма из текущей выбранной папки с сервера по переданным uid

        :type uids: List[int]
        """
        email_messages=self._raw_by_uids(uids)
        emails: List[NecessaryEmail] = []

        for i,uid in enumerate(uids):
            necessary_email = NecessaryEmail(email_messages[i], uid, self._text_maker)
            emails.append(necessary_email)

        return emails

    def get_uids(self):
        """
        Возвращает список uid всех писем в текущей выбранной папке сервера

        :rtype: List[int]
        """
        mail = self.mail
        # type,data=self.mail.recent()
        type, data = mail.uid("search", None, "all")
        emails_uids = [uid.decode() for uid in data[0].split()]
        return emails_uids

    def _raw_by_uids(self, uids:List[str]):
        email_messages:List[EmailMessage]=[]
        for uid in uids:
            result, data = self.mail.uid('fetch', uid, '(RFC822)')
            if result != "OK":
                raise EmailUidError("Письма с заданным uid не существует",self._folder,uid=uid)
            raw_email = data[0][1]
            email_messages.append(self._parser.parsebytes(raw_email))
        return email_messages
