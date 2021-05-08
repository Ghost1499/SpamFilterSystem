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

def get_text_maker():
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True
    text_maker.bypass_tables = False
    text_maker.skip_internal_links = True
    text_maker.ignore_images = True
    text_maker.ignore_tables = True
    text_maker.ignore_emphasis = True
    return text_maker


class MailReceiver(MailSystem):
    """description of class"""
    mail: imaplib.IMAP4_SSL

    @property
    def folder(self):
        return self._folder


    def __init__(self, server, port, login, password):
        MailSystem.__init__(self, server, port, login, password)
        self.text_maker = get_text_maker()

        self._folder=None
        self._parser = email.parser.BytesParser(EmailMessage)

    def _delete(self):
        self.mail.logout()

    def _connect_to_server(self, server, port) -> imaplib.IMAP4_SSL:
        return imaplib.IMAP4_SSL(server, port)

    def select(self,folder):
        try:
            # folders=[shlex.split(imaputf7decode(folder.decode()))[-1] for folder in folders]

            # if self._folder:
            #     self.mail.close()
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

        :type uids: List[str]
        """
        # email_messages=self._raw_by_uids(uids)
        emails: List[NecessaryEmail] = []
        mails=self._raw_by_uids(uids)
        for i,uid in enumerate(uids):
            email_message=next(mails)
            necessary_email = NecessaryEmail(email_message, uid, self.text_maker)
            yield necessary_email
            # emails.append(necessary_email)

        # return emails

    def get_uids(self):
        """
        Возвращает список uid всех писем в текущей выбранной папке сервера

        :rtype: List[str]
        """
        mail = self.mail
        # type,data=self.mail.recent()
        type, data = mail.uid("search", None, "all")
        emails_uids = [uid.decode() for uid in data[0].split()]
        return emails_uids

    def _raw_by_uids(self, uids:List[str]):
        """

        :rtype: EmailMessage
        """
        # email_messages:List[EmailMessage]=[]
        for uid in uids:
            result, data = self.mail.uid('fetch', uid, '(RFC822)')
            if result != "OK":
                raise EmailUidError("Письма с заданным uid не существует",self._folder,uid=uid)
            raw_email = data[0][1]
            yield self._parser.parsebytes(raw_email)
            # email_messages.append(self._parser.parsebytes(raw_email))
        # return email_messages
