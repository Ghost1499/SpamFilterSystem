import email
import email.policy as epolicy
import mailbox
import os
from typing import List
import csv

import pandas as pd
from progress.bar import IncrementalBar

from email_system.NecessaryEmail import NecessaryEmail
from email_system.MailReceiver import MailReceiver, get_text_maker
from spamfilter.classifiers.utils import int2label
from email_system.errors import *
from spamfilter.errors import *


class Extractor(object):
    recieve_mail_system: MailReceiver
    ham: List[NecessaryEmail]
    spam: List[NecessaryEmail]

    def __init__(self, recieve_mail_system: MailReceiver):
        self.spam_filename = "spam_uids.txt"
        self.ham_filename = "ham_uids.txt"
        self.recieve_mail_system = recieve_mail_system
        self.text_maker = get_text_maker()
        self.spam_uids = self._read_uids(self.spam_filename)
        self.ham_uids = self._read_uids(self.ham_filename)

    def from_mbox(self, filename, is_spam=None):
        mbox = mailbox.mbox(filename,factory=lambda f: email.message_from_binary_file(f, policy=epolicy.default))
        for message in mbox:
            yield NecessaryEmail(message, None, self.text_maker, is_spam)

    # def from_csv(self,filename):
    #     df=pd.read_csv(filename)
    #     for message in df.iloc:
    #         yield NecessaryEmail(None)
    def from_email_folder(self, folder, uids, is_spam: bool):
        try:
            self.recieve_mail_system.select(folder)
            existed_uids = uids
            new_uids = set(self.recieve_mail_system.get_uids())
            new_uids.difference_update(existed_uids)
            # mails=[]
            # bar=IncrementalBar('Countdown',max=len(new_uids))
            for mail in self.recieve_mail_system.by_uid(list(new_uids)):
                mail.is_spam = is_spam
                yield mail
                # mails.append(mail)
            # bar.finish()
            return new_uids
        except SelectFolderError:
            raise ExtractionError("Ошибка извлечения писем с сервера")

    # def extract(self, spam_folder, ham_folder):
    #     self.spam, self.spam_uids = self.from_email_folder(spam_folder, self.spam_uids)
    #     for mail in self.spam:
    #         mail.is_spam = True
    #     print("Spam extracted")
    #     self.ham, self.ham_uids = self.from_email_folder(ham_folder, self.ham_uids)
    #     for mail in self.ham:
    #         mail.is_spam = False
    #     print("Ham extracted")

    @staticmethod
    def from_excel(filename):
        try:
            return pd.read_excel(sheet_name=filename)
        except FileNotFoundError as ex:
            raise FileNotFoundError(ex, message=f"Файл {filename} для считывания набора писем не найден")

    @staticmethod
    def get_dataframe(mails) -> pd.DataFrame:
        """
        Возвращает DataFrame, созданный из извлеченных писем

        :return: DataFrame писем
        :rtype: pd.DataFrame
        """
        # mails: List[NecessaryEmail] = spam + ham
        if not mails:
            return None
        subjects = []
        texts = []
        labels = []
        uids = []

        for mail in mails:
            subjects.append(mail.prepared_subejct)
            texts.append(mail.prepared_body)
            if mail.is_spam is not None:
                labels.append(int2label[int(mail.is_spam)])
            else:
                labels.append("")
            uids.append(mail.uid)
        data = {'uid': uids, 'label': labels, 'subject': subjects, 'text': texts}
        df = pd.DataFrame(data=data)
        return df

    def save_to_csv(self, spam, ham, filename: str):
        self._write_uids(self.spam_filename, self.spam_uids)
        self._write_uids(self.ham_filename, self.ham_uids)
        ind = os.path.exists(filename)
        with open(filename, 'a', newline="", encoding='utf-8') as file:
            writer = csv.writer(file)
            if not ind:
                writer.writerow(["uid", "label", "subject", "text"])
            mails = self.spam + self.ham
            for mail in mails:
                writer.writerow([mail.uid, int2label[int(mail.is_spam)], mail.prepared_subejct, mail.prepared_body])

    @staticmethod
    def _write_uids(filename, uids):
        """

        :type filename: str название файла для записи uids
        :type uids: List[int] список uids
        """
        with open(filename, 'a') as file:
            file.write(" ".join(uids) + " ")

    @staticmethod
    def _read_uids(filename):
        if os.path.exists(filename):
            with open(filename, "r") as file:
                text = file.read()
            return [uid for uid in text.split()]
        return []

# def main():
#     mail_system = RecieveMailSystem(config.server_imap, config.port, config.login, config.password)
#     _, folders_list = mail_system.mail.list()
#     for folder in folders_list:
#         print(shlex.split(imaputf7decode(folder.decode()))[-1])
#
#     emails = mail_system.from_folder(config.folder)
#     filename="myspam.txt"
#     with open(filename,'w',newline="",encoding='utf-8') as file:
#         writer=csv.writer(file)
#         writer.writerow(["uid","subject","text"])
#         for mail in emails:
#             writer.writerow([mail.uid,mail.subject,mail.text])
