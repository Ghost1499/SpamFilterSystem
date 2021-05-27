import email
import email.header
import string
from email.message import Message, EmailMessage
import html2text
import email.contentmanager
import regex as re


class NecessaryEmail(object):
    subject_enc = ""
    subject = ""
    prepared_subejct = ""
    body_enc = ""
    body = ""
    prepared_body = ""
    is_spam = None
    def __init__(self, email_message: EmailMessage, uid: str, text_maker: html2text.HTML2Text,is_spam=None):
        self.is_spam = is_spam
        self.email_message = email_message
        self.uid = uid
        self._text_maker = text_maker


        try:
            self.subj_enc = email_message['subject']
            self.subject = self._decode_subject(self.subj_enc)
            if self.subject:
                self.prepared_subejct = self.get_prepared_subject()
        except Exception:
            pass
            # raise Exception("Ошибка извлечения темы письма")
        try:
            self.body_enc = email_message.get_body(preferencelist=["plain", "html"])
            if self.body_enc:
                self.body = self.extract_body(self.body_enc)
                self.prepared_body = self.get_prepared_body()
        except Exception as ex:
            print(ex)
            # raise Exception("Ошибка извлечения тела письма")

    @staticmethod
    def _decode_subject(subject):
        while subject.startswith("FWD:") or subject.startswith("Fwd:"):
            if subject.startswith("FWD:"):
                subject = subject.lstrip("FWD:")
            if subject.startswith("Fwd:"):
                subject = subject.lstrip("Fwd:")
        return NecessaryEmail.decode_header(subject)

    @staticmethod
    def decode_header(header: str):
        parts = []
        for part in email.header.decode_header(header):
            header_string, charset = part
            if charset:
                decoded_part = header_string.decode(charset)
            else:
                decoded_part = header_string
            parts.append(decoded_part)
        return "".join(parts)

    def get_prepared_subject(self):
        return NecessaryEmail.lstrip_subject(self.subject)

    def get_prepared_body(self):
        return self.clean_up_text(self.body)

    @staticmethod
    def lstrip_subject(subject: str):
        if subject.startswith("FWD:"):
            subject = subject.lstrip("FWD:")
        if subject.startswith("Fwd:"):
            subject = subject.lstrip("Fwd:")
        if subject.startswith("Re:"):
            subject = subject.lstrip("Re:")
        if subject.startswith("RE:"):
            subject = subject.lstrip("RE:")
        return subject.lstrip()

    def get_text_from_html(self, html: str):
        text: bytes = self._text_maker.handle(html)
        return text

    def extract_body(self, body):
        content_subtype = body.get_content_subtype()
        text: str = email.contentmanager.raw_data_manager.get_content(body)
        if content_subtype == "html":
            text = self.get_text_from_html(text)
        return text

    @staticmethod
    def clean_up_text(text: str) -> str:
        text = re.sub(r'[^\w\s' + string.punctuation + ']', '', text)
        text: str = re.sub(r'[\n‌‌]+', '/n', text)
        text: str = re.sub(r'[\s‌‌]+', ' ', text)
        splited_lines = text.split("/n")
        new_splited_lines = []
        for line in splited_lines:
            if line == "-------- Исходное сообщение --------":
                break
            if line == " ":
                line = ""
            if line != "":
                new_splited_lines.append(line)

        if "Пересылаемое сообщение" in new_splited_lines[0]:
            for i, line in enumerate(reversed(new_splited_lines)):
                if "Конец пересылаемого сообщения" in line:
                    text = " ".join(new_splited_lines[2:-(i + 1)])
                    break
                if i > 10:
                    text = " ".join(new_splited_lines[2:])
                    break
        else:
            text = " ".join(new_splited_lines)
        return text
