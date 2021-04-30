import email
import email.header
import string
from email.message import Message, EmailMessage
import html2text
import email.contentmanager
import regex as re


class NecessaryEmail(object):

    def __init__(self, email_message: EmailMessage, uid: str, text_maker: html2text.HTML2Text):
        try:
            self.email_message = email_message
            self.uid = uid
            self._text_maker = text_maker
            self.subj_enc = email_message['subject']
            self.subject = self._decode_subject(self.subj_enc)
            self.raw_body =email_message.get_body(preferencelist=["plain", "html"])
            if self.raw_body:
                self.body=self.extract_body(self.raw_body)
            self.is_spam=None

            # text = self.extract_text(body)
            # text = self.clean_up_text(text)
            # if text:
            #     self.text = text
            # else:
            #     self.text = None

        except Exception as ex:
            self.subject = None
            self.text = None
            raise ex
            # raise Exception("Subject or text of necessary email didnt parsed")

    def _decode_subject(self,subject):
        while subject.startswith("FWD:") or subject.startswith("Fwd:"):
            if subject.startswith("FWD:"):
                subject = subject.lstrip("FWD:")
            if subject.startswith("Fwd:"):
                subject = subject.lstrip("Fwd:")
        return self.decode_header(subject)

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

    @staticmethod
    def _lstrip_subject(subject: str):
        if subject.startswith("FWD:"):
            subject = subject.lstrip("FWD:")
        if subject.startswith("Fwd:"):
            subject = subject.lstrip("Fwd:")
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

    def clean_up_text(self, text: str):
        text=re.sub(r'[^\w\s'+string.punctuation+']', '',text)
        text: str=re.sub(r'[\n‌‌]+','/n',text)
        text: str=re.sub(r'[\s‌‌]+',' ',text)
        splited_lines = text.split("/n")
        new_splited_lines=[]
        for line in splited_lines:
            if line==" ":
                line=""
            if line!="":
                new_splited_lines.append(line)

        if "Пересылаемое сообщение" in new_splited_lines[0]:
            for i,line in enumerate(reversed(new_splited_lines)):
                if "Конец пересылаемого сообщения" in line:
                    text = " ".join(new_splited_lines[2:-(i+1)])
                    break
                if i>10:
                    text = " ".join(new_splited_lines[2:])
                    break
        else:
            text = " ".join(new_splited_lines)
        return text
