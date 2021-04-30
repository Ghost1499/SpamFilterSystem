import smtplib

from email_system.MailSystem import MailSystem


class MailSender(MailSystem):
    """description of class"""
    mail: smtplib.SMTP_SSL

    def _connect_to_server(self,server,port) ->smtplib.SMTP_SSL:
        return smtplib.SMTP_SSL(server,port)

    def _delete(self):
        self.mail.quit()

    def send_mail(self,subject,body_text, to_addr, from_addr):
        if from_addr==None:
            from_addr=self.login
        BODY = "\r\n".join(("From: %s" % from_addr,
        "To: %s" % to_addr,
        "Subject: %s" % subject ,
        "",
        body_text))
        self.mail.sendmail(from_addr, [to_addr], BODY)

    


