from spamfilter.SpamFilter import SpamFilter
from email_system.MailReceiver import MailReceiver
import config_app as config
receiver=MailReceiver(config.server_imap,config.port,config.login,config.password)
spam_filter=SpamFilter(receiver,config.spam_folder,config.ham_folder)
receiver.select("INBOX")
result=spam_filter.classify(receiver.get_all())