from typing import List

from .classifiers import LSTMClassifier,Classifier
from .Extractor import Extractor
from email_system.NecessaryEmail import NecessaryEmail

class SpamFilter(object):
    def __init__(self,recieve_mail_system,spam_folder,ham_folder):
        self.extractor=Extractor(recieve_mail_system)
        self.extractor.extract(spam_folder, ham_folder)
        self.df=self.extractor.get_dataframe()
        self._navec_path="navec_hudlit_v1_12B_500K_300d_100q.tar"
        # self._dataset_path="myspam.csv"
        # self.df=pd.read_csv(self._dataset_path)
        self.subject_classifier:Classifier=LSTMClassifier.LSTMClassifier(self._navec_path)
        self.subject_classifier.set_up(embedding_data=self.df['subject'],x=self.df['subject'],y=self.df["label"])

    def classify(self,emails:List[NecessaryEmail]):
        classified=[]
        for email in emails:
            classified.append([email, self.subject_classifier.get_predictions(email.subject)])# добавить другие результаты классификации
        return classified
# def main():
#     #mail_sender=SendMailSystem()
#     subject="Letter to send"
#     body="This is letter to send"
#     reciever_address=config.reciever_login
#     #mail_sender.send_mail(subject,body,reciever_address,None)
#     #mail_reciever=RecieveMailSystem()
#     #mail_reciever.recieve_mail()
#     system_controller= SystemController()
#     csv_filename= "../spam_ham_dataset.csv"
#     system_controller.train_from_file(csv_filename)
#
# if __name__=="__main__":
#     main()