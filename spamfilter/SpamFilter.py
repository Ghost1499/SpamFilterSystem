import os
from typing import List

from .classifiers import LSTMClassifier,utils
from .Extractor import Extractor
from email_system.NecessaryEmail import NecessaryEmail

class SpamFilter(object):
    def __init__(self,recieve_mail_system,spam_folder,ham_folder,is_load=False):
        self.is_load=is_load
        self._navec_path="spamfilter/navec_hudlit_v1_12B_500K_300d_100q.tar"
        self.extractor=Extractor(recieve_mail_system)
        spam,spam_uids=self.extractor.from_email_folder(spam_folder,self.extractor.spam_uids)
        ham,ham_uids=self.extractor.from_email_folder(ham_folder,self.extractor.ham_uids)
        # self.extractor.extract(spam_folder, ham_folder)
        self.extractor.save_to_csv(spam,ham,"spam.csv")
        self.df=self.extractor.get_dataframe(spam,ham)
        if  self.df is not None :
            self.df=self.df.sample(frac=1)
            # self._dataset_path="myspam.csv"
            # self.df=pd.read_csv(self._dataset_path)
            self.subject_classifier:utils=LSTMClassifier.LSTMClassifier(self._navec_path)
            self.subject_classifier.fit_tokenizer(self.df['subject'])
            if os.path.exists(self.subject_classifier.model_file) and self.is_load: #and os.path.exists(self.subject_classifier.weights_file):
                self.subject_classifier.load_model()
            else:
                self.subject_classifier.build_model(self.df['subject'])
                self.subject_classifier.save_model()

            self.subject_classifier._compile_model()
            if os.path.exists(self.subject_classifier.weights_file) and self.is_load:
                self.subject_classifier.load_weights()
            loss,accuracy,precision,recall= self.subject_classifier.train(x=self.df['subject'],y=self.df["label"])
            print(f"Loss -- {loss}\nAccuracy -- {accuracy}\nPrecision -- {precision}\nRecall -- {recall}")
            # self.subject_classifier.set_up(embedding_data=self.df['subject'],x=self.df['subject'],y=self.df["label"])
        else:
            print("Новых писем нет")

    def classify(self,emails:List[NecessaryEmail]):
        classified=[]
        self.subject_classifier.fit_tokenizer([email.prepared_subejct for email in emails])
        for email in emails:
            classified.append([email, self.subject_classifier.get_predictions(email.prepared_subejct)])# добавить другие результаты классификации
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