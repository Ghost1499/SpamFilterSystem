from abc import abstractmethod,ABCMeta
from ..Tokenizer import Tokenizer


label2int = {"ham": 0, "spam": 1}
int2label = {0: "ham", 1: "spam"}


class Classifier(object):

    _tokenizer:Tokenizer

    def __init__(self,navec_path,embedding_size=300):
        self.embedding_size = embedding_size

    @abstractmethod
    def set_up(self,**kwargs):
        pass

    @abstractmethod
    def get_predictions(self, text: str) -> str:
        pass
