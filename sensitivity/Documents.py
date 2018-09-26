import pandas as pd

class Documents:
    """describe phenotypes documents, one row is an individual record, one column is a phenotype"""


    def __init__(self, df):
        self.df = df

    def search_cocur(self, terms):
        msks = ''
        for idx, t in enumerate(terms):
            try:
                if idx == 0:
                    msks = self.df[t] > 0
                else:
                    msk = self.df[t] > 0
                    msks = pd.concat((msks, msk), axis=1)

            except:
                return 0
        if idx<1:
            n_cocurr = self.df[msks].shape[0]
        else:
            n_cocurr = self.df[msks.all(axis=1)].shape[0]

        return n_cocurr




