import numpy as np

class toy_DC:
    def __init__(self):
        self.data = np.array([1,5,3,8,0,8,6,4,2,3])
        self.size = len(self.data)

    def score(self, data_hat):
        delta = np.abs(np.array(data_hat)- self.data)

        missed = np.zeros(len(delta))
        missed[np.where(delta>1)] = 1

        return 1 - sum(missed)/len(missed)

    def get_baseline(self):
        return 0.3

    def get_data_bounds(self):
        return (0,11)

class intro_grades_DB_DC:

    def __init__(self):
        filename = "secret_data/intro_grades.csv"

        intro_DB = []

        f = open(filename, "r", encoding="utf-8")
        for row in f:
            try:
                intro_DB += [int(row.split(',')[1].replace('\n', ''))]
            except:
                pass
        f.close()

        self.data = np.array(intro_DB)
        self.size = len(self.data)

    def score(self, data_hat):
        delta = np.abs(np.array(data_hat)- self.data)

        missed = np.zeros(len(delta))
        missed[np.where(delta>2)] = 1

        return 1 - sum(missed)/len(missed)

    def get_data_name(self):
        return 'Intro grades sample'

    def get_data_bounds(self):
        return (0,101)

class census_citizenship_DB_DC:

        def __init__(self):
            filename = "secret_data/2010_census.csv"

            census_DB = []

            f = open(filename, "r")
            for row in f:
                try:
                    census_DB += [int(row.split(',')[11])]
                except:
                    pass
            f.close()

            self.data = np.array(census_DB)
            self.size = len(self.data)

        def score(self, data_hat):

            delta = np.abs(np.array(data_hat) - self.data)

            return 1 - sum(delta) / len(delta)

        def get_data_name(self):
            return 'Census citizenship sample'

        def get_data_bounds(self):
            return (0, 2)