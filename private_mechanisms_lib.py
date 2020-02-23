import numpy as np
import math

class No_privacy:

    def __init__(self, _epsilon, data, _data_bounds):
        self.data = data

    def respond_query(self, query):
        true_answer = np.sum(self.data @ query)
        return true_answer

class Random_Answers:
    def __init__(self, _epsilon, _data, data_bounds):
        self.low, self.high = data_bounds

    def respond_query(self, query):

        random_elements_amount = np.random.randint(0, len(query))
        random_elements = np.random.randint(self.low, self.high,random_elements_amount)
        noise = np.sum(random_elements)
        return noise


class Round_to_R_multiplication:
    def __init__(self, R, data, _data_bounds):
        self.R = R
        self.data = data

    def respond_query(self, query):
        true_answer = np.sum(self.data @ query)

        # do epsilon manipulation
        # noised_answer = self.R * math.ceil(true_answer / self.R)
        noised_answer = self.R * np.rint(true_answer / self.R)
        return noised_answer

# class Positive_epsilon_gaussian_noise:
class Epsilon_gausian_noise:
    def __init__(self, epsilon, data, _data_bounds):
        self.epsilon = epsilon
        self.data = data

    def respond_query(self, query):
        true_answer = np.sum(self.data @ query)

        # do epsilon manipulation
        # noised_answer = true_answer + abs(np.random.normal(loc=0.0, scale=self.epsilon, size=None))
        noised_answer = true_answer + np.random.normal(loc=0.0, scale=self.epsilon, size=None)

        return noised_answer

