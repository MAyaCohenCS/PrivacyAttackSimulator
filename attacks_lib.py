from random import randrange
import numpy as np
from scipy.optimize import linprog

class LP_reconstructor:

    def __init__(self, data_size, data_bounds):
        self.data_size = data_size
        self.low, self.high = data_bounds
        self.A = []
        self.b = []

        self.inner_state = None

    def generate_query(self):
        sampled_subset = np.random.randint(2, size=self.data_size)
        self.A += [sampled_subset]
        return sampled_subset

    def learn_from_response(self, response):
        self.b += [response]

    def predict_origin(self):

        # gather knowledge to an LP canonic form
        m = len(self.A)
        A = np.array(self.A)
        e = np.identity(m)
        square_zero_matrix = np.zeros((m,m))

        A_top = np.hstack((A,e,square_zero_matrix,square_zero_matrix))
        A_bot = np.hstack((np.zeros(A.shape),e, e, e*(-1)))

        A = np.vstack((A_top,A_bot))

        b = np.hstack( (np.array(self.b), np.zeros(m) ))

        n = len(self.A[0])
        c = np.hstack( (  np.zeros(n+m) , np.ones(2*m)   ) )

        x_bounds = (self.low, self.high-1)
        no_bounds = (None, None)
        positive_bound = (0, None)

        bounds = [x_bounds] * n + [no_bounds] * m + [positive_bound] * (2*m)


        # run fractional LP reconstruction
        res = linprog(c, A_eq=A, b_eq=b, bounds=bounds)

        try:
            # round values
            lst = [int(round(h)) for h in res.x]
        except:
            print("couldn't solve LP, sending a random guess")
            lst = np.random.randint(low = self.low, high = self.high, size = self.data_size)


        return lst[:self.data_size]