import argparse
import numpy as np
import pandas as pd
import os
from tqdm import trange
from tqdm.contrib import tzip
from sklearn.preprocessing import StandardScaler

class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=True):
        """
        :R: 평점 행렬 
        :param k: latent 차원 수
        :param learning_rate: 학습률
        :param reg_param: weight의 regularization 값
        :param epochs: 학습 횟수
        :param verbose: 학습 과정 출력 여부
        """

        self._R = R
        self._num_users, self._num_items = R.shape  # 29755 * 1938
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose     


    def fit(self):

        # init latent features
        self._P = np.random.normal(size=(self._num_users, self._k))  # 29755 * k
        self._Q = np.random.normal(size=(self._num_items, self._k))  # 1938 * k

        # init biases
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)  
        self._b = np.mean(self._R[np.where(self._R != 0)]) # 전체 점수의 평균

        # train while epochs
        self._training_process = [] # epoch, cost 저장 리스트
        for epoch in range(self._epochs):
            # 점수가 존재하는 index를 기준으로 training
            xi, yi = self._R.nonzero() # 고객이 구매한 상품에 대한 값 추출
            for i, j in tzip(xi, yi):
                self.gradient_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))
            print("epoch: %d ; cost = %.4f" % (epoch + 1, cost))


    def cost(self):
        xi, yi = self._R.nonzero()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - self.get_prediction(x, y), 2)
        return np.sqrt(cost/len(xi)) # rmse function


    def gradient(self, error, i, j):

        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        return dp, dq


    def gradient_descent(self, i, j, rating):
        """
        i: 유저 인덱스
        j: 아이템 인덱스
        rating: 점수
        """
        # get error
        prediction = self.get_prediction(i, j)
        error = rating - prediction

        # update biases
        self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

        # update latent feature
        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._learning_rate * dp
        self._Q[j, :] += self._learning_rate * dq


    def get_prediction(self, i, j):
        """
        get predicted rating: user_i, item_j
        return: prediction of r_ij
        """
        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)


    def get_complete_matrix(self):
        """
        - PXQ 행렬에 b_P[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
        - b_Q[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
        - b를 더하는 것은 각 element마다 bias를 더해주는 것

        - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을 해주기위해 차원을 추가하는 것.

        :return: complete matrix R^
        """
        return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)


# run example
if __name__ == "__main__":
    
    import argparse
    import numpy as np
    import pandas as pd
    import os
    from tqdm import trange
    from tqdm.contrib import tzip
    from sklearn.preprocessing import StandardScaler

    #argparse
    #################### Arguments ####################

    parser = argparse.ArgumentParser(description="MatrixFactorization.")
    parser.add_argument('--path', nargs='?',default = './',
                            help='Input data path.')
    parser.add_argument('--dataset', nargs='?',default = 'ds.csv',
                            help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
                            help='Number of epochs.')
    parser.add_argument('--num_factors', type=int, default=20,
                            help='Number of latent_factors.')
    parser.add_argument('--reg_param',  type=float, default=0.01,
                            help="reg_param")
    parser.add_argument('--lr', type=float, default=0.01,
                            help='Learning rate.')
    args = parser.parse_args()
    

    k = args.num_factors
    reg_param = args.reg_param
    learning_rate = args.lr
    epochs = args.epochs

    #load dataset
    data_input = pd.read_csv(os.path.join(args.path+args.dataset),index_col='고객번호').drop(columns = 'Unnamed: 0')

    data_input_filter = data_input.groupby(['고객번호','상품코드'])['구매수량'].agg('count')
    data_input_filter = data_input_filter.unstack('상품코드')

    #StandardScaler
    scaler = StandardScaler()
    R = scaler.fit_transform(data_input_filter)
    R[np.isnan(R)] = 0
    R.astype('float16')

    #model
    factorizer = MatrixFactorization(R, k, learning_rate, reg_param, epochs, verbose=True)
    factorizer.fit()
    
    #show output
    data_final = factorizer.get_complete_matrix() * np.std(data_input_filter.fillna(0).values, axis=0) + np.mean(data_input_filter.fillna(0).values, axis=0)
   
    mf_data = pd.DataFrame(data_final, index=data_input_filter.index, columns=data_input_filter.columns)
    mf_data.to_csv('mf_data.csv')
