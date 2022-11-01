import numpy as np
import pandas as pd
import mydl


def load_csv(path):
    df = pd.read_csv(path, header=None)
    return df.values


class Paramters: pass
    

class WParamter(Paramters):
    
    def __init__(self, row, col):
        self.paramter = mydl.randn(
            row, col, requires_grad=True)
        
        
class BParamter(Paramters):
    
    def __init__(self, col):
        self.paramter = mydl.randn(
            1, col, requires_grad=True)


class Layer:
    
    def __init__(self, input_num, output_num):
        pass
    
    def __call__(self, X):
        return self.forward(X)
        
    def forward(self, X):
        return 0.
    
    @property
    def paramters(self):
        pass
        

class LinearLayer(Layer):
    
    def __init__(self, input_num, output_num):
        self.weights = WParamter(input_num, output_num)
        self.bias = BParamter(output_num)
    
    def forward(self, X):
        out = self.weights.paramter.matmul(X, order='neg')
        out = out+mydl.ones((X.shape[0], 1)).matmul(self.bias.paramter)
        return out
    
    @property
    def paramters(self):
        return self.weights, self.bias
    
    
class SoftmaxLoss:
    
    def __init__(self, X, *, dim):
        if dim == 0:
            X = X.T
        self.X = X
        
    def __call__(self, Y):
        return self.cal_loss(Y)
    
    def cal_loss(self, Y):
        return mydl.softmax_loss(self.X, Y)
    
    
class Optimize: pass


class SGD(Optimize):
    
    def __init__(self, *paramters, alpha, lambda_=0):
        self.weights_set = set()
        self.bias_set = set()
        for paramter in paramters:
            if not isinstance(paramter, WParamter):
                self.bias_set.add(paramter)
            else:
                self.weights_set.add(paramter)
        self.back_up_w = None
        self.back_up_b = None
        self.alpha = alpha
        self.lambad_ = lambda_
        
    def update(self):
        for w in self.weights_set:
            w.paramter -= self.lambad_*w.paramter+ \
                          self.alpha*w.paramter.grad
        for b in self.bias_set:
            b.paramter -= self.alpha*w.paramter.grad
            
    def backup(self):
        self.back_up_w = self.weights_set
        self.back_up_b = self.bias_set
    
    def back(self):
        if self.back_up_w is not None:
            self.weights_set = self.back_up_w
        if self.back_up_b is not None:
            self.bias_set = self.back_up_b
            
            
class NNN:
    
    def __init__(self, input_num):
        self.layer1 = LinearLayer(input_num, 256)
        # self.layer2 = LinearLayer(9, 6)
        # self.layer3 = LinearLayer(9, 9)
        # self.layer4 = LinearLayer(9, 9)
        # self.layer5 = LinearLayer(9, 9)
        # self.layer6 = LinearLayer(9, 9)
        self.layer7 = LinearLayer(256, 10)
        self.opt = SGD(*self.layer1.paramters,
                       # *self.layer2.paramters,
                       # *self.layer3.paramters,
                       # *self.layer4.paramters,
                       # *self.layer5.paramters,
                       # *self.layer6.paramters,
                       *self.layer7.paramters,
                       alpha=0.01, lambda_=0)
        self.test_datas = None
        self.test_lables = None
        
    def __call__(self, X):
        out = self.forward(X)
        return out.tensor.argmax(axis=1)
        
    def forward(self, X):
        out = self.layer1(X).relu()
        # out = self.layer2(out).relu()
        # out = self.layer3(out).relu()
        # out = self.layer4(out).relu()
        # out = self.layer5(out).relu()
        # out = self.layer6(out).relu()
        out = self.layer7(out).relu()
        return out
        
    def add_test(self, test_datas, test_lables):
        self.test_datas = test_datas
        self.test_lables = test_lables
        
    def train(self, datas, lables, max_iter=99):
        M, _ = datas.shape
        iter_num = 0
        min_loss = np.inf
        min_err = 1
        while iter_num <= max_iter:
            permulation = np.random.permutation(M)
            for small_batch in np.array_split(permulation, 10):
                small_batch_datas = datas[small_batch]
                small_batch_lables = lables[small_batch]
                out = self.forward(small_batch_datas)
                loss = SoftmaxLoss(out, dim=1)(small_batch_lables)/10
                predict = self.__call__(self.test_datas)
                err = np.count_nonzero(predict-self.test_lables.ravel())/10
                if loss.tensor < min_loss:
                    min_loss = loss.tensor
                    print('min loss:', min_loss)
                else: self.opt.alpha /= 2
                if err < min_err:
                    min_err = err
                    self.opt.backup()
                    print('min err:', err)
                loss.backward()
                self.opt.update()
            iter_num += 1
        self.opt.back()


if __name__ == '__main__':
    train_set_path = './source/mnist_train_100.csv'
    train_set = load_csv(train_set_path)
    train_lables, train_datas = \
                  np.split(train_set, [1], axis=1)
    train_datas = train_datas/255
    test_set_path = './source/mnist_test_10.csv'
    test_set = load_csv(test_set_path)
    test_lables, test_datas = \
                 np.split(test_set, [1], axis=1)
    test_datas = test_datas/255
    
    M, N = train_datas.shape
    nn = NNN(input_num=N)
    nn.add_test(test_datas, test_lables)
    nn.train(train_datas, train_lables)
    
    result = nn(test_datas)
    err = result-test_lables.ravel()
    error = np.count_nonzero(err)/10
    print(f'错误率为{error:.3f}')
    
    
