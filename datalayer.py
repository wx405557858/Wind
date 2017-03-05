__author__ = 'shawn'
import numpy as np

class Datalayer(object):
    def __init__(self,
                 fn='../WindData/jiangsu/M2865_RD_no_20140101-20150101.txt',
                 log_duration=30 * 24 * 6,
                 duration=24,
                 param_id=0):
        # read data
        f = open(fn, 'r')

        # read schema
        line = f.readline()
        keys = line.rstrip().split('\t')
        id_key = dict([(i, k) for i, k in enumerate(keys)])
        key_id = dict([(k, i) for i, k in enumerate(keys)])

        data = [line.rstrip().split('\t') for line in f.readlines()]
        N = len(data)
        print "N=%d" % N

        # transform format
        print "reading data..."
        wind = np.zeros([N, len(key_id)-1])
        for i in range(N):
            for j in range(1, len(key_id)):
                try:
                    wind[i][j-1] = eval(data[i][j])
                except:
                    wind[i][j-1] = 0.0

        for i in range(wind.shape[1]):
            wind[wind[:,i]==0,i] = (np.sum(wind[:,i])+1e-6) / (np.sum(wind[:,i]!=0)+1e-6)

        print wind.shape

        # extract features
        print "extracting features..."
        X = []
        Y = []
        for i in range(log_duration, N-6*duration):
            if data[i+6*duration][param_id+1] != 0:
                feature = [wind[i-log_duration:i,param_id]]
                X.append(np.reshape(feature,[-1]))
                Y.append([wind[i+6*duration,param_id]])

        X = np.array(X)
        Y = np.array(Y)
        self.X, self.Y, self.N = X, Y, len(X)
        print self.X.shape

    def k_fold(self, K):
        fold_size = self.N / K
        for i in range(K):
            test_id = range(i*fold_size, (i+1)*fold_size)
            train_id = range(0, i*fold_size) + range((i+1)*fold_size, K*fold_size)
            yield (self.X[train_id], self.Y[train_id], self.X[test_id], self.Y[test_id])

if __name__ == '__main__':
    data = Datalayer()
    for d in data.k_fold(5):
        train_X, train_Y, test_X, test_Y = d
        print train_X.shape,test_X.shape
