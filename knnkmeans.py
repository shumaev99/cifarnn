######################################################
# Простая имплементация алгоритма k-means clustering #
######################################################

# Работа алгоритма:
# 0. Накидать центров случайно
# 1. Определить для каждой точки ближайший центр и определить ее в соответствующий кластер
# 2. Пересчитать новые центры как средние всех точек в кластерах
# 3. Повторять 1 и 2, пока центры не перестанут меняться

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def kmeans(input_data, clusters = 5, max_iterations = 10000, dist_func = lambda x,y: np.linalg.norm(x-y), eps=1e-10, verbose=False):
    '''
    input_data: список точек
    clusters: количество кластеров
    dist_func: метрика расстоянияб берет 2 точки и возвращает расстояние
    eps: точность обнаружения условия остановки алгоритма
    verbose: булевая переменная, выводить ли сообщения
    возвращает: список меток, номер кластера для каждой точки, а также центры кластеров
    '''
    if len(input_data) == 0:
        return ([], [])
    # преобразуем входные данные в nparray
    data = np.array(input_data)
    # запоминаем размерность входных данных
    point_shape = data[0].shape
    # масштабируем данные
    scaler = MinMaxScaler().fit(data)
    scaled_data = scaler.transform(data)
    # нарандомим исходные положения центров
    centers = np.random.rand(clusters, *point_shape)
    closest_center_per_point = []
    # поехали итерации
    for i in range(max_iterations):
        if verbose:
            print("Entering iteration", i)
        # находим центр, ближайший к каждой точке
        closest_center_per_point = []
        for pt in range(len(data)):
            dist_list = [dist_func(centers[ct], data[pt]) for ct in range(clusters)]
            closest_center_per_point.append(np.argmin(dist_list))
        # пересчитываем положения центров
        new_centers = []
        for ct in range(clusters):
            filtered_points = [data[pt] for pt in range(len(data)) if closest_center_per_point[pt] == ct]
            # может такое случиться, что один из кластеров просто пропадет, в этом
            # случае надо запустить алгоритм сначала
            if len(filtered_points) == 0:
                if verbose:
                    print("Bad cluster center, restarting algorithm")
                return kmeans(data, clusters, max_iterations, dist_func, eps, verbose)
            new_centers.append(np.mean(filtered_points, axis=0))
        newcent = np.array(new_centers)
        # проверяем условие выхода из алгоритма
        done = True
        for ct in range(clusters):
            if dist_func(centers[ct], newcent[ct]) >= eps:
                done = False
                break
        if done:
            return (closest_center_per_point, centers)
        centers = newcent
    return (closest_center_per_point, centers)

# простая симуляция алгоритма kmeans
def create_kmeans_simulation():
    # generating a random example
    # small circles: data
    # triangles: cluster centers
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    # parameters
    npoints = 50 # number of points in simulation
    nclusters = 6 # number of clusters
    shape = (2,) # has to be (2,) for 2-dimensional data
    # generate random data
    alldata = np.random.rand(npoints, *shape)
    # apply kmeans algorithm
    clst, cent = kmeans(alldata, clusters=nclusters)
    # plotting code
    plt.title("K-means on random data")
    for cl in range(nclusters):
        thispts = [i for i in range(npoints) if clst[i] == cl]
        colors = ([col.hsv_to_rgb((cl/nclusters, 1, 0.9))], [col.hsv_to_rgb((cl/nclusters, 1, 0.8))])
        plt.scatter(*cent[cl], marker='^', s=100, c=colors[1])
        plt.scatter(*(alldata[thispts,].T), marker='o', s=20, c=colors[0])
    plt.show()

#############################################
# Простой k-nearest neighbors классификатор #
#############################################

# Алгоритм:
# 1. Найти k ближайших соседей к запрашиваемой точке
# 2. Выбрать класс, наиболее часто встречающийся среди этих соседей

class EvaluationError(Exception):
    pass

class SimpleKNNClassifier:
    def __init__(self):
        # создать пустой контейнер
        self.data = np.array([])
        self.labels = np.array([])
        self.fitted = False
        self.k = None
        self.dist_func = None

    def fit(self, data, labels, k, dist_func = lambda x,y: np.linalg.norm(x-y)):
        # запомнить тренировочные точки
        # если модель уже натренирована, вернуть ошибку
        if self.fitted:
            raise EvaluationError("Classifier already fitted.")
        if not len(data) == len(labels):
            raise EvaluationError("Size of data does not match size of labels")
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.k = k
        self.fitted = True
        self.dist_func = dist_func

    def refit(self, data, labels, k, dist_func = lambda x,y: np.linalg.norm(x-y)):
        # перезаписать тренировочные данные
        if not len(data) == len(labels):
            raise EvaluationError("Size of data does not match size of labels")
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.k = k
        self.fitted = True
        self.dist_func = dist_func

    def predict(self, newdata):
        # предсказание новой точки
        preds = []
        if not self.fitted:
            raise EvaluationError("Classifier not fitted.")
        if self.k > len(self.data):
            raise EvaluationError("Too few data points, or too high k.")
        for pt in range(len(newdata)):
            # расстояния до всех точек
            dists = [self.dist_func(newdata[pt], self.data[other]) for other in range(len(self.data))]
            # классы соседей
            votes = [self.labels[lbl] for lbl in range(len(self.labels)) if lbl in np.argpartition(dists, self.k)[:self.k]]
            # частотная таблица классов соседей
            unique, counts = np.unique(votes, return_counts=True)
            # нам нужен самый частый класс
            preds.append(unique[np.argmax(counts)])
        return np.array(preds)

# простая симуляция алгоритма knn
def create_knn_simulation():
    # generating a random example
    # small circles: training data
    # squares: testing data
    import matplotlib.pyplot as plt
    import matplotlib.colors as col
    # parameters
    k = 5 # number of neighbors to consider
    ntrain = 100 # number of training points in simulation
    ntest = 10 # number of testing points in simulation
    nclasses = 5 # number of classes
    shape = (2,) # has to be (2,) for 2-dimensional data
    likelihoods_from_dists = lambda dists: np.exp(-20*np.array(dists)**2)
    # generate random training and testing features
    Xtrain = np.random.rand(ntrain, *shape)
    Xtest = np.random.rand(ntest, *shape)
    # cluster training data to create a base for feature assignment
    cent = kmeans(Xtrain, clusters=nclasses)[1]
    # assign features randomly with probabilities found from distances to cluster centers, for smoother data
    ytrain = []
    for i in range(ntrain):
        probs = likelihoods_from_dists([np.linalg.norm(Xtrain[i]-cent[cn]) for cn in range(nclasses)])
        probs /= np.sum(probs)
        ytrain.append(np.random.choice(np.array(range(nclasses)), p=probs))
    # now apply knn classification
    knncl = SimpleKNNClassifier()
    knncl.fit(Xtrain, ytrain, k)
    ans = knncl.predict(Xtest)
    # plotting code
    plt.title("KNN on random data")
    for cl in range(nclasses):
        trainpts = [i for i in range(ntrain) if ytrain[i] == cl]
        testpts = [i for i in range(ntest) if ans[i] == cl]
        colors = ([col.hsv_to_rgb((cl/nclasses, 1, 0.9))], [col.hsv_to_rgb((cl/nclasses, 1, 0.7))])
        plt.scatter(*(Xtest[testpts,].T), marker='s', s=60, c=colors[1])
        plt.scatter(*(Xtrain[trainpts,].T), marker='o', s=15, c=colors[0])
    plt.show()


if __name__ == '__main__':
    create_kmeans_simulation()
    create_knn_simulation()
