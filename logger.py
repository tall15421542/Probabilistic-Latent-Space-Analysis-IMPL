class Logger:
    def __init__(self):
        self.train_likelihood_vec = []
        self.val_likelihood_vec = []
        self.test_likelihood_vec = []
        self.map_train_vec = []
        self.map_test_vec = []

    def log(train_likelihood, val_likelihood, test_likelihood, map_train, map_test):
        self.train_likelihood_vec.append(train_likelihood)
        self.val_likelihood_vec.append(val_likelihood)
        self.test_likelihood_vec.append(test_likelihood)
        self.map_train_vec.append(map_train)
        self.map_test_vec.append(map_test)

