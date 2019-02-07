import pickle

data_dir = '../data/'
    
def read_pickle(f):
    return pickle.load(open(f, 'rb'))

def write_pickle(obj, f):
    pickle.dump(obj, open(f, 'wb'))
    
def load_network():
    return read_pickle(data_dir + 'network.p')