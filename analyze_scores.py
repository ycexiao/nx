import os
import re
import pickle
import numpy as np
from matplotlib import  pyplot as plt
plt.style.use('seaborn-v0_8')

def get_meta_data(file_name):
    try:
        element = re.search(r'^.*?(?=-)', file_name).group(0)
        features = re.search(r'(?<=-F-).*(?=-T-)', file_name).group(0).split('-')
        target = re.search(r'(?<=-T-).*(?=.pickle)', file_name).group(0)
    except AttributeError:
        return None
    return element, features, target
    
def load_data(data_dir, get_meta_data):
    file_names = os.listdir(data_dir)
    file_path = [os.path.join(data_dir, file) for file in file_names]
    Data = []
    for i in range(len(file_names)):
        out = get_meta_data(file_names[i])
        if out is None:
            continue
        else:
            element, features, target = out
        
        data = {'element':element, 'features':features, 'target':target}
        with open(file_path[i], 'rb') as f:
            tmp = pickle.load(f)
            tmp = np.array(tmp)
            data['train_score'] = tmp[0]
            data['test_score'] = tmp[1]
        Data.append(data)
    return Data


def filter_data(data, *conditions):
    masks = np.zeros(len(data), dtype=bool)
    for i in range(len(conditions)):
        if isinstance(conditions[i], list):
            mask = np.array(list(map(lambda x: np.all([conditions[i][j](x) for j in range(len(conditions[i]))]), data)))
        else:
            mask = np.array(list(map(lambda x: conditions[i](x), data)))
        # print(mask)
        masks += mask
    
    inds = np.arange(len(data))[masks]
    return [data[i] for i in inds]

def generate_condition(key, value, mode='str'):
    if mode == 'str':
        if isinstance(value, list):
            return lambda x: set(x[key]) == set(value)
        else:
            return lambda x: x[key] == value
    else:
        return lambda x: len(x[key])==value

def combine_data(data, key, method=None):
    if not method:
        out = [data[i][key] for i in range(len(data))]
    else:
        out = [method(data[i][key]) for i in range(len(data))]

    return out


class Painter():
    def __init__(self,ax):
        self.metrics = []
        self.plot_params = []
        self.ax = ax

    def set_ax(self, ax):
        self.ax = ax
        
    def plot(self, xtick_labels, metrics, plot_param):
        
        self.metrics.append(metrics)
        self.plot_params.append(plot_param)
        n_class = len(self.metrics)
        width = 1/(n_class+0.5)
        x0 = np.arange(len(self.metrics[0]))

        self.ax.cla()
        self.ax.set_xticks(x0, xtick_labels, rotation=45)
    
        for i in range(n_class):
            x = x0 + width*i
            self.ax.bar(x, self.metrics[i], width=width, **self.plot_params[i])
                
            for j in range(len(self.metrics[i])):
                self.ax.text(x[j], self.metrics[i][j]*1.01, "{:.2f}".format(self.metrics[i][j]), horizontalalignment='center')
            

def sort_by_match(labels, values, matched_labels):
    return [values[i] for i in range(len(labels)) for j in range(len(matched_labels)) if set(labels[i]) == set(matched_labels[j]) ]


def lookup_scores(data, selections, ax=None):
    keys = ['features', 'target', 'element']
    conditions = [generate_condition('features', selections[0][i]) for i in range(len(selections[0]))]
    data = filter_data(Data, *conditions)
    match_labels = selections[0]
        
    labels = combine_data(data, 'features')
    data = sort_by_match(labels, data, match_labels)
    

    if ax is None:
        fig, ax = plt.subplots()
        print('!!!')
    painter = Painter(ax)


    for i in range(len(selections[1])):
        target_condition = generate_condition('target', selections[1][i])
        for j in range(len(selections[2])):
            element_condition = generate_condition('element', selections[2][j])
            tmp_data = filter_data(data, [target_condition, element_condition],)

            mean = combine_data(tmp_data, 'test_score', np.mean)
            std = combine_data(tmp_data, 'test_score', np.std)
            painter.plot(match_labels, mean, plot_param={'yerr':std, 'label': selections[1][i]})


if __name__ == '__main__':
    Data = load_data('results', get_meta_data)
    elements = ['Ti', 'Cu', 'Fe', 'Mn']
    fig, axes = plt.subplots(1,4, sharey=True)
    for i in range(len(elements)):        
        lookup_scores(Data, [[['x_pdf'], ['n_pdf'], ['nx_pdf'], ['nx_pdf', 'x_pdf'], ['nx_pdf', 'n_pdf']], ['cn', 'cs'], [elements[i]]],ax=axes[i])
        plt.legend(bbox_to_anchor=(0.9,1))
        axes[i].set_title(elements[i])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.ylim([0.6, 1])
    plt.show()
    
        # break

    # fig , ax = plt.subplots()
    # lookup_scores(Data, [[['x_pdf'], ['n_pdf'], ['nx_pdf'], ['nx_pdf', 'x_pdf'], ['nx_pdf', 'n_pdf']], ['cn', 'cs'], ['Ti']], ax=ax)
    # plt.legend(bbox_to_anchor=(0.9,1))
    # plt.show()

