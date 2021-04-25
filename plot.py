import matplotlib.pyplot as plt
import numpy as np

def fname_transform(name):
    return name.replace('/', '_').replace(' ', '_').lower()

def save_fig(name):
    import os
    path = f'pics/{name}.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close('all')

def sort_xy(d):
    x = sorted(list(d.keys()))
    y = [d[k] for k in x]
    return np.asarray(x),np.asarray(y)

def plot_xy(title, name, x_name, y_name, x2y, special_values = False):
    try:
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_title(title)
        x, y = sort_xy(x2y)
        if len(y.shape) > 1:
            mean = y.mean(axis=1)
        else:
            mean = y
        ax.set_xlabel(x_name.title())
        ax.set_ylabel(y_name.title())
        if special_values:
            ax.set_xticks(x)
            ax.set_xscale('log')
        ax.plot(x, mean, color='green')
        if len(y.shape) > 1:
            std = y.std(axis=1)
            ax.fill_between(x, mean - std, mean + std, alpha=0.4, color='green')
        save_fig(f'{name}/{fname_transform(x_name)}_{fname_transform(y_name)}')
    except:
        print(f'Error while handling {x2y}')
        raise