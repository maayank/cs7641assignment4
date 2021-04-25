import hiive.mdptoolbox.example as example
import hiive.mdptoolbox.mdp as mdp
import numpy as np
import pickle
from plot import *
from myfrozenlake import make_lake
from copy import deepcopy

np.random.seed(42)

def experiment_forest(make_solver, gamma, S, r1, r2, p, **kwargs):
    P, R = example.forest(S, r1, r2, p)
    solver = make_solver(P, R, gamma, **kwargs)
    res = solver.run()
    return res, solver.V, solver.policy

def experiment_lake(make_solver, gamma, S, **kwargs):
    P, R = make_lake(S, seed=42)

    solver = make_solver(P, R, gamma, **kwargs)
    res = solver.run()
    return res, solver.V, solver.policy


p2results = {}

def cached_experiment(pname, solver_name, p: dict, extra_tag=0, ret_key = False):
    key = [pname, extra_tag, solver_name]
    for k in sorted(p.keys()):
        v = p[k]
        if type(v) is list:
            v = tuple(v)
        key.append(v)
    key = tuple(key)
    if key not in p2results:
        print(f'Creating {key}')
        if pname == 'forest':
            p2results[key] = experiment_forest(solvers[solver_name], **p)
        elif pname == 'lake':
            p2results[key] = experiment_lake(solvers[solver_name], **p)
        else:
            raise Exception(pname)
    if ret_key:
       return p2results[key], ret_key
    else:
       return p2results[key]

FOREST_DEMO_PARAMS = {
    'gamma': [0.1, .5, 1-1e-1, 1-1e-2, 1-1e-3],
    'r1': [1, 2, 3, 4],
    'r2': [1, 2, 3, 4],
    'p': [0, .1, .5, .95 , 1]
}

FOREST_PARAMS2 = {
    'gamma': [0.1, .5, 1-1e-1],
    'S': [50, 500],
}

BASELINE_FOREST = {
    'gamma': 1-1e-1,
    'S': 1000,
    'r1': 4,
    'r2': 2,
    'p': .1,
    'alpha_decay': .999,
    'alpha': 1,
    'alpha_min': .01,
    'epsilon_min': .01,
    'epsilon_decay': 0.999,
    'n_iter': 100000
}

BASELINE_LAKE = {
    'gamma': 1-1e-1,
    'S': 64,
    'alpha_decay': .999,
    'alpha': 1,
    'alpha_min': .01,
    'epsilon_min': .01,
    'epsilon_decay': 0.999,
    'n_iter': 100000
}

q_stuff = {
                'alpha': [0.001, 0.01, 0.1, 0.5, .9, 1.0],
                'alpha_decay': [0.9, 0.99, 0.999, 1.0],
                'alpha_min': [1],
                'n_iter': [10000, 1000000, 5000000, 10000000],
                'epsilon_min': [0.001, 0.01, 0.1],
                'epsilon_decay': [0.9, 0.99, 0.999]
            }


BASELINE_DEMO_FOREST = {
    'gamma': 1-1e-1,
    'S': 100,
    'r1': 4,
    'r2': 2,
    'p': .1
}

problems = {
    'forest': {
        'baseline': BASELINE_FOREST,
        'interesting_states': range(100, 1000, 100),
        'interesting_gamma': [0.1, .5, 1-1e-1],
        'demo_params': {
            'gamma': [0.1, .5, 1-1e-1],
            'r1': [1,2,3,4],
            'r2': [1,2,3,4],
            'p': [0, .1, .5, .99]
        }
    },
    'lake': {
        'baseline': BASELINE_LAKE,
        'interesting_states': [i*i for i in range(8,14)],
        'interesting_gamma': [0.1, .5, 1-1e-1],
        'demo_params': {
            'gamma': [0.1, .5, 1-1e-1],
        }
    },
}

solvers = {
    'vi': lambda P, R, gamma, **kwargs: mdp.ValueIteration(transitions=P, reward=R, gamma=gamma, epsilon=.001, skip_bound=True, max_iter=10000),
    'pi': lambda P, R, gamma, **kwargs: mdp.PolicyIteration(transitions=P, reward=R, gamma=gamma, max_iter=10000),
    'q': lambda P, R, gamma, **kwargs: mdp.QLearning(P, R, gamma, **kwargs)
}

def generate_params(params_dict):
    from itertools import product
    keys = list(params_dict.keys())
    values = params_dict.values()
    for assignment in product(*values):
        yield {k: assignment[i] for i, k in enumerate(keys)}

def generate_params2(baseline, params_dict, yield_baseline=True):
    if yield_baseline:
        yield baseline
    for k in params_dict:
        for v in params_dict[k]:
            res = deepcopy(baseline)
            res[k] = v
            if res != baseline or not yield_baseline:
                yield res

def evaluate_demo():
    solver_name = 'vi'
    for pname in problems:
        fname = f'{pname}/demo'
        run_params = problems[pname]
        baseline = deepcopy(run_params['baseline'])
        if pname == 'forest': baseline['S'] = 20

        demo_params = run_params['demo_params']
        p_demo = list(generate_params2(baseline, demo_params))
        def foo(pi):
            policy = cached_experiment(pname, solver_name, pi)[2]
            print(f'Params:\t{pi}')
            print(f'Policy:\t{policy}')

        for pi in p_demo: foo(pi)


def evaluate2():
    '''
    Want here to evaluate the effects of the gamma and states
    '''
    #solver_name = 'vi'

    # Example of one result:
    # {'State': None, 'Action': None, 'Reward': 23.163862159935118, 'Error': 0.0009596435130951875, 'Time': 0.1786341667175293, 'Max V': 23.163862159935118, 'Mean V': 5.086812152213748, 'Iteration': 60}

    for pname in problems:
        run_params = problems[pname]
        baseline = run_params['baseline']
        interesting_states = run_params['interesting_states']
        interesting_gamma = run_params['interesting_gamma']
        
        for solver_name in solvers:
            fname = f'{pname}/{solver_name}'
            # If Q, how the max_v changes with alpha:
            if solver_name == 'q':
                p_q = list(generate_params2(baseline, q_stuff))
                def bar(pi):
                    res = []
                    for i in range(1):
                        last = cached_experiment(pname, solver_name, pi, i)[0][-1]
                        res.append(last['Mean V'])
                    return res
                def bar2(pi):
                    res = []
                    for i in range(1):
                        last = cached_experiment(pname, solver_name, pi, i)[0][-1]
                        res.append(last['Max V'])
                    return res
                for q_attr in q_stuff:
                    results = {pi[q_attr]: bar(pi) for pi in p_q if q_attr in pi}
                    results2 = {pi[q_attr]: bar2(pi) for pi in p_q if q_attr in pi}
                    plot_xy(f'Mean V as a function of {q_attr} for Q-Learning over {pname}', fname, q_attr, 'Mean V', results, special_values='decay' not in q_attr)
                    plot_xy(f'Max V as a function of {q_attr} for Q-Learning over {pname}', fname, q_attr, 'Max V', results, special_values='decay' not in q_attr)

            # Is mean time per iteration consistent over states?
            p_s = list(generate_params2(baseline, {
                'S': interesting_states
            }))
            def foo(pi):
                res = []
                for i in range(1):
                    last = cached_experiment(pname, solver_name, pi, i)[0][-1]
                    res.append(last['Iteration']/last['Time'])
                return res
            results = {pi['S']: foo(pi) for pi in p_s}
            plot_xy(f'Throughput as a function of states for {solver_name.upper()} over {pname}', fname, 'states', 'iterations/sec', results)

            # Is mean time per iteration consistent over gamma?
            p_g = list(generate_params2(baseline, {
                'gamma': interesting_gamma
            }))
            results = {pi['gamma']: foo(pi) for pi in p_g}
            plot_xy(f'Throughput as a function of gamma for {solver_name.upper()} over {pname}', fname, 'gamma', 'iterations/sec', results)

            def foo(pi):
                res = []
                for i in range(1):
                    last = cached_experiment(pname, solver_name, pi, i)[0][-1]
                    res.append(last['Iteration'])
                return res

            results = {pi['S']: foo(pi) for pi in p_s}
            plot_xy(f'Iterations as a function of states for {solver_name.upper()} over {pname}', fname, 'states', 'iterations', results)
            results = {pi['gamma']: foo(pi) for pi in p_g}
            plot_xy(f'Iterations as a function of gamma for {solver_name.upper()} over {pname}', fname, 'gamma', 'iterations', results)

            def foo(pi):
                res = []
                for i in range(1):
                    last = cached_experiment(pname, solver_name, pi, i)[0][-1]
                    res.append(last['Time'])
                return res

            results = {pi['S']: foo(pi) for pi in p_s}
            plot_xy(f'Time as a function of states for {solver_name.upper()} over {pname}', fname, 'states', 'sec', results)
            results = {pi['gamma']: foo(pi) for pi in p_g}
            plot_xy(f'Time as a function of gamma for {solver_name.upper()} over {pname}', fname, 'gamma', 'sec', results)

            def foo(pi):
                res = []
                for i in range(1):
                    last = cached_experiment(pname, solver_name, pi, i)[0][-1]
                    res.append(last['Max V'])
                return res

            results = {pi['gamma']: foo(pi) for pi in p_g}
            plot_xy(f'Max V as a function of gamma for {solver_name.upper()} over {pname}', fname, 'gamma', 'Max V', results)
 
            evaluate_baseline_attribute(pname, solver_name, 'Reward')
            evaluate_baseline_attribute(pname, solver_name, 'Error')
            evaluate_baseline_attribute(pname, solver_name, 'Max V')
            evaluate_baseline_attribute(pname, solver_name, 'Mean V')

def evaluate_baseline_attribute(pname, solver_name, attr):
    res = cached_experiment(pname, solver_name, problems[pname]['baseline'])[0]
    it2attr = {i['Iteration']: i[attr] for i in res}
    fname = f'{pname}/{solver_name}'
    plot_xy(f'{attr} as a function of iteration for {solver_name.upper()} over {pname}', fname, 'Iteration', attr, it2attr)

if __name__ == '__main__':
    evaluate_demo()
    evaluate2()