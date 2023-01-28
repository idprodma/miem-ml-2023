import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

log_epochs = False
use_2_opt_swap = True

def metric(A, B):
    return np.sum(np.sqrt((A - B)**2))

def calculate_energy(sequence, cities):
    n = np.size(sequence, 0)
    E = 0
    for i in range(n - 1):
        E += metric(cities[sequence[i], :], cities[sequence[i+1], :])
    E += metric(cities[sequence[-1], :], cities[sequence[0], :])

    return E

def decrease_temperature(init_temp, k):
    return init_temp * 0.1 / k

def get_transition_probability(dE, T):
    return np.exp(-dE / T)

def is_transition(prob):
    return 1 if np.random.rand(1) <= prob else 0

def generate_state_candidate(seq):
    n = np.size(seq, 0)
    i = np.random.randint(n)
    j = np.random.randint(n)

    # topic of this HW
    if use_2_opt_swap:
        seq[min(i, j):max(i,j)] = reversed(seq[min(i, j):max(i,j)])
    else: #simple swap
      t = seq[i]
      seq[i] = seq[j]
      seq[j] = t

    return seq

def optimise_route(cities, init_temp, end_temp):
    n_cities = np.size(cities, 0)
    state = list(range(n_cities))

    cur_energy = calculate_energy(state, cities)
    print(f'initial route length: {cur_energy}')

    T = init_temp

    for k in range(1, 1000001):
        state_candidate = generate_state_candidate(state)
        candidate_energy = calculate_energy(state_candidate, cities)
        if candidate_energy < cur_energy:
            state = state_candidate
            cur_energy = candidate_energy
        else:
            p = get_transition_probability(candidate_energy-cur_energy, T)
            if is_transition(p):
                state = state_candidate
                cur_energy = candidate_energy

        T = decrease_temperature(init_temp, k)

        if T <= end_temp:
            break

        if log_epochs and k % 1000 == 0:
            print('epoch: ', k)

    print(f'final route length: {cur_energy}')
    return state

def simulated_annealing():
    n_cities = 100
    init_temperature = 100
    end_temperature = 0

    cities = np.random.rand(n_cities, 2) * 10

    plt.plot(cities[:, 0], cities[:, 1], "b--o")
    plt.title('initial route')
    plt.show()

    state = optimise_route(cities, init_temperature, end_temperature)
    plt.plot(cities[state, 0], cities[state, 1], "r--o")
    plt.title('final route')
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-l", "--epochs", type=int, choices=range(2), default=0,
                        help="log epochs")
    parser.add_argument("-opt", "--twooptswap", type=int, choices=range(2) default=1,
                        help="use 2-opt swap instead of simple swap")

    args = parser.parse_args()
    log_epochs = args.epochs == 1
    use_2_opt_swap = args.twooptswap == 1
    try:
        simulated_annealing()
    except KeyboardInterrupt:
        pass
