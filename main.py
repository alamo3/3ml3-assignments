import numpy as np
import matplotlib.pyplot as plt

# random search function
def random_search(g,alpha_choice,max_its,w,num_samples):
    # run random search
    w_history = [w] # container for w history
    cost_history = [g(w)] # container for corresponding cost function history
    alpha = alpha_choice

    num_dimensions = len(w)
    for k in range(1,max_its+1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice

        directions = np.zeros((num_samples,num_dimensions), dtype=np.ndarray)
        for i in range(num_samples):
            direction = np.random.randn(num_dimensions)
            norm = np.sqrt(direction.dot(direction))

            direction = direction / norm

            directions[i] = direction


        w_candidates = w + (alpha * directions)


        min_eval = g(w_candidates[0])
        min_index = 0

        for i in range(1, len(w_candidates)):
            eval_curr = g(w_candidates[i])

            if eval_curr < min_eval:
                min_eval = eval_curr
                min_index = i

        w = w_candidates[min_index]
        cost_history.append(g(w))
        w_history.append(w)

    print(w_history[-1])
    return cost_history





def experiment_1():
    g = lambda w: np.dot(w.T, w) + 2
    alpha_choice = 0.3
    w = np.array([3, 4])
    num_samples = 1000
    max_its = 50

    cost_history = random_search(g, alpha_choice, max_its, w, num_samples)
    print(cost_history[-1])
    plt.plot(cost_history)
    plt.show()

def experiment_2():
    g = lambda w: 100 * (w[1] - w[0]**2)**2 + (w[0] - 1) ** 2

    alpha = 'diminishing'
    w = np.array([-2, -2])
    num_samples = 1000
    max_its = 50

    cost_history = random_search(g, alpha, max_its, w, num_samples)
    print(cost_history[-1])
    plt.plot(cost_history)
    plt.show()

def experiment_3():
    g = lambda w: (4 - 2.1*(w[0]**2) + (w[0]**4)/3)*(w[0] ** 2) + w[0]*w[1] + (-4 + 4*(w[1]**2)) * (w[1] ** 2)

    alpha = 0.5
    w = np.array([0, 0])
    num_samples = 1000
    max_its = 50

    cost_history = random_search(g, alpha, max_its, w, num_samples)
    print(cost_history[-1])
    plt.plot(cost_history)
    plt.show()


def exercise_2(N, P):
    g = lambda w: np.dot(w.T, w) + 2

    w0 = np.zeros(N)
    w0[0] = 1

    directions = np.zeros((P, N), dtype=np.ndarray)
    for i in range(P):
        direction = np.random.randn(N)
        norm = np.sqrt(direction.dot(direction))

        direction = direction / norm

        directions[i] = direction

    w_candidates = w0 + (1.0 * directions)
    num_descent = 0
    for i in range(P):
        if g(w_candidates[i]) < g(w0):
            num_descent += 1

    return num_descent / P

def exercise_2_run():
    fractions_x = []
    fractions_y = []
    for i in range(1,30):
        fractions_x.extend([i] * 4)
        fractions_y.append(exercise_2(i, 10))
        fractions_y.append(exercise_2(i, 100))
        fractions_y.append(exercise_2(i, 1000))
        fractions_y.append(exercise_2(i, 10_000))

    plt.scatter(fractions_x, fractions_y)
    plt.show()


def coordinate_search(g,alpha_choice,max_its,w):
    w_history = [w]  # container for w history
    cost_history = [g(w)]  # container for corresponding cost function history
    num_dimensions = len(w)

    directions = np.identity(num_dimensions)
    for k in range(1, max_its + 1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1 / float(k)
        else:
            alpha = alpha_choice

        w_candidates = w + (alpha * directions)

        min_eval = g(w_candidates[0])
        min_index = 0

        for i in range(1, len(w_candidates)):
            eval_curr = g(w_candidates[i])

            if eval_curr < min_eval:
                min_eval = eval_curr
                min_index = i

        w = w_candidates[min_index]
        cost_history.append(g(w))

    print(w_history[-1])
    return cost_history


def coordinate_descent(g,alpha_choice,max_its,w):
    w_history = [w]  # container for w history
    cost_history = [g(w)]  # container for corresponding cost function history
    num_dimensions = len(w)

    directions = np.identity(num_dimensions)
    for k in range(1, max_its + 1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1 / float(k)
        else:
            alpha = alpha_choice

        w_candidates = w + (alpha * directions)

        min_index = 0

        for i in range(1, len(w_candidates)):
            eval_curr = g(w_candidates[i])

            if eval_curr < g(w):
                min_index = i
                break

        w = w_candidates[min_index]
        cost_history.append(g(w))

    print(w_history[-1])
    return cost_history


def exercise_3_1():
    g = lambda w : 0.26*(w[0] ** 2 + w[1] ** 2) - 0.48*w[0]*w[1]

    alpha = 'diminishing'
    w = np.array([3, 4])
    max_its = 40

    cost_history = coordinate_search(g, alpha, max_its, w)
    print(cost_history[-1])
    plt.plot(cost_history)
    plt.show()

def exercise_3_2():
    g = lambda w : 0.26*(w[0] ** 2 + w[1] ** 2) - 0.48*w[0]*w[1]

    alpha = 'diminishing'
    w = np.array([3, 4])
    max_its = 40

    cost_history = coordinate_descent(g, alpha, max_its, w)
    print(cost_history[-1])
    plt.plot(cost_history)
    plt.show()


if __name__ == '__main__':
    exercise_3_1()






