import matplotlib.pyplot as plt


def plot_graph():
    mean = [2, 10, -7]
    sigma = [9, 3, 15]
    number = [10, 20, 2]
    y_val = []
    bias_val = []
    variance_val = []
    total_err_val = []

    for u in mean:
        for s in sigma:
            for n in number:
                y = 0
                while y < 30:
                    bias = ((y ** 2) * (u ** 2)) / ((1 + y) ** 2)
                    variance = s / (n * ((1 + y)**2))
                    y_val.append(y)
                    bias_val.append(bias)
                    variance_val.append(variance)
                    total_err_val.append(bias + variance)
                    y = y + 1
                plt.clf()
                plt.plot(y_val, bias_val, label="Bias",
                         markersize=4, color='red', linewidth=2)
                plt.plot(y_val, variance_val, label="Variance",
                         markersize=4, color='blue', linewidth=2)
                plt.plot(y_val, total_err_val, label="Total Error",
                         markersize=4, color='yellow', linewidth=2)
                plt.title("Bias vs. Variance as a Function of λ with μ = {0}, σ^2 = {1}, n = {2}".format(u, s, n))
                plt.xlabel("λ")
                plt.ylabel("Units")
                plt.xticks(range(0, 31, 2))
                plt.legend()
                plt.savefig("graph {0} {1} {2}.pdf".format(u, s, n))
                y_val = []
                bias_val = []
                variance_val = []
                total_err_val = []


if __name__ == "__main__":
    plot_graph()
