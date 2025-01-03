import matplotlib.pyplot as plt
import seaborn as sns


def plot_from_list_acc(list_acc):
    n_values = list(list_acc.keys())
    acc_values = [acc for acc_list in list_acc.values() for acc in acc_list]

    n_labels = [n for n in list_acc.keys() for _ in list_acc[n]]

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=n_labels, y=acc_values)

    plt.xlabel('Number of Shots (n)')
    plt.ylabel('Accuracy')
    plt.title('Distribution of Accuracy for Different n (k Runs)')
    plt.grid(True)

    plt.show()
