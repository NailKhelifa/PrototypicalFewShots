import matplotlib.pyplot as plt
import seaborn as sns
from .test_model import load_data
from sklearn.manifold import TSNE


def plot_from_list_acc(list_acc):
    # n_values = list(list_acc.keys())
    acc_values = [acc for acc_list in list_acc.values() for acc in acc_list]

    n_labels = [n for n in list_acc.keys() for _ in list_acc[n]]

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=n_labels, y=acc_values)

    plt.xlabel('Number of Shots (n)')
    plt.ylabel('Accuracy')
    plt.title('Distribution of Accuracy for Different n (k Runs)')
    plt.grid(True)

    plt.show()


def viz_tsne(model, task="enroll", data_dir="../data", max_x=500, device="cuda:0"):
    x, y, _ = load_data(task, data_dir)
    latent_representation = model(x[:max_x].to(device))

    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_representation.detach().cpu().numpy())
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y[:max_x], cmap='viridis', s=50,
                edgecolor='k', alpha=0.7)

    if task == "train":
        title = "In sample representation"
    elif task == "enroll":
        title = "Out of sample representation"
    else:
        title = ""

    plt.title(title)

    plt.show()
