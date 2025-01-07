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


def viz_tsne(model, task="enroll", data_dir="../data", max_x=500, device="cpu"):
    x, y, _ = load_data(task, data_dir)
    latent_representation = model(x[:max_x].to(device))

    # Réduction de dimension avec t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_representation.detach().cpu().numpy())

    # Création de la figure
    plt.figure(figsize=(10, 8))

    # Nuage de points avec couleurs selon les labels
    scatter = plt.scatter(
        latent_2d[:, 0], 
        latent_2d[:, 1], 
        c=y[:max_x], 
        cmap='viridis', 
        s=50, 
        edgecolor='k', 
        alpha=0.7
    )

    # Définir un titre en fonction de la tâche
    if task == "train":
        title = "In-sample Representation (Training Data)"
    elif task == "enroll":
        title = "Out-of-sample Representation (Enrollment Data)"
    else:
        title = "Latent Space Representation"

    # Ajout du titre
    plt.title(title, fontsize=16)

    # Ajout des étiquettes des axes
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)

    # Ajout d'une barre de couleur (pour indiquer les classes ou labels)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Classes", fontsize=12)

    # Ajout d'une légende pour les labels de classe, si applicable
    plt.legend(*scatter.legend_elements(), title="Classes", loc="upper right", fontsize=10)

    # Affichage de la figure
    plt.tight_layout()
    plt.show()

