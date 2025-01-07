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


def viz_tsne(models, task="enroll", data_dir="../data", max_x=500, device="cpu"):
    """
    Visualise les embeddings t-SNE pour une liste de modèles.

    :param models: Liste de modèles PyTorch.
    :param task: Tâche associée ("train", "enroll", etc.).
    :param data_dir: Chemin vers le dossier contenant les données.
    :param max_x: Nombre maximum d'exemples à considérer.
    :param device: Appareil sur lequel exécuter les modèles (par exemple, "cpu" ou "cuda:0").
    """
    x, y, _ = load_data(task, data_dir)
    num_models = len(models)

    # Vérification que le nombre de modèles est pair
    assert num_models % 2 == 0, "Le nombre de modèles doit être pair."

    # Création d'une figure avec sous-plots
    fig, axes = plt.subplots(
        nrows=num_models // 2, ncols=2, figsize=(12, 6 * (num_models // 2))
    )
    axes = axes.flatten()  # Aplatir pour un accès plus simple aux axes

    for i, model in enumerate(models):
        # Calcul des embeddings pour chaque modèle
        latent_representation = model(x[:max_x].to(device))

        # Réduction de dimension avec t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_representation.detach().cpu().numpy())

        # Création du nuage de points
        scatter = axes[i].scatter(
            latent_2d[:, 0],
            latent_2d[:, 1],
            c=y[:max_x],
            cmap="viridis",
            s=50,
            edgecolor="k",
            alpha=0.7,
        )

        # Définir un titre spécifique pour chaque sous-plot
        if task == "train":
            title = f"Model {i+1}: In-sample (Training)"
        elif task == "enroll":
            title = f"Model {i+1}: Out-of-sample (Enrollment)"
        else:
            title = f"Model {i+1}: Latent Space"

        axes[i].set_title(title, fontsize=14)
        axes[i].set_xlabel("t-SNE Dimension 1", fontsize=12)
        axes[i].set_ylabel("t-SNE Dimension 2", fontsize=12)

        # Ajouter une barre de couleur spécifique à chaque sous-plot
        cbar = fig.colorbar(scatter, ax=axes[i])
        cbar.set_label("Classes", fontsize=12)

    # Ajuster l'espacement entre les sous-plots
    plt.tight_layout()
    plt.show()
