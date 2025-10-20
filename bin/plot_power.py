import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_power_curves(data_dir='measure/data/600sec/'):
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Trier par nombre d'utilisateurs

    x_min, x_max, y_min, y_max = float('inf'), -float('inf'), float('inf'), -float('inf')
    for file in files:
        df = pd.read_csv(os.path.join(data_dir, file))
        df.iloc[:,1]=df.iloc[:,1]/1000
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        x_min, x_max = min(x_min, x.min()), max(x_max, x.max())
        y_min, y_max = min(y_min, y.min()), max(y_max, y.max())

    n_plots = len(files)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots), sharex=True, sharey=True)

    if n_plots == 1:
        axes = [axes]  # Pour uniformiser le traitement

    for i, (file, ax) in enumerate(zip(files, axes)):
        # Extraire le nombre d'utilisateurs depuis le nom de fichier
        n_users = int(file.split('_')[-1].split('.')[0])

        # Lire le fichier CSV
        df = pd.read_csv(os.path.join(data_dir, file))
        df.iloc[:,1]=df.iloc[:,1]/1000
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        # Tracer la courbe
        ax.plot(x, y, label=f'{n_users} users')
        ax.set_ylabel('Puissance (W)')
        ax.set_title(f'Consommation pour {n_users} utilisateurs')
        ax.grid(True)
        ax.legend()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Mettre un label x seulement sur le dernier graphique
    axes[-1].set_xlabel('Temps (s)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_power_curves()

