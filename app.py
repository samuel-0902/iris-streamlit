# Contenu de votre fichier app.py (copiez ceci dans un éditeur de texte et enregistrez-le sous "app.py")
import pandas as pd
import numpy as np
from sklearn import datasets
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, f_oneway, kruskal

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

import streamlit as st # Importation de Streamlit

# --- 1. Charger et préparer le dataset Iris ---
@st.cache_data # Mise en cache des données pour améliorer les performances de l'application
def load_iris_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({i: name for i, name in enumerate(iris.target_names)})
    return df, iris.target_names

data, target_names = load_iris_data()

# --- 2. Fonction de tracé Matplotlib (histo_streamlit) ---
# Adaptée pour Streamlit
def histo_streamlit(df, column, mode='percent', species_selected=None, mean_to_show='None'):
    if species_selected is None:
        species_selected = [target_names[0]] # Utilise le premier nom d'espèce par défaut

    fig, ax = plt.subplots(1, 1, figsize=(10, 5)) # Taille de figure ajustée pour Streamlit
    stat_type = 'density' if mode == 'percent' else 'count'
    palette = {'setosa': 'green', 'versicolor': 'blue', 'virginica': 'orange'}

    if len(species_selected) == 0:
        ax.set_title("Veuillez sélectionner au moins une espèce.")
        st.pyplot(fig) # Affiche la figure vide
        plt.close(fig)
        return

    group_values = []
    group_sizes = []

    for sp in species_selected:
        subset = df[df['species'] == sp][column].dropna()
        group_values.append(subset)
        group_sizes.append(len(subset))
        sns.histplot(subset, ax=ax, stat=stat_type, kde=False,
                     label=f"{sp.capitalize()} (n={len(subset)})", color=palette[sp],
                     element='step', fill=True, alpha=0.4)

    ax.legend(title='Espèces')
    normal_flags = [shapiro(vals)[1] > 0.05 for vals in group_values]

    # Affichage de la moyenne sélectionnée
    if mean_to_show == 'Global':
        values = df[df['species'].isin(species_selected)][column].dropna()
        mean = values.mean()
        std = values.std()
        ax.axvline(mean, color='black', linestyle='-', linewidth=2, label='Moyenne Globale')
        ax.axvline(mean - std, color='black', linestyle='--', linewidth=1)
        ax.axvline(mean + std, color='black', linestyle='--', linewidth=1)

        mean_text = f"Moyenne Globale = {mean:.2f}\nSD = {std:.2f}"
        ax.text(0.02, 0.98, mean_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    elif mean_to_show in species_selected:
        subset = df[df['species'] == mean_to_show][column].dropna()
        mean = subset.mean()
        std = subset.std()
        color = palette[mean_to_show]
        ax.axvline(mean, color=color, linestyle='-', linewidth=2)
        ax.axvline(mean - std, color=color, linestyle='--', linewidth=1)
        ax.axvline(mean + std, color=color, linestyle='--', linewidth=1)

        mean_text = f"{mean_to_show.capitalize()} Moyenne = {mean:.2f}\nSD = {std:.2f}"
        ax.text(0.02, 0.98, mean_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Statistiques
    stat_text = ""
    if len(species_selected) == 1:
        values = group_values[0]
        stat, p_value = shapiro(values)
        normality = 'Normale' if p_value > 0.05 else 'Non-normale'
        stat_text = (
            f"Test de Shapiro-Wilk\n"
            f"W = {stat:.3f}\n"
            f"p = {p_value:.3f}\n"
            f"→ {normality}"
        )
    elif len(species_selected) == 2:
        vals1, vals2 = group_values
        label1, label2 = species_selected
        n1, n2 = group_sizes
        if all(normal_flags):
            stat, pval = ttest_ind(vals1, vals2)
            test_name, stat_label = "T-test", "t"
        else:
            stat, pval = mannwhitneyu(vals1, vals2)
            test_name, stat_label = "Mann-Whitney", "U"
        p_display = "< 0.001" if pval < 0.001 else f"{pval:.3f}"
        stat_text = (
            f"{test_name} ({label1} vs {label2})\n"
            f"{stat_label} = {stat:.2f}\n"
            f"p = {p_display}\n"
            f"N = {n1} vs {n2}"
        )
    elif len(species_selected) == 3:
        if all(normal_flags):
            stat, pval = f_oneway(*group_values)
            test_name, stat_label = "ANOVA", "F"
        else:
            stat, pval = kruskal(*group_values)
            test_name, stat_label = "Kruskal-Wallis", "H"
        p_display = "< 0.001" if pval < 0.001 else f"{pval:.3f}"
        sizes_text = " / ".join([f"{n}" for n in group_sizes])
        stat_text = (
            f"{test_name}\n"
            f"{stat_label} = {stat:.2f}\n"
            f"p = {p_display}\n"
            f"N = {sizes_text}"
        )
    
    if stat_text:
        ax.text(0.98, 0.98, stat_text,
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    if mode == 'percent':
        ax.set_ylabel('Pourcentage')
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    else:
        ax.set_ylabel('Compte')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: int(y)))

    ax.set_title(f"Distribution de {column}")

    st.pyplot(fig) # Affiche la figure Matplotlib dans Streamlit
    plt.close(fig) # Important pour libérer la mémoire Matplotlib


# --- Interface Streamlit ---
st.title("Analyse Interactive du Dataset Iris")

# Sélection de la colonne
column = st.sidebar.selectbox(
    "Sélectionnez une colonne:",
    [col for col in data.columns if col != 'species']
)

# Sélection du mode d'affichage
mode = st.sidebar.radio(
    "Axe Y:",
    ('percent', 'absolute'),
    format_func=lambda x: 'Pourcentage' if x == 'percent' else 'Compte'
)

# Sélection des espèces
species_options = [{'label': str(sp).capitalize(), 'value': str(sp)} for sp in target_names]
selected_species = st.sidebar.multiselect(
    "Sélectionnez les espèces à afficher:",
    options=[opt['value'] for opt in species_options],
    default=[str(target_names[0])],
    format_func=lambda x: dict(species_options)[x] if x in dict(species_options) else x
)


# Option d'affichage de la moyenne/SD
mean_to_show_options = ['None', 'Global'] + list(selected_species)
mean_to_show = st.sidebar.selectbox(
    "Afficher Moyenne/SD:",
    mean_to_show_options,
    format_func=lambda x: x.capitalize() if x != 'None' and x != 'Global' else x
)

# Appel de la fonction de tracé avec les sélections de l'utilisateur
histo_streamlit(data, column, mode, selected_species, mean_to_show)