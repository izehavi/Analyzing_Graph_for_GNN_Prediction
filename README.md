# Analyzing Graphs for GNN Prediction

## Description

Ce projet a été réalisé dans le cadre d'un stage visant à identifier les caractéristiques des graphes liées aux performances des Graph neural network (GNN) pour la prédiction de la consommation électrique. Le travail comprend une analyse théorique approfondie des graphes utilisés pour entraîner les GNN, ainsi qu'une exploration des relations entre les propriétés structurelles des graphes et leur impact sur les performances des modèles.

## Contenu du dépôt

- **`Rapport_stage_graphes.pdf`** : Le rapport complet du stage, détaillant les approches, les résultats et les conclusions.
- **`Final_code/main_plot_report.ipynb`** : Le notebook principal pour générer toutes les visualisations présentes dans le rapport.
- **`requirements.txt`** : La liste des dépendances Python nécessaires pour exécuter le projet.

---

## Installation

### 1. Cloner le repository
```bash
git clone https://github.com/izehavi/Analyzing_Graph_for_GNN_Prediction.git
cd Analyzing_Graph_for_GNN_Prediction
```

### 2. Créer un environnement virtuel et installer les dépendances
- **Sous Linux/macOS** :
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- **Sous Windows** :
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Lancer Jupyter Notebook (pour visualiser les plots du rapport)
```bash
jupyter notebook
```
Ouvrez ensuite le fichier `Final_code/main_plot_report.ipynb` pour générer les visualisations.

---

## Visualisations

Tous les plots présents dans le rapport peuvent être générés et analysés dans le notebook `Final_code/main_plot_report.ipynb`. Cela inclut des graphiques montrant les caractéristiques des graphes, leur évolution dans le temps, et les relations entre les propriétés structurelles des graphes et les performances du GNN.

---

## Perspectives et Utilisation

L'objectif final de ce projet est d'exploiter les analyses pour optimiser les graphes utilisés par les GNN, en identifiant les caractéristiques clés qui influencent directement leurs performances. Les travaux futurs incluront l'exploration de nouvelles méthodes pour ajuster les graphes et leur impact sur les prédictions.

---

## Auteur

Ce projet a été développé par [Itai Zehavi](https://github.com/izehavi) dans le cadre d'un stage.
