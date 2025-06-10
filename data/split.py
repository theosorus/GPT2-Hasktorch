import os
import random
from pathlib import Path

def split_large_file(input_file, train_ratio=0.9, shuffle=True, seed=42):
    """
    Divise un fichier de grande taille en fichiers train.txt et valid.txt
    
    Args:
        input_file (str): Chemin vers le fichier d'entrée
        train_ratio (float): Proportion pour l'entraînement (0.9 = 90%)
        shuffle (bool): Mélanger les lignes avant la division
        seed (int): Graine pour la reproductibilité
    """
    
    # Définir les noms des fichiers de sortie
    input_path = Path(input_file)
    output_dir = input_path.parent
    train_file = output_dir / "train.txt"
    valid_file = output_dir / "valid.txt"
    
    print(f"Lecture du fichier: {input_file}")
    
    # Lire toutes les lignes du fichier
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # Essayer avec un autre encodage si UTF-8 échoue
        with open(input_file, 'r', encoding='latin-1') as f:
            lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Nombre total de lignes: {total_lines}")
    
    # Mélanger les lignes si demandé
    if shuffle:
        random.seed(seed)
        random.shuffle(lines)
        print("Lignes mélangées")
    
    # Calculer la division
    train_size = int(total_lines * train_ratio)
    valid_size = total_lines - train_size
    
    print(f"Lignes d'entraînement: {train_size} ({train_ratio*100:.1f}%)")
    print(f"Lignes de validation: {valid_size} ({(1-train_ratio)*100:.1f}%)")
    
    # Diviser les données
    train_lines = lines[:train_size]
    valid_lines = lines[train_size:]
    
    # Écrire le fichier d'entraînement
    print(f"Écriture de {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # Écrire le fichier de validation
    print(f"Écriture de {valid_file}")
    with open(valid_file, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)
    
    # Afficher les statistiques
    train_size_mb = os.path.getsize(train_file) / (1024 * 1024)
    valid_size_mb = os.path.getsize(valid_file) / (1024 * 1024)
    
    print("\n--- Résumé ---")
    print(f"Fichier d'entraînement: {train_file} ({train_size_mb:.2f} MB)")
    print(f"Fichier de validation: {valid_file} ({valid_size_mb:.2f} MB)")
    print("Division terminée avec succès!")

def split_large_file_memory_efficient(input_file, train_ratio=0.9, shuffle=False, seed=42):
    """
    Version optimisée pour les très gros fichiers (traitement ligne par ligne)
    Note: shuffle=False pour cette version car le mélange nécessiterait de charger tout en mémoire
    """
    
    input_path = Path(input_file)
    output_dir = input_path.parent
    train_file = output_dir / "train.txt"
    valid_file = output_dir / "valid.txt"
    
    print(f"Traitement du fichier: {input_file} (mode économe en mémoire)")
    
    # Première passe: compter les lignes
    print("Comptage des lignes...")
    total_lines = 0
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
    except UnicodeDecodeError:
        with open(input_file, 'r', encoding='latin-1') as f:
            for line in f:
                total_lines += 1
    
    print(f"Nombre total de lignes: {total_lines}")
    
    # Calculer la division
    train_size = int(total_lines * train_ratio)
    
    print(f"Lignes d'entraînement: {train_size} ({train_ratio*100:.1f}%)")
    print(f"Lignes de validation: {total_lines - train_size} ({(1-train_ratio)*100:.1f}%)")
    
    # Deuxième passe: écrire les fichiers
    print("Division des données...")
    try:
        with open(input_file, 'r', encoding='utf-8') as input_f, \
             open(train_file, 'w', encoding='utf-8') as train_f, \
             open(valid_file, 'w', encoding='utf-8') as valid_f:
            
            for i, line in enumerate(input_f):
                if i < train_size:
                    train_f.write(line)
                else:
                    valid_f.write(line)
                
                # Afficher le progrès
                if (i + 1) % 100000 == 0:
                    print(f"Traité: {i + 1}/{total_lines} lignes")
    
    except UnicodeDecodeError:
        with open(input_file, 'r', encoding='latin-1') as input_f, \
             open(train_file, 'w', encoding='utf-8') as train_f, \
             open(valid_file, 'w', encoding='utf-8') as valid_f:
            
            for i, line in enumerate(input_f):
                if i < train_size:
                    train_f.write(line)
                else:
                    valid_f.write(line)
                
                if (i + 1) % 100000 == 0:
                    print(f"Traité: {i + 1}/{total_lines} lignes")
    
    print(f"\nDivision terminée!")
    print(f"Fichier d'entraînement: {train_file}")
    print(f"Fichier de validation: {valid_file}")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "data.txt"  # Remplacez par le chemin de votre fichier
    TRAIN_RATIO = 0.9  # 90% pour l'entraînement, 10% pour la validation
    
    # Vérifier si le fichier existe
    if not os.path.exists(INPUT_FILE):
        print(f"Erreur: Le fichier {INPUT_FILE} n'existe pas.")
        print("Veuillez modifier la variable INPUT_FILE avec le bon chemin.")
        exit(1)
    
    # Obtenir la taille du fichier
    file_size_mb = os.path.getsize(INPUT_FILE) / (1024 * 1024)
    print(f"Taille du fichier: {file_size_mb:.2f} MB")
    
    # Choisir la méthode selon la taille du fichier
    if file_size_mb > 500:  # Plus de 500 MB
        print("Fichier volumineux détecté, utilisation du mode économe en mémoire")
        split_large_file_memory_efficient(INPUT_FILE, TRAIN_RATIO)
    else:
        print("Utilisation du mode standard avec mélange des données")
        split_large_file(INPUT_FILE, TRAIN_RATIO, shuffle=True)