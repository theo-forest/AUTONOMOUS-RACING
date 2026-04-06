import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import os

# Import de votre fonction d'entraînement
from improved_ddpg import train

# --- 1. DÉFINITION DE L'ESPACE DE RECHERCHE "FORMULE 1" ---
space = [
    # LRA : On cherche très bas pour la précision (Smoothness)
    Real(1e-5, 3e-4, prior='log-uniform', name='lra'),
    
    # LRC : On garde une plage standard, souvent 10x le LRA
    Real(1e-4, 2e-3, prior='log-uniform', name='lrc'),
    
    # TAU : Mise à jour lente des réseaux cibles pour la stabilité
    Real(0.001, 0.01, name='tau'),
    
    # SIGMA : Bruit FAIBLE. À 1.8 m/s, on veut de la précision chirurgicale.
    Real(0.05, 0.15, name='sigma')
]

@use_named_args(space)
def objective(**params):
    print(f"\nEvaluating with params: {params}")
    
    # --- 2. LANCEMENT DE L'ENTRAÎNEMENT ROBUSTE ---
    # On augmente episodes à 100 pour laisser le temps à la vitesse de se stabiliser
    # On impose min_buffer=5000 pour avoir une "culture générale" avant d'apprendre
    rewards, avg_rewards = train(
        episodes=100, 
        lra=params['lra'], 
        lrc=params['lrc'], 
        tau=params['tau'], 
        sigma=params['sigma'],
        min_buffer=5000  # <--- LE SECRET pour la haute performance
    )
    
    # On maximise la récompense moyenne des 20 derniers épisodes (plus représentatif)
    last_avg = np.mean(rewards[-20:])
    
    # Sécurité : Si l'entraînement a crashé ou donné des NaN
    if np.isnan(last_avg):
        last_avg = -1000.0
        
    print(f"Outcome average (last 20 eps): {last_avg:.2f}")
    
    # gp_minimize minimise, donc on retourne l'opposé
    return -last_avg

if __name__ == "__main__":
    # Eval avec LCB (Lower Confidence Bound) - Prudent
    print("Starting Bayesian Optimization with LCB...")
    res_lcb = gp_minimize(objective, space, n_calls=20, acq_func='LCB', random_state=42)
    
    print("\nBest parameters LCB:")
    print(res_lcb.x)
    print(f"Score LCB: {-res_lcb.fun:.2f}")
    
    # Sauvegarde des résultats
    with open("../weights/best_params_f1.txt", "w") as f:
        f.write(f"LRA: {res_lcb.x[0]}\n")
        f.write(f"LRC: {res_lcb.x[1]}\n")
        f.write(f"TAU: {res_lcb.x[2]}\n")
        f.write(f"SIGMA: {res_lcb.x[3]}\n")
        f.write(f"Best score: {-res_lcb.fun:.2f}\n")