import torch
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import functools

# Eigene Module
import config
import data_prep
import plots
import aggregation
from Base_Line import QuantumModel
from train_and_eval import train_local_model, evaluate_model


def run_single_seed(seed, label, actual_type, n_p, rsna_root, chexpert_root):
    """
    Führt einen kompletten Durchlauf für einen Seed aus.
    Lädt Daten pro Seed neu für maximale Varianz.
    """
    print(f"\n>>> START SEED {seed} | Konfiguration: {label}")

    # Daten neu laden pro Seed (wissenschaftliche Varianz)
    loaders, meta = data_prep.get_federated_pca_loaders(
        rsna_root,
        chexpert_root,
        current_seed=seed
    )
    g_val_loader = aggregation.build_global_val_loader(loaders)
    cent_loader = data_prep.get_centralized_loader(loaders, config.BATCH_SIZE)

    torch.manual_seed(seed)
    np.random.seed(seed)

    seed_results = []
    # Speicher für jede Epoche des Trainingsverlaufs
    qfl_loc_accum = {
        s: {'train_acc': [], 'train_loss': []} for s in config.ALL_SUBCLIENTS
    }
    seed_histories = {
        'Local Baseline': {s: [] for s in config.ALL_SUBCLIENTS},
        'lds_cms': {}
    }

    # --- A. ZENTRALE BASELINE ---
    c_model = QuantumModel(config.NUM_LAYERS, actual_type, n_p).to(config.DEVICE)
    train_local_model(
        c_model,
        cent_loader,
        g_val_loader,
        torch.optim.Adam(c_model.parameters(), lr=config.LR),
        config.EPOCHS_BASELINE,
        config.DEVICE,
        client_id=f"S{seed}_Cent"
    )

    for sid in config.ALL_SUBCLIENTS:
        # Evaluierung mit Dictionary-Rückgabe
        m = evaluate_model(
            c_model,
            loaders[sid]['test'],
            config.DEVICE,
            meta[sid]['minority_idx']
        )
        seed_results.append({
            'Seed': seed,
            'Szenario': 'Zentrale Baseline',
            'Noise_Type': actual_type or "None",
            'Noise_P': n_p,
            'Client': sid.split('_sub')[0],
            'Subclient': sid.split('_sub_')[1],
            'Testgenauigkeit': m['Testgenauigkeit'],
            'Testverlust': m['Testverlust'],
            'ROC_AUC': m['ROC_AUC'],
            'PR_AUC': m['PR_AUC'],
            'Recall_Minderheit': m['Recall_Minderheit'],
            'F1_Minderheit': m['F1_Minderheit'],
            'Balancierte_Genauigkeit': m['Balancierte_Genauigkeit']
        })

    # --- B. LOKALE BASELINE ---
    for sid in config.ALL_SUBCLIENTS:
        l_model = QuantumModel(config.NUM_LAYERS, actual_type, n_p).to(config.DEVICE)
        _, hist = train_local_model(
            l_model,
            loaders[sid]['train'],
            loaders[sid]['val'],
            torch.optim.Adam(l_model.parameters(), lr=config.LR),
            config.EPOCHS_BASELINE,
            config.DEVICE,
            client_id=f"S{seed}_{sid}"
        )
        # Speichern der lokalen Historie für Konvergenzplots
        seed_histories['Local Baseline'][sid] = hist

        m = evaluate_model(
            l_model,
            loaders[sid]['test'],
            config.DEVICE,
            meta[sid]['minority_idx']
        )
        seed_histories['lds_cms'][(sid, 'Lokale Baseline')] = m['cm']
        seed_results.append({
            'Seed': seed,
            'Szenario': 'Lokale Baseline',
            'Noise_Type': actual_type or "None",
            'Noise_P': n_p,
            'Client': sid.split('_sub')[0],
            'Subclient': sid.split('_sub_')[1],
            'Testgenauigkeit': m['Testgenauigkeit'],
            'Testverlust': m['Testverlust'],
            'ROC_AUC': m['ROC_AUC'],
            'PR_AUC': m['PR_AUC'],
            'Recall_Minderheit': m['Recall_Minderheit'],
            'F1_Minderheit': m['F1_Minderheit'],
            'Balancierte_Genauigkeit': m['Balancierte_Genauigkeit']
        })

    # --- C. QUANTUM FEDERATED LEARNING ---
    g_model = QuantumModel(config.NUM_LAYERS, actual_type, n_p).to(config.DEVICE)
    qfl_global_conv = {'acc': [], 'loss': []}

    for r in range(config.QFL_GLOBAL_ROUNDS):
        weights = []
        for sid in config.ALL_SUBCLIENTS:
            loc_m = QuantumModel(config.NUM_LAYERS, actual_type, n_p).to(config.DEVICE)
            loc_m.load_state_dict(g_model.state_dict())

            # Lokales Training
            w, h = train_local_model(
                loc_m,
                loaders[sid]['train'],
                loaders[sid]['val'],
                torch.optim.Adam(loc_m.parameters(), lr=config.LR),
                config.QFL_LOCAL_EPOCHS,
                config.DEVICE,
                client_id=f"S{seed}_{sid}_R{r}"
            )
            weights.append(w)
            # Sammeln jeder einzelnen lokalen Epoche
            qfl_loc_accum[sid]['train_acc'].extend(h['train_acc'])
            qfl_loc_accum[sid]['train_loss'].extend(h['train_loss'])

        # Globale Aggregation
        g_model.load_state_dict(aggregation.federated_averaging(weights))

        # Validierung der Runde
        v = evaluate_model(g_model, g_val_loader, config.DEVICE)
        qfl_global_conv['acc'].append(v['Testgenauigkeit'])
        qfl_global_conv['loss'].append(v['Testverlust'])

    for sid in config.ALL_SUBCLIENTS:
        m = evaluate_model(
            g_model,
            loaders[sid]['test'],
            config.DEVICE,
            meta[sid]['minority_idx']
        )
        seed_histories['lds_cms'][(sid, 'QFL')] = m['cm']
        seed_results.append({
            'Seed': seed,
            'Szenario': 'QFL',
            'Noise_Type': actual_type or "None",
            'Noise_P': n_p,
            'Client': sid.split('_sub')[0],
            'Subclient': sid.split('_sub_')[1],
            'Testgenauigkeit': m['Testgenauigkeit'],
            'Testverlust': m['Testverlust'],
            'ROC_AUC': m['ROC_AUC'],
            'PR_AUC': m['PR_AUC'],
            'Recall_Minderheit': m['Recall_Minderheit'],
            'F1_Minderheit': m['F1_Minderheit'],
            'Balancierte_Genauigkeit': m['Balancierte_Genauigkeit']
        })

    return seed_results, seed_histories, qfl_loc_accum, qfl_global_conv


def main():
    plots.setup_german_plot_style()
    master_results = []
    # Speicher für die Konvergenz-Evolution über p
    evolution_master = {
        nt: {0.0: None, 0.01: None, 0.05: None, 0.1: None} for nt in config.NOISE_TYPES
    }

    for n_type, n_p in config.PHASE1_SCENARIOS:
        label = f"{n_type}_p{n_p}" if n_type != "None" else "Noiseless"
        s_dir = os.path.join(config.BASE_RESULTS_DIR, label)
        os.makedirs(s_dir, exist_ok=True)
        actual_type = None if n_type == "None" else n_type

        print(f"\n" + "=" * 80 + f"\n SZENARIO: {label.upper()}\n" + "=" * 80)

        with ProcessPoolExecutor(max_workers=2) as executor:
            worker = functools.partial(
                run_single_seed,
                label=label,
                actual_type=actual_type,
                n_p=n_p,
                rsna_root=config.RSNA_ROOT,
                chexpert_root=config.CHEXPERT_ROOT
            )
            results = list(executor.map(worker, config.SEEDS))

        # Datenaggregation für dieses Rausch-Szenario
        scen_data = []
        qfl_high_res_agg = []
        qfl_round_agg = []
        total_lds_cms = {}
        hist_lb = {'Local Baseline': {s: [] for s in config.ALL_SUBCLIENTS}}

        for s_res, s_hist, qfl_loc, qfl_glob in results:
            scen_data.extend(s_res)
            master_results.extend(s_res)
            qfl_round_agg.append(qfl_glob)
            for sid in config.ALL_SUBCLIENTS:
                qfl_high_res_agg.append(qfl_loc[sid])
                hist_lb['Local Baseline'][sid].append(s_hist['Local Baseline'][sid])
            for k, cm in s_hist['lds_cms'].items():
                total_lds_cms[k] = total_lds_cms.get(k, 0) + cm

        # Mittelwert für Evolution-Plots
        mean_hist = {
            'acc': np.mean([h['acc'] for h in qfl_round_agg], axis=0),
            'loss': np.mean([h['loss'] for h in qfl_round_agg], axis=0)
        }
        if n_type == "None":
            for nt in config.NOISE_TYPES:
                evolution_master[nt][0.0] = mean_hist
        else:
            evolution_master[n_type][n_p] = mean_hist

        df_scen = pd.DataFrame(scen_data)
        flat_lb = [h for sl in hist_lb['Local Baseline'].values() for h in sl]

        # --- SCENARIO-SPECIFIC PLOTS ---
        plots.plot_training_comparison(
            flat_lb,
            qfl_high_res_agg,
            label,
            os.path.join(s_dir, "Training_Verlauf.png")
        )
        plots.plot_global_comparison_summary(df_scen, s_dir)
        plots.plot_paper_style_boxplot(df_scen, s_dir)
        plots.plot_recall_improvement_lds(df_scen, s_dir)

        # Konvergenz-Verzögerung für dieses p
        plots.plot_convergence_speed_comparison(
            flat_lb,
            qfl_high_res_agg,
            s_dir
        )

        for sid in ['client_1_sub_1', 'client_4_sub_1']:
            if (sid, 'Lokale Baseline') in total_lds_cms and (sid, 'QFL') in total_lds_cms:
                plots.plot_confusion_matrix_comparison(
                    total_lds_cms[(sid, 'Lokale Baseline')],
                    total_lds_cms[(sid, 'QFL')],
                    sid,
                    s_dir
                )

    # --- MASTER ANALYSEN ---
    master_df = pd.DataFrame(master_results)
    master_df.to_csv(
        os.path.join(config.BASE_RESULTS_DIR, "Master_Ergebnisse.csv"),
        index=False
    )

    # Übergreifende Analysen über alle p-Werte hinweg
    plots.plot_master_noise_sensitivity(master_df, config.BASE_RESULTS_DIR)
    plots.plot_noise_rescue_erosion(master_df, config.BASE_RESULTS_DIR)
    plots.plot_skew_noise_heatmap(master_df, config.BASE_RESULTS_DIR)
    plots.plot_stability_analysis(master_df, config.BASE_RESULTS_DIR)

    # Neue Master-Plots aus Phase 1
    plots.plot_noise_type_impact_ranking(master_df, config.BASE_RESULTS_DIR)
    plots.plot_critical_thresholds_annotated(master_df, config.BASE_RESULTS_DIR)

    for nt in config.NOISE_TYPES:
        if evolution_master[nt][0.0] is not None:
            plots.plot_noise_evolution_convergence(
                evolution_master[nt],
                nt,
                config.BASE_RESULTS_DIR
            )

    print(f"\n✔ Alle Experimente abgeschlossen. Ergebnisse: {config.BASE_RESULTS_DIR}")


if __name__ == "__main__":
    main()
