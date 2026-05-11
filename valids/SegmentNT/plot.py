
import matplotlib.pyplot as plt
import seaborn as sns

def plot_features(predicted_probabilities_all, seq_length, features, order_to_plot, plot_path, fig_width=8):

    ## seaborn settings
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1, rc={ "font.size": 14, "axes.titlesize": 18, "axes.labelsize": 18, 
                    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16})
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    colors = sns.color_palette("Set2").as_hex()
    colors2 = sns.color_palette("husl").as_hex()
    
    ## plotting figures
    
    sc = 1.8
    n_panels = 7

    _, axes = plt.subplots(n_panels, 1, figsize=(fig_width * sc, (n_panels + 4) * sc))

    for n, feat in enumerate(order_to_plot):
        feat_id = features.index(feat)
        prob_dist = predicted_probabilities_all[:, feat_id]

        # Use the appropriate subplot
        ax = axes[n // 2]

        try:
            id_color = colors[feat_id]
        except:
            id_color = colors2[feat_id - 8]
        ax.plot(
            prob_dist,
            color=id_color,
            label=feat,
            linestyle="-",
            linewidth=1.5,
        )
        ax.set_xlim(0, seq_length)
        ax.grid(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')

    for a in range (0,n_panels):
        axes[a].set_ylim(0, 1.05)
        axes[a].set_ylabel("Prob.")
        axes[a].legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0)
        if a != (n_panels-1):
            axes[a].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

    # Set common x-axis label
    axes[-1].set_xlabel("Nucleotides")
    # axes[0].axis('off')  # Turn off the axis
    axes[n_panels-1].grid(False)
    axes[n_panels-1].tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False)

    axes[0].set_title("Probabilities predicted over all genomics features", fontweight="bold")

    plt.tight_layout()
    plt.savefig(plot_path)