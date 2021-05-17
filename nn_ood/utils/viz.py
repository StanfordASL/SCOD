import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
import pandas as pd

# utilities for visualizing results
def plot_histogram(ax, results):
    """
    plots a kde of uncertainty scores given to data by each technique
    in results on axis ax
    """
    for name, data in results.items():
        sns.kdeplot(data["uncs"], shade=True, label=name, ax=ax)
    ax.set_ylabel("freq")
    ax.set_xlabel("uncertainty")
    ax.legend()

def plot_scatter(axes, results):
    """
    plots a scatterplot of uncertainty vs error for each technique
    in results. each technique is plotted on a separate axis
    axes is a list of matplotlib axes of the same size as results
    """
    for i, (name, data) in enumerate(results.items()):
        axes[i].scatter(data["metrics"], data["uncs"], label=name, alpha=0.002, color='C'+str(i))
        axes[i].legend()
        axes[i].set_xlabel(data["metric_name"])
        axes[i].set_xlim([1e-8,1e2])
        axes[i].set_xscale("log")
        axes[i].grid()
            
    axes[0].set_ylabel("uncertainty estimate")


def compute_roc_and_pr(negative_label_values, positive_label_values):
    """
    Given two arrays:
        negative_label_values: signal values on inputs known to have a negative label
        positive_label_values: signal values on inputs known to have a positive label
    Outputs:
        tprs: true positive rate, at various thresholds
        fprs: false positive rate, at various thresholds
        precisions: precision of classifier at various thresholds
        recalls: recall of classifier at various thresholds
        auroc: area under roc curve
        aupr: area under precision recall curve
    """
    true_labels = np.concatenate([np.zeros_like(negative_label_values), np.ones_like(positive_label_values)])
    
    uncs = np.concatenate([negative_label_values, positive_label_values])
    idx_to_keep = np.where(np.logical_not(np.isnan(uncs)))
    true_labels = true_labels[idx_to_keep]
    uncs = uncs[idx_to_keep]
    
    precisions, recalls, _ = precision_recall_curve(true_labels, uncs)
    false_pos_rates, true_pos_rates, _ = roc_curve(true_labels, uncs)

    auroc = np.trapz(true_pos_rates, false_pos_rates)
    aupr = np.trapz(precisions[::-1], recalls[::-1])    
    
    return {
        'tprs': true_pos_rates,
        'fprs': false_pos_rates,
        'precisions': precisions,
        'recalls': recalls,
        'auroc': auroc,
        'aupr': aupr
    }

def bootstrapped_auroc_and_aupr(negative_label_values, positive_label_values,
                                num_bootstrap_samples):
    aggregates = {
        'auroc': [],
        'aupr': []
    }
            
    N = min(len(negative_label_values), len(positive_label_values))
    for k in range(num_bootstrap_samples):
        in_dist_idx = np.random.choice(len(negative_label_values), N, replace=True)
        out_dist_idx = np.random.choice(len(positive_label_values), N, replace=True)

        stats = compute_roc_and_pr(negative_label_values[in_dist_idx],
                                   positive_label_values[out_dist_idx])
        for metric in ['auroc', 'aupr']:
            aggregates[metric].append(stats[metric])

    stats = {}
    for metric in aggregates.keys():
        mean_metric = np.mean(aggregates[metric])
        std_err_metric = np.std(aggregates[metric]) / np.sqrt(num_bootstrap_samples)

        stats[metric] = mean_metric
        stats[metric+"_conf"] = std_err_metric*1.96
        
    return stats

def plot_rocs_by_error(results_dict, exp_list, error_thres, keys_to_compare=None, colors=None, **kwargs):
    if keys_to_compare is None:
        keys_to_compare = results_dict.keys()
    
    fig, axes = plt.subplots(2,1, **kwargs)
    roc_axis = axes[0]
    pr_axis = axes[1]

    roc_axis.set_title('ROC')
    roc_axis.set_ylabel('TPR')
    roc_axis.set_xlabel('FPR')
    
    pr_axis.set_title('PR Curve')
    pr_axis.set_ylabel('Precision')
    pr_axis.set_xlabel('Recall')
    
    summary = {}
    i = 0
    for i,name in enumerate(keys_to_compare):
        if name not in results_dict:
            continue
        results = results_dict[name]
        
        #combine uncs, errs
        errs = np.concatenate([results[exp]['metrics'] for exp in exp_list])
        uncs = np.concatenate([results[exp]['uncs'] for exp in exp_list])
        in_dist = uncs[errs <= error_thres]
        out_dist = uncs[errs > error_thres]
        
        stats = compute_roc_and_pr(in_dist, out_dist)
        summary[name] = stats
        
        if colors is not None:
            color = colors[i]
        else:
            color = 'C'+str(i)
        
        roc_axis.plot(stats['fprs'], stats['tprs'], label=name+(" AUC: %0.2f" % stats['auroc']), linewidth=2, color=color)
        pr_axis.plot(stats['recalls'], stats['precisions'], label=name+(" AUC: %0.2f" % stats['aupr']), linewidth=2, color=color)
        
        i+=1
        
    roc_axis.legend()
    pr_axis.legend()
    
    plt.tight_layout()
    
    return summary


def plot_times(results_dict, keys_to_compare=None, colors=None):
    times = []
    stds = []
    labels = []

    for i,name in enumerate(keys_to_compare):
        if name not in results_dict:
            continue
        results = results_dict[name]['train']

        color = 'C'+str(i)
        if colors is not None:
            color = colors[i]

        times.append(results['time_per_item'])
        stds.append(results['time_per_item_std'])
        labels.append(name)
        colors.append(color)
        
    plt.barh(np.arange(len(labels)), times, color=colors,
                       xerr=stds, tick_label=labels)
    plt.gca().set_xscale('log')
    
def plot_rocs_by_dist(results_dict, in_dist_splits, out_dist_splits, keys_to_compare=None, colors=None, **kwargs):
    if keys_to_compare is None:
        keys_to_compare = results_dict.keys()
    
    fig, axes = plt.subplots(2,1, **kwargs)
    roc_axis = axes[0]
    pr_axis = axes[1]

    roc_axis.set_title('ROC')
    roc_axis.set_ylabel('TPR')
    roc_axis.set_xlabel('FPR')
    
    pr_axis.set_title('PR Curve')
    pr_axis.set_ylabel('Precision')
    pr_axis.set_xlabel('Recall')
    
    summary = {}
    for i,name in enumerate(keys_to_compare):
        print(name)
        if name not in results_dict:
            continue
        results = results_dict[name]
        
        #combine uncs, errs
        in_dist = np.concatenate([results[exp]['uncs'] for exp in in_dist_splits])
        out_dist = np.concatenate([results[exp]['uncs'] for exp in out_dist_splits])
        
        stats = compute_roc_and_pr(in_dist, out_dist)
        summary[name] = stats
        
        if colors is not None:
            color = colors[i]
        else:
            color = 'C'+str(i)
        
        roc_axis.plot(stats['fprs'], stats['tprs'], label=name+(" AUC: %0.2f" % stats['auroc']), linewidth=2, color=color)
        pr_axis.plot(stats['recalls'], stats['precisions'], label=name+(" AUC: %0.2f" % stats['aupr']), linewidth=2, color=color)
        
        
    roc_axis.legend()
    pr_axis.legend()
    
    plt.tight_layout()
    
    return summary

def summarize_ood_results(results_dict, 
                          in_dist_splits, out_dist_splits,
                          keys_to_compare=None,
                          metrics=['auroc','aupr'],
                          num_bootstrap_samples=25):
    """
    inputs:
        results_dict: saved results from test_unc_models(), dict of dicts where
            results_dict[unc_model][split] contains results from applying
            unc_model to data in the dataset split.
            
        in_dist_splits: test dataset splits to use as in distribution data
        out_dist_splits: test dataset splits to use as out-of-distribution data
        keys_to_compare: if not None, list of keys in results_dict to compute
            results for (e.g., if only )
        
    returns summary, a dictionary where summary[key] is the following results
        {
            'auroc': AUROC,
            'auroc_conf': 95% confidence bounds on AUROC
            'aupr': AUPR,
            'aupr_conf': 95% confidence bound on AUPR,
            'runtime': avg runtime per sample in ms
            'runtime_conf': 95% confidence bound on runtime
        }
    """
    if keys_to_compare is None:
        keys_to_compare = results_dict.keys()
    
    summary = {}
    for name in keys_to_compare:
        if name not in results_dict:
            continue
        results = results_dict[name]
        
        #combine uncs, errs
        in_dist = np.concatenate([results[exp]['uncs'] for exp in in_dist_splits])
        out_dist = np.concatenate([results[exp]['uncs'] for exp in out_dist_splits])
        
        stats = bootstrapped_auroc_and_aupr(in_dist, out_dist, num_bootstrap_samples)
        
        time_per_item = np.mean([results[exp]['time_per_item'] 
                                 for exp in in_dist_splits+out_dist_splits])
        time_per_item_std = np.sqrt( 
            np.mean(
                [
                    (results[exp]['time_per_item_std']*np.sqrt(len(results[exp]['uncs'])))**2 
                    for exp in in_dist_splits+out_dist_splits
                ]
            ) 
        )
        time_per_item_std /= np.sqrt(np.sum([len(results[exp]['uncs']) 
                                             for exp in in_dist_splits+out_dist_splits]))
        
        stats['runtime'] = time_per_item*1000
        stats['runtime_conf'] = time_per_item_std*1000*1.96
        
        summary[name] = stats
        
    return summary

def summarize_ood_results_by_error(results_dict, 
                                   exp_list, error_thres,
                                   keys_to_compare=None,
                                   metrics=['auroc','aupr'],
                                   num_bootstrap_samples=25):
    """
    inputs:
        results_dict: saved results from test_unc_models(), dict of dicts where
            results_dict[unc_model][split] contains results from applying
            unc_model to data in the dataset split.
            
        exp_list: test dataset splits to use
        error_thres: theshold to separate in and out-of-dist data
        keys_to_compare: if not None, list of keys in results_dict to compute
            results for (e.g., if only )
        
    returns summary, a dictionary where summary[key] is the following results
        {
            'auroc': AUROC,
            'auroc_conf': 95% confidence bounds on AUROC
            'aupr': AUPR,
            'aupr_conf': 95% confidence bound on AUPR,
            'runtime': avg runtime per sample in ms
            'runtime_conf': 95% confidence bound on runtime
        }
    """
    if keys_to_compare is None:
        keys_to_compare = results_dict.keys()
    
    summary = {}
    for name in keys_to_compare:
        if name not in results_dict:
            continue
        results = results_dict[name]
        
        #combine uncs, errs
        errs = np.concatenate([results[exp]['metrics'] for exp in exp_list])
        uncs = np.concatenate([results[exp]['uncs'] for exp in exp_list])

        in_dist = uncs[errs <= error_thres]
        out_dist = uncs[errs > error_thres]
        
        stats = bootstrapped_auroc_and_aupr(in_dist, out_dist, num_bootstrap_samples)
        
        time_per_item = np.mean([results[exp]['time_per_item'] 
                                 for exp in exp_list])
        time_per_item_std = np.sqrt( 
            np.mean(
                [
                    (results[exp]['time_per_item_std']*np.sqrt(len(results[exp]['uncs'])))**2 
                    for exp in exp_list
                ]
            ) 
        )
        time_per_item_std /= np.sqrt(np.sum([len(results[exp]['uncs']) 
                                             for exp in exp_list]))
        
        stats['runtime'] = time_per_item*1000
        stats['runtime_conf'] = time_per_item_std*1000*1.96
        
        summary[name] = stats
        
    return summary

def generate_table_block(summarized_results, metrics=['auroc'],
                         keys_to_compare=None):
    if keys_to_compare is None:
        keys_to_compare = summarized_results.keys()
        
    title_line = "Method & \multicolumn{2}{c}{Runtime (ms)} " 
    for metric in metrics:
        title_line += "& \multicolumn{2}{c}{%s}" % metric
    print(title_line)
    
    for i,name in enumerate(keys_to_compare):
        if name not in summarized_results:
            continue
        stats = summarized_results[name]
    
        table_line = "& " + name
        
        runtime = stats['runtime']
        runtime_conf = stats['runtime_conf']
        table_line += r"& $%0.3f$ & $\pm%0.3f$" % (runtime, runtime_conf)
        
        for metric in metrics:
            mean_metric = stats[metric]
            mean_metric_conf = stats[metric+"_conf"]
            
            table_line += r"& $%0.3f$ & $\pm%0.3f$" % (mean_metric, mean_metric_conf)
        table_line += r" \\"
        print(table_line)

        
def plot_perf_vs_runtime(summarized_results, metric='auroc', 
                         keys_to_compare=None, colors=None, 
                         normalize_x=True, **kwargs):
    if keys_to_compare is None:
        keys_to_compare = summarized_results.keys()
    
    np.random.seed(1)
    
    fig, ax = plt.subplots(**kwargs)
    
    for i,name in enumerate(keys_to_compare):
        if name not in summarized_results:
            continue
        stats = summarized_results[name]
                
        runtime = stats['runtime']
        runtime_conf = stats['runtime_conf']
        mean_metric = stats[metric]
        mean_metric_conf = stats[metric+"_conf"]
        
        if colors is not None:
            color = colors[i]
        else:
            color = 'C'+str(i)
        
        ax.set_ylabel(metric.upper())
        ax.set_xlabel("Runtime (ms)")
        ax.errorbar(runtime, mean_metric, 
                    xerr=runtime_conf, yerr=mean_metric_conf, 
                    linestyle='', marker='o', label=name, color=color, capsize=2)
                    
    if normalize_x:
        _,b = plt.xlim()
        ax.set_xlim(0,b)

    ax.grid()
    plt.tight_layout()
    
    
def plot_transform_sweep(results_dict, keys_to_compare=None, **kwargs):
    if keys_to_compare is None:
        keys_to_compare = results_dict.keys()
                
    summary = {}
    for name in keys_to_compare:
        if name not in results_dict:
            continue
        
        results = results_dict[name]


        df = pd.DataFrame(results)
        df = df.drop(index=['metric_name', 'time_per_item', 'time_per_item_std'])
        df = df.swapaxes("index","columns").apply(pd.Series.explode).rename_axis('transform').reset_index().infer_objects()
        sns.jointplot(
            data = df,
            x = "uncs",
            y = "metrics",
            hue = "transform",
            s=10,
        )
        plt.title(name)
        plt.show()
            
        
            
            
            
            