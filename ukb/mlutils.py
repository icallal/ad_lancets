import pandas as pd
import numpy as np
import sys
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_curve,
    auc,
    matthews_corrcoef,
)
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import os
# from utils import save_pickle
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
#import pyarrow as pa


def concat_labels_and_probas(dirpath):
    true_labels = []
    probas = []

    if "mci" not in dirpath:
        if "nacc" not in dirpath:
            for i in range(10):
                tl = pickle.load(
                    open(f"{dirpath}/test_true_labels_region_{i}.pkl", "rb")
                )
                true_labels.append(tl[0])

                p = pickle.load(open(f"{dirpath}/test_probas_region_{i}.pkl", "rb"))

                if "feature_selection" in dirpath:
                    df = pd.read_csv(f"{dirpath}/training_results_region_{i}.csv")
                    df = df.iloc[:20]
                    best_idx = df["auroc"].idxmax()
                    probas.append(p[best_idx])
                else:
                    probas.append(p[0])
        else:
            # Handle NACC data
            tl = pickle.load(open(f"{dirpath}/test_true_labels_region_9.pkl", "rb"))
            p = pickle.load(open(f"{dirpath}/test_probas_region_9.pkl", "rb"))

            for i in range(10):
                true_labels.append(tl[i])
                probas.append(p[i])
    else:
        # Handle MCI data
        tl = pickle.load(open(f"{dirpath}/test_true_labels.pkl", "rb"))
        true_labels.append(tl[0])

        p = pickle.load(open(f"{dirpath}/test_probas.pkl", "rb"))
        probas.append(p[0])

    return true_labels, probas


def mcc_from_conf_mtx(tp, fp, tn, fn):
    return (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def encode_categorical_vars(df, catcols):
    enc = OneHotEncoder(drop="if_binary")
    enc.fit(df.loc[:, catcols])
    categ_enc = pd.DataFrame(
        enc.transform(df.loc[:, catcols]).toarray(),
        columns=enc.get_feature_names_out(catcols),
    )
    return categ_enc


def encode_ordinal_vars(df, ordvars):
    enc = OrdinalEncoder()
    enc.fit(df.loc[:, ordvars])
    ord_enc = pd.DataFrame(
        enc.transform(df.loc[:, ordvars]), columns=enc.get_feature_names_out(ordvars)
    )
    return ord_enc


def pick_threshold(y_true, y_probas, youden=True, beta=1):
    scores = []

    if youden is True:
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_true, y_probas)

        for i, t in enumerate(thresholds):
            # youden index = sensitivity + specificity - 1
            # AKA sensitivity + (1 - FPR) - 1 (NOTE: (1-FPR) = TNR)
            # AKA recall_1 + recall_0 - 1
            youdens_j = tpr[i] + (1 - fpr[i]) - 1
            scores.append(youdens_j)

    else:
        # calculate pr-curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_probas)

        # convert to f score
        for i, t in enumerate(thresholds):
            fscore = ((1 + beta**2) * precision[i] * recall[i]) / (
                (beta**2 * precision[i]) + recall[i]
            )
            scores.append(fscore)

    ix = np.nanargmax(scores)
    best_threshold = thresholds[ix]

    return best_threshold


# def calc_results(metric, y_true, y_probas, youden=False, beta=1, threshold=None):
def calc_results(
    y_true, y_probas, youden=True, beta=1, threshold=None, suppress_output=True
):
    auroc = roc_auc_score(y_true, y_probas)
    ap = average_precision_score(y_true, y_probas)

    # if metric == 'roc_auc':
    #     youden = True

    # return_threshold = False
    # if threshold is None:
    #     threshold = pick_threshold(y_true, y_probas, youden, beta)
    #     return_threshold = True

    return_threshold = False
    if threshold is not None:
        pass
    else:
        threshold = pick_threshold(y_true, y_probas, youden, beta)
        return_threshold = True

    test_pred = (y_probas >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, test_pred).ravel()
    acc = accuracy_score(y_true, test_pred)
    bal_acc = balanced_accuracy_score(y_true, test_pred)
    prfs = precision_recall_fscore_support(y_true, test_pred, beta=beta)
    mcc = matthews_corrcoef(y_true, test_pred)

    # print(f'AUROC: {auroc}, AP: {ap}, Fscore: {best_fscore}, Accuracy: {acc}, Bal. Acc.: {bal_acc}, Best threshold: {best_threshold}')
    if suppress_output:
        pass
    else:
        print(
            f"AUROC: {np.round(auroc, 4)}, AP: {np.round(ap, 4)}, \nAccuracy: {np.round(acc, 4)}, Bal. Acc.: {np.round(bal_acc, 4)}, \nBest threshold: {np.round(threshold, 4)}"
        )
        print(f"Precision/Recall/Fscore: {prfs}")
        print("\n")
    res = pd.Series(
        data=[
            auroc,
            ap,
            threshold,
            tp,
            tn,
            fp,
            fn,
            acc,
            bal_acc,
            prfs[0][0], # precision negative (NPV)
            prfs[0][1], # precision positive (PPV)
            prfs[1][0], # recall negative (Sensitivity)
            prfs[1][1], # recall positive (Specificity)
            prfs[2][0], # fbeta negative
            prfs[2][1], # fbeta positive
            mcc,
        ],
        index=[
            "auroc",
            "avg_prec",
            "threshold",
            "TP",
            "TN",
            "FP",
            "FN",
            "accuracy",
            "bal_acc",
            "prec_n",
            "prec_p",
            "recall_n",
            "recall_p",
            f"f{beta}_n",
            f"f{beta}_p",
            "mcc",
        ],
    )
    if return_threshold == True:
        return res, threshold
    else:
        return res
    # return res


def save_labels_probas(
    filepath,
    train_labels,
    train_probas,
    test_labels,
    test_probas,
    other_file_info="",
    survival=False,
    surv_model=None,
    train_surv_fn=None,
    test_surv_fn=None,
):
    save_pickle(f"{filepath}/train_true_labels{other_file_info}.pkl", train_labels)
    save_pickle(f"{filepath}/train_probas{other_file_info}.pkl", train_probas)
    save_pickle(f"{filepath}/test_true_labels{other_file_info}.pkl", test_labels)
    save_pickle(f"{filepath}/test_probas{other_file_info}.pkl", test_probas)

    if survival is True:
        save_pickle(f"{filepath}/surv_model{other_file_info}.pkl", surv_model)

        train_surv_fn = pd.DataFrame(train_surv_fn)
        test_surv_fn = pd.DataFrame(test_surv_fn)

        print("Saving training survival functions")
        start_time = datetime.now()
        train_surv_fn.to_parquet(
            f"{filepath}/train_survival_fns{other_file_info}.parquet", engine="pyarrow"
        )
        end_time = datetime.now()
        print(f"pyarrow, Time to save: {end_time - start_time}")

        print("Saving test survival functions")
        start_time = datetime.now()
        test_surv_fn.to_parquet(
            f"{filepath}/test_survival_fns{other_file_info}.parquet", engine="pyarrow"
        )
        end_time = datetime.now()
        print(f"pyarrow, Time to save: {end_time - start_time}")

        # pa_table = pa.table({"train_survival_functions": train_surv_fn})
        # pa.parquet.write_table(pa_table, f"{filepath}/train_survival_fns{other_file_info}.parquet")

        # pa_table = pa.table({"test_survival_functions": test_surv_fn})
        # pa.parquet.write_table(pa_table, f"{filepath}/test_survival_fns{other_file_info}.parquet")
        # np.save(f'{filepath}/train_survival_fns{other_file_info}.npy', train_surv_fn, allow_pickle=False)
        # np.save(f'{filepath}/test_survival_fns{other_file_info}.npy', test_surv_fn, allow_pickle=False)


# def get_fold_number(fname):
#     last_underscore = fname.rfind('_')
#     last_period = fname.rfind('.')
#     fold = fname[last_underscore+1:last_period]
#     return fold

# def sort_fold_results(fold_numbers, fold_results):
#     # Pair strings with their corresponding numbers
#     paired_list = list(zip(fold_numbers, fold_results))

#     # Sort the paired list based on the numbers
#     sorted_paired_list = sorted(paired_list)

#     # Extract the sorted strings
#     sorted_results = [fold_result for fold_number, fold_result in sorted_paired_list]
#     sorted_results = pd.concat(sorted_results)

#     return sorted_results


def concat_results(filepath):
    train_results = []
    test_results = []

    for i in range(10):
        train_results.append(
            pd.read_csv(f"{filepath}/training_results_region_{i}.csv", index_col=0)
        )
        test_results.append(
            pd.read_csv(f"{filepath}/test_results_region_{i}.csv", index_col=0)
        )

    train_results = pd.concat(train_results)
    test_results = pd.concat(test_results)
    return train_results, test_results
    # for fname in file_list:
    #     if fname[:2] == '._':
    #         continue
    #     if '.csv' in fname:
    #         if 'training_results' in fname and 'region' in fname:
    #             train_results.append(pd.read_csv(f'{filepath}/{fname}', index_col=0))

    #             fold = get_fold_number(fname)
    #             train_fold.append(fold)

    #         elif 'test_results' in fname and 'region' in fname:
    #             test_results.append(pd.read_csv(f'{filepath}/{fname}', index_col=0))

    #             fold = get_fold_number(fname)
    #             test_fold.append(fold)

    # train_results = sort_fold_results(train_fold, train_results)
    # test_results = sort_fold_results(test_fold, test_results)


def concat_and_save_results(filepath):

    train_results, test_results = concat_results(filepath)

    train_results.to_csv(f"{filepath}/train_results.csv")
    test_results.to_csv(f"{filepath}/test_results.csv")


def probas_to_results(filepath, youden=True, beta=1, threshold=None):
    train_res_l = []
    test_res_l = []

    if "mci" not in filepath:

        if "nacc" not in filepath:
            for i in range(10):
                train_labels = pickle.load(
                    open(f"{filepath}/train_true_labels_region_{i}.pkl", "rb")
                )
                test_labels = pickle.load(
                    open(f"{filepath}/test_true_labels_region_{i}.pkl", "rb")
                )
                train_probas = pickle.load(
                    open(f"{filepath}/train_probas_region_{i}.pkl", "rb")
                )
                test_probas = pickle.load(
                    open(f"{filepath}/test_probas_region_{i}.pkl", "rb")
                )

                if "feature_selection" in filepath:
                    df = pd.read_csv(f"{filepath}/training_results_region_{i}.csv")
                    df = df.iloc[:20]
                    best_idx = df["auroc"].idxmax()

                    # res = calc_results(test_labels[0], test_probas[best_idx], youden=youden, beta=beta, threshold=threshold)
                    # res_l.append(res)

                    train_res, thresh = calc_results(
                        train_labels[0],
                        train_probas[best_idx],
                        youden=youden,
                        beta=beta,
                        threshold=threshold,
                    )
                    res = calc_results(
                        test_labels[0],
                        test_probas[best_idx],
                        youden=youden,
                        beta=beta,
                        threshold=thresh,
                    )
                    train_res_l.append(train_res)
                    test_res_l.append(res)

                else:
                    train_res, thresh = calc_results(
                        train_labels[0],
                        train_probas[0],
                        youden=youden,
                        beta=beta,
                        threshold=threshold,
                    )
                    res = calc_results(
                        test_labels[0],
                        test_probas[0],
                        youden=youden,
                        beta=beta,
                        threshold=thresh,
                    )
                    train_res_l.append(train_res)
                    test_res_l.append(res)
        else:
            train_labels = pickle.load(
                open(f"{filepath}/train_true_labels_region_9.pkl", "rb")
            )
            test_labels = pickle.load(
                open(f"{filepath}/test_true_labels_region_9.pkl", "rb")
            )
            train_probas = pickle.load(
                open(f"{filepath}/train_probas_region_9.pkl", "rb")
            )
            test_probas = pickle.load(
                open(f"{filepath}/test_probas_region_9.pkl", "rb")
            )

            for i in range(10):
                train_res, thresh = calc_results(
                    train_labels[i],
                    train_probas[i],
                    youden=youden,
                    beta=beta,
                    threshold=threshold,
                )
                res = calc_results(
                    test_labels[i],
                    test_probas[i],
                    youden=youden,
                    beta=beta,
                    threshold=thresh,
                )
                train_res_l.append(train_res)
                test_res_l.append(res)

    else:
        test_labels = pickle.load(open(f"{filepath}/test_true_labels.pkl", "rb"))
        test_probas = pickle.load(open(f"{filepath}/test_probas.pkl", "rb"))
        res = calc_results(
            test_labels[0],
            test_probas[0],
            youden=youden,
            beta=beta,
            threshold=threshold,
        )
        test_res_l.append(res)

    train_results = pd.concat(train_res_l, axis=1).T
    test_results = pd.concat(test_res_l, axis=1).T

    return train_results, test_results


# if __name__ == "__main__":
#    if len(sys.argv) > 1:
#        function_name = sys.argv[1]
#        args = sys.argv[2:]
#        if function_name in globals():
#            globals()[function_name](*args)
#        else:
#            print(f"No function named '{function_name}' found.")
#    else:
#        print("No function name provided.")


from sklearn.metrics import (
    RocCurveDisplay,
    roc_curve,
    auc,
    roc_auc_score,
    d2_absolute_error_score,
    d2_pinball_score,
    d2_tweedie_score,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_tweedie_deviance,
    mean_pinball_loss,
    root_mean_squared_error,
    root_mean_squared_log_error,
)


def calculate_regression_metrics(y_true, y_pred, tweedie_power=0):
    """
    Calculates various regression metrics for predictions and true values.

    Parameters:
    - y_true: array-like of shape (n_samples,) True target values.
    - y_pred: array-like of shape (n_samples,) Predicted target values.
    - tweedie_power: Power parameter for Tweedie distribution deviance and D2 Tweedie score.
        Default is 0 (Gaussian).

    Returns:
    - metrics_dict: Dictionary containing all calculated metrics.
    """
    metrics_dict = {
        "r2_score": r2_score(y_true, y_pred),
        "median_absolute_error": median_absolute_error(y_true, y_pred),
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "mean_squared_error": mean_squared_error(y_true, y_pred),
        "d2_absolute_error_score": d2_absolute_error_score(y_true, y_pred),
        "d2_pinball_score": d2_pinball_score(y_true, y_pred, alpha=0.5),
        "d2_tweedie_score": d2_tweedie_score(y_true, y_pred, power=tweedie_power),
        "explained_variance_score": explained_variance_score(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(
            y_true, y_pred
        ),
        "mean_pinball_loss": mean_pinball_loss(y_true, y_pred, alpha=0.5),
        "root_mean_squared_error": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

    # check if all y_true and y_pred values are positive for mean_gamma_deviance
    # if all(y_true >= 0):
    # metrics_dict["root_mean_squared_log_error"] = np.sqrt(mean_squared_log_error(y_true, y_pred))

    # check if all y_true and y_pred values are positive for mean_gamma_deviance
    if all(y_true > 0) and all(y_pred > 0):
        metrics_dict["root_mean_squared_log_error"] = np.sqrt(
            mean_squared_log_error(y_true, y_pred)
        )
        metrics_dict["mean_squared_log_error"] = mean_squared_log_error(y_true, y_pred)
        metrics_dict["mean_gamma_deviance"] = mean_gamma_deviance(y_true, y_pred)
        metrics_dict["mean_tweedie_deviance"] = mean_tweedie_deviance(
            y_true, y_pred, power=tweedie_power
        )
        metrics_dict["mean_poisson_deviance"] = mean_poisson_deviance(y_true, y_pred)
    return metrics_dict
