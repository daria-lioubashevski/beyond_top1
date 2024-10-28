import argparse
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
from consts import MIN_NUM_EMBDS_IN_CLASS
from utils import write_results


def load_train_data(pkl_path, num_tasks):
    with open(pkl_path, 'rb') as fp:
        layer_embd_dict = pickle.load(fp)

    all_layers_embd = []
    labels = []
    for layer in layer_embd_dict:
        min_len = min(len(layer_embd_dict[layer][i]) for i in range(num_tasks))
        if min_len < MIN_NUM_EMBDS_IN_CLASS:
            continue
        for i in range(num_tasks):
            # balance data by taking same number of samples from each class (task)
            all_layers_embd += layer_embd_dict[layer][i][:min_len]
            labels += [i] * min_len

    features = np.vstack(all_layers_embd)
    labels = np.array(labels)
    return features, labels


def create_random_features(features, labels, num_classes):
    random_features = np.zeros(features.shape)
    for i in range(num_classes):
        class_idxs = np.where(labels == i)[0]
        class_features = features[class_idxs, :]
        class_mean = np.mean(class_features)
        class_std = np.std(class_features)
        new_features = np.random.normal(class_mean, class_std, (len(class_idxs), features.shape[1]))
        random_features[class_idxs, :] = new_features
    return random_features


def get_train_test_indxs(kfolds, folds, i):
    folds_ixs = np.roll(range(kfolds), i)
    test_fold = folds_ixs[-1]
    train_folds = folds_ixs[:-1]
    test_indxs = folds[test_fold]
    train_indxs = np.concatenate([folds[j] for j in train_folds])
    return train_indxs, test_indxs


def train_logistic_regression(X_train, Y_train):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, multi_class="ovr"))

    clf.fit(X_train, Y_train)
    return clf


def calc_logistic_metrics(clf, X_test, Y_test, num_tasks):
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    acc_score = balanced_accuracy_score(Y_test, y_pred)

    # calculate roc_auc score per class
    auc_scores = []
    for i in range(num_tasks):
        cur_proba = y_proba[:, i]
        cur_label = [1 if y == i else 0 for y in Y_test]
        auc_scores.append(roc_auc_score(cur_label, cur_proba))

    # calculate accuracy per class
    cm = confusion_matrix(Y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc_per_class = cm.diagonal()

    return acc_score, auc_scores, acc_per_class


def run_logistic_regression_probing(features, labels, num_tasks, kfolds=5, save_clf=False):
    num_samples = len(labels)
    acc_arr = []
    skf = KFold(n_splits=kfolds, shuffle=True)
    folds = [t[1] for t in skf.split(np.arange(num_samples))]
    avg_roc_auc_scores = np.zeros(num_tasks)
    avg_acc_per_class = np.zeros(num_tasks)
    clfs = []

    for i in range(kfolds):
        train_indxs, test_indxs = get_train_test_indxs(kfolds, folds, i)
        X_train = features[train_indxs, :]
        X_test = features[test_indxs, :]
        Y_train = labels[train_indxs]
        Y_test = labels[test_indxs]

        clf = train_logistic_regression(X_train, Y_train)
        clfs.append(clf)
        acc_score, auc_scores, acc_per_class = calc_logistic_metrics(clf, X_test, Y_test, num_tasks)
        acc_arr.append(acc_score)
        avg_roc_auc_scores += np.array(auc_scores) / kfolds
        avg_acc_per_class += np.array(acc_per_class) / kfolds

    if save_clf:
        best_clf_ind = np.argmax(acc_arr)
        with open(f'top_{num_tasks}_task_clf.pkl', 'wb') as f:
            pickle.dump(clfs[best_clf_ind], f)

    return np.mean(acc_arr), np.std(acc_arr) / np.sqrt(len(acc_arr)), avg_roc_auc_scores, avg_acc_per_class


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, help="path to pkl containing training data")
    parser.add_argument("-n", "--num_tasks", type=int, help="number of tasks (should probably be between 3 to 5)")
    parser.add_argument("-k", "--kfolds", type=int, default=5, help="number of kfolds")
    parser.add_argument("--save_clf", action="store_true", help="save trained probing classifier")
    parser.add_argument("-o", "--output_path", type=str)
    return parser.parse_args()


def main(args):
    features, labels = load_train_data(args.data_path, args.num_tasks)
    avg_acc, acc_ste, avg_roc_auc_scores, avg_acc_per_class = run_logistic_regression_probing(features, labels,
                                                                                              args.num_tasks,
                                                                                              args.kfolds,
                                                                                              args.save_clf)
    results = []
    results.append('------------------------- Layer Embeddings -------------------------')
    results.append(f'Overall average accuracy: {round(avg_acc, 3)} \u00B1 {round(acc_ste, 4)}')
    results.append(f'Average accuracy per task: {[round(x, 3) for x in avg_acc_per_class]}')
    results.append(f'Average ROC-AUC per task: {[round(x, 3) for x in avg_roc_auc_scores]}\n\n')

    random_features = create_random_features(features, labels, args.num_tasks)
    avg_acc, acc_ste, avg_roc_auc_scores, avg_acc_per_class = run_logistic_regression_probing(random_features, labels,
                                                                                              args.num_tasks,
                                                                                              args.kfolds,
                                                                                              args.save_clf)
    results.append('------------------------- Random Embeddings -------------------------')
    results.append(f'Overall average accuracy: {round(avg_acc, 3)} \u00B1 {round(acc_ste, 4)}')
    results.append(f'Average accuracy per task: {[round(x, 3) for x in avg_acc_per_class]}')
    results.append(f'Average ROC-AUC per task: {[round(x, 3) for x in avg_roc_auc_scores]}')
    write_results(results, args.output_path)


if __name__ == '__main__':
    args = args_parse()
    main(args)
