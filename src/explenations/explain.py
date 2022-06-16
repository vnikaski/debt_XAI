from sklearn.inspection import permutation_importance


def explain_permutation_importance(model, x_test, y_test):
    r = permutation_importance(model, x_test, y_test, n_repeats=30, random_state=0, scoring='f1')
    print("PERMUTATION IMPORTANCE")
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] > 0.08:
            print(f"{x_test.columns[i]:<30} {r.importances_mean[i]:.3f} +/- {r.importances_std[i]:.3f}")
    print()
