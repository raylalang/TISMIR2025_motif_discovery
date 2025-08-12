import sys, os, json
import numpy as np

from scipy import stats

if __name__ == '__main__':
        proposed_pl_acc = [0.8310, 0.8313, 0.8351]
        proposed_mi_acc = [0.8392, 0.8365, 0.8410]
        # proposed_pl_acc = [0.831, 0.831, 0.835]
        # proposed_mi_acc = [0.839, 0.837, 0.841]

        proposed_pl_f1 = [0.7160, 0.7129, 0.7147]
        proposed_mi_f1 = [0.7211, 0.7200, 0.7234]
        # proposed_pl_f1 = [0.716, 0.713, 0.715]
        # proposed_mi_f1 = [0.721, 0.720, 0.723]

        interval_pl = stats.t.interval(0.95, len(proposed_pl_acc)-1
            , loc=np.mean(proposed_pl_acc), scale=stats.sem(proposed_pl_acc))
        interval_mi = stats.t.interval(0.95, len(proposed_mi_acc)-1
            , loc=np.mean(proposed_mi_acc), scale=stats.sem(proposed_mi_acc))
        # interval3 = stats.t.interval(0.95, len(combined_scores3)-1
        #     , loc=np.mean(combined_scores3), scale=stats.sem(combined_scores3))

        print ('=====Accuracy=====')
        print ("PL: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(
            interval_pl[0], interval_pl[1]
            , np.mean(proposed_pl_acc), np.mean(proposed_pl_acc) - interval_pl[0]))

        print ("MI: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(
            interval_mi[0], interval_mi[1]
            , np.mean(proposed_mi_acc), np.mean(proposed_mi_acc) - interval_mi[0]))
        # print ("PL: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(
        #     interval_pl[0], interval_pl[1]
        #     , np.mean(proposed_pl_acc), np.mean(proposed_pl_acc) - interval_pl[0]))

        result = stats.ttest_ind(proposed_mi_acc, proposed_pl_acc, equal_var=False, nan_policy='propagate', alternative='greater')
        print ("Ind t-test for MI > PL result:", result)

        # result = stats.ttest_ind(combined_scores2, combined_scores1, equal_var=False, nan_policy='propagate', alternative='greater')
        # print ("Ind t-test for #2 > #1 result:", result)

        interval_pl = stats.t.interval(0.95, len(proposed_pl_f1)-1
            , loc=np.mean(proposed_pl_f1), scale=stats.sem(proposed_pl_f1))
        interval_mi = stats.t.interval(0.95, len(proposed_mi_f1)-1
            , loc=np.mean(proposed_mi_f1), scale=stats.sem(proposed_mi_f1))
        # interval3 = stats.t.interval(0.95, len(combined_scores3)-1
        #     , loc=np.mean(combined_scores3), scale=stats.sem(combined_scores3))

        print ('=====F1-score=====')
        print ("PL: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(
            interval_pl[0], interval_pl[1]
            , np.mean(proposed_pl_f1), np.mean(proposed_pl_f1) - interval_pl[0]))

        print ("MI: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(
            interval_mi[0], interval_mi[1]
            , np.mean(proposed_mi_f1), np.mean(proposed_mi_f1) - interval_mi[0]))
        # print ("PL: {:.4f} ~ {:.4f} mean: {:.4f} +- {:.4f}".format(
        #     interval_pl[0], interval_pl[1]
        #     , np.mean(proposed_pl_f1), np.mean(proposed_pl_f1) - interval_pl[0]))

        result = stats.ttest_ind(proposed_mi_f1, proposed_pl_f1, equal_var=False, nan_policy='propagate', alternative='greater')
        print ("Ind t-test for MI > PL result:", result)

        # result = stats.ttest_ind(combined_scores2, combined_scores1, equal_var=False, nan_policy='propagate', alternative='greater')
        # print ("Ind t-test for #2 > #1 result:", result)