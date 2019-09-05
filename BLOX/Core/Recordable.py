import inspect
import ignite.metrics as _metrics
from ignite.metrics.metric import Metric
from sklearn import metrics
METRICS = dict(inspect.getmembers(_metrics))
del METRICS['ConfusionMatrix']
del METRICS['Accuracy']

import numpy as np
from socket import error as socket_error
from BLOX.Common.utils import try_func

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
def AUC(y_h, y):
    fpr, tpr, thresholds = metrics.roc_curve(y.detach().view(-1).numpy(), y_h.detach().view(-1).numpy(), pos_label=y.argmax().item())
    print(tpr,fpr)
    return metrics.auc(fpr,tpr)
def cm(y_h,y):
    return metrics.confusion_matrix(np.asarray(y).argmax(axis=1),np.asarray(y_h).argmax(axis=1))
def accuracy(y_h,y):return (sum( [1. if (y[i].argmax() if len(y) > 1 else y[0] ) == y_h[i].argmax() else 0. for i in range(len(y))] ) /float(len(y)))*100.
def roc(y_h,y):
    '''
        since we support multiclass classification and sklearn only does binary classification,
        we will have to use whether the prediction was correct or not
    '''
    return metrics.roc_auc_score(np.asarray(y),np.asarray(y_h).max(axis=1))

def f1_macro(y_h,y):
    y = [1 if y[i].argmax() == np.argmax(y_h[i]) else 0  for i in range(len(y)) ]

    return metrics.f1_score(np.asarray(y),np.asarray(y_h).argmax(axis=1),average='macro')

# @try_func(False,False)
def cc(y_h,y, n_classes = None):
    y = np.asarray(y)
    y_h = np.asarray(y_h)
    if not n_classes:n_classes = y_h.shape[-1] if y.shape[-1] > 1 else y.max()
    
    y_idx = np.argmax(y,axis=1)
    y_h_idx = np.argmax(y_h,axis=1)
    clf_scores = []
    mpvs = []
    fops = []
    ps = []
    for i in range(n_classes):
        _y = np.zeros(y.shape[0])
        _y[ i == y_idx ] = 1.0
        y_pred = np.zeros(y.shape[0])
        y_pred[ i== y_h_idx ] == 1.0
        p = y_h[:,i]
        p = (p-p.min())/(p.max() - p.min())     if p.min() != p.max() else p
        clf_scores.append(metrics.brier_score_loss(_y, p ))
        fraction_of_positives, mean_predicted_value = calibration_curve(_y, p, n_bins=10)
        mpvs.append(mean_predicted_value)
        fops.append(fraction_of_positives)
        ps.append(p)

    return clf_scores,fops,mpvs,ps

def ev(y_h,y):
    y = np.asarray(y)
    y_h = np.asarray(y_h)
    return [ metrics.explained_variance_score(y[:,i], y_h[:,i], multioutput='uniform_average') for i in range(y.shape[-1])]

def cks(y_h,y):
    y = np.asarray(y)
    y_h = np.asarray(y_h)
    return metrics.cohen_kappa_score(y.argmax(axis=1),y_h.argmax(axis=1))

def hgs(y_h,y):
    y = np.asarray(y)
    y_h = np.asarray(y_h)
    return metrics.homogeneity_score(y.argmax(axis=1),y_h.argmax(axis=1))

def abde(y_h,y):
    y = np.asarray(y)
    y_h = np.asarray(y_h)
    
    m = []
    s = []
    for i in range(y.shape[-1]):
        m.append( (y[:,i] - y_h[:,i]).mean() )
        s.append( (y[:,i] - y_h[:,i]).std() )
    return m,s

EPOCH_METRICS = {}
EPOCH_METRICS['F1_MACRO'] = f1_macro
EPOCH_METRICS['ExplainedVariance'] = ev

ADDITIONAL_METRICS = {}
ADDITIONAL_METRICS['AUC'] = AUC
ADDITIONAL_METRICS['ConfusionMatrix'] = cm
ADDITIONAL_METRICS['ROC_Curve'] = roc
ADDITIONAL_METRICS['ConfidenceCurves'] = cc
ADDITIONAL_METRICS['CohenKappa'] = cks
ADDITIONAL_METRICS['HomogeneityScore'] = hgs
ADDITIONAL_METRICS['AbsoluteDifferenceError'] = abde
ADDITIONAL_METRICS['Accuracy'] = accuracy


class EpochMetrics:pass



class Vizzy:

    def __init__(self):
        try:
            from visdom import Visdom
            viz = Visdom(port=8097, server='http://localhost')
        except socket_error as e:
            viz = None

    @staticmethod
    def Accuracy(self,x):
        if self.viz:
            self.viz.text('Accuracy: {:.2f}%'.format(float(x)))

    # @try_func(False,False)
    @staticmethod
    def ConfusionMatrix(self,x):
        if self.viz: self.viz.heatmap(
            X=x,
            opts=dict(
                title='ConfusionMatrix',
                colormap='Electric',
            )
        )
    # @try_func(False,False)
    @staticmethod
    def ROC_Curve(self,args):
        (fpr,tpr,threshold) = args
        print(fpr)
        print(tpr)
        print(threshold)
    # @try_func(False,False)
    @staticmethod
    def HomogeneityScore(self,x):
        print(x)
    # @try_func(False,False)
    @staticmethod
    def AbsoluteDifferenceError(self,args):
        if self.viz:
            import matplotlib.pyplot as plt
            (m,s) = args
            fig, ax = plt.subplots()
            x = np.arange(len(m))
            ax.bar(x, m, yerr=s, align='center', alpha=0.5, ecolor='black')
            ax.set_title('Mean Error in inference w/ Std')
            plt.tight_layout()
            self.viz.matplot(plt)
    # @try_func(False,False)
    @staticmethod
    def F1_MACRO(self,args):
        print(args)
    # @try_func(False,False)
    @staticmethod
    def ExplainedVariance(self,x):
        print(x)
    # @try_func(False,False)
    @staticmethod
    def CohenKappa(self,x):
        print(x)
    # @try_func(False,False)
    @staticmethod
    def ConfidenceCurves(self,args):
        if self.viz:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10, 10))
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))
            (clf_scores, fops,mpvs,ps) = args
            for i in range(len(clf_scores)):
                name = 'IDX {}'.format(i)
                fraction_of_positives = fops[i]
                mean_predicted_value = mpvs[i]
                clf_score = clf_scores[i]
                prob_pos = ps[i]
                ax1.plot(mean_predicted_value, fraction_of_positives, "o-",
                        label="%s (%1.3f)" % (name, clf_score))

                ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                        histtype="step", lw=2)

            ax1.set_ylabel("Fraction of positives")
            ax1.set_ylim([-0.05, 1.05])
            ax1.legend(loc="lower right")
            ax1.set_title('Confidence Curves')

            ax2.set_xlabel("Mean predicted value")
            ax2.set_ylabel("Count")
            ax2.legend(loc="lower center", ncol=int(len(mpvs)/2))

            plt.tight_layout()
            self.viz.matplot(plt)



