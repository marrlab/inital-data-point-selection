
# datasets
DATASETS_ORDER = ['matek', 'isic', 'retinopathy', 'jurkat'] 

# criteria
OUR_CRITERIA_ORDER_NO_RANDOM = ['furthest', 'closest', 'half_in_half', 'fps']
CRITERIA_ORDER_NO_RANDOM = ['cold_paws'] + OUR_CRITERIA_ORDER_NO_RANDOM 
CRITERIA_ORDER = ['random'] + CRITERIA_ORDER_NO_RANDOM
HALF_IN_HALF_TRAIN_SAMPLES_TO_CLUSTERS = {
    10: 5, 100: 50, 200: 100, 500: 250
}
ANNOTATION_BUDGETS = [100, 200, 500]

# metrics
REQUIRED_METRICS = [
    'test_f1_macro_epoch_end',
    'test_accuracy_epoch_end',
    'test_balanced_accuracy_epoch_end',
    'test_matthews_corrcoef_epoch_end',
    'test_cohen_kappa_score_epoch_end',
    'test_roc_auc_curve_epoch_end',
    'test_pr_auc_curve_epoch_end',
]
REQUIRED_METRICS_SHORT_NAME = [
    'F1-MACRO',
    'ACC',
    'BACC',
    'MCC',
    'KAPPA',
    'ROC AUC',
    'PR AUC',
]
REQUIRED_METRICS_FULL_NAME = [
    'f1-macro',
    'accuracy',
    'balanced accuracy',
    'Matthew\'s correlation coefficient',
    'Cohen\'s kappa',
    'area under receiver operating characteristic curve',
    'area under precision-recall curve',
]
SSL_ORDER = [
    'SimCLR',
    'SwAV',
    'DINO'
]
