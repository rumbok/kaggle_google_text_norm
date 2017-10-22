import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import seaborn as sns


dtrain = xgb.DMatrix('models/class.txt.train#dtrain.cache')
dtest = xgb.DMatrix('models/class.txt.test#dtest.cache')

num_classes = len(set(dtrain.get_label()))

xgb_param = {
    'learning_rate': 0.1,
    'num_boost_round': 1000,
    'max_depth': 5,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softmax',
    'num_class': num_classes,
    'nthread': 4,
    'scale_pos_weight': 1,
    'seed': 2017}
cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=xgb_param['num_boost_round'],
                  stratified=True, nfold=10,
                  metrics=['merror'], early_stopping_rounds=100, verbose_eval=True)
print('Best num_boost_round value', cvresult.shape[0])

pyplot.errorbar(cvresult.index, cvresult['train-merror-mean'],
                yerr=cvresult['train-merror-std'], ecolor='r')
pyplot.errorbar(cvresult.index, cvresult['test-merror-mean'],
                yerr=cvresult['test-merror-std'], ecolor='g')
pyplot.title("XGBoost num_boost_round vs MError")
pyplot.xlabel('num_boost_round')
pyplot.ylabel('MError')
pyplot.savefig('num_boost_round.png')