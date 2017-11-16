from loaders.loading import load_train, load_external
from sklearn.model_selection import train_test_split
from pipeline import transform

SUBM_PATH = r'../input/norm_challenge_ru'
INPUT_PATH = r'../input/norm_challenge_ru'
DATA_INPUT_PATH = r'../input/norm_challenge_ru/ru_with_types'

df = load_train(columns=['class', 'before', 'after'], input_path=INPUT_PATH)
df['prev'] = df['before'].shift(1)
df['next'] = df['before'].shift(-1)
df = df.fillna('')
print(df.info())

x_train, x_test, y_train, y_test = train_test_split(df.drop(['after'], axis=1), df['after'], test_size=0.1, random_state=2017)
del df
del x_test['class']

transform(x_train, x_test, y_train, y_test)