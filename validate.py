from loaders.loading import load_train, load_external
from sklearn.model_selection import train_test_split
from pipeline import transform

SUBM_PATH = r'../input/norm_challenge_ru'
INPUT_PATH = r'../input/norm_challenge_ru'

# df = load_train(columns=['class', 'before', 'after'], input_path=INPUT_PATH)
df = load_external(columns=['class', 'before', 'after'], input_path=INPUT_PATH)
df['before_prev'] = df['before'].shift(1)
df['before_next'] = df['before'].shift(-1)
df = df.fillna('')
print(df.info())

x_train, x_test, y_train, y_test = train_test_split(df.drop(['after'], axis=1), df['after'], test_size=0.1, random_state=2017)
x_test.drop(['class'], axis=1, inplace=True)

transform(x_train, x_test, y_train, y_test)