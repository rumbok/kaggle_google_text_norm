from loaders.loading import load_train, load_external
from models.lstm.trainer import train


CARDINAL_REGEXP = '\d'
INPUT_MAX_LEN = 33
OUTPUT_MAX_LEN = 23
INPUT_VOCAB_SIZE = 5000
OUTPUT_VOCAB_SIZE = 158

LAYER_NUM = 2
HIDDEN_DIM = 64
EMBEDDING_DIM = 0
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MEM_SIZE = 1000
NB_EPOCH = 1
DROPOUT = 0.0

df = load_train(['before', 'after', 'class'], input_path=r'../input/norm_challenge_ru').fillna('')
# df = load_external(['before', 'after'],
#                   only_diff=True,
#                   input_path=r'../input/norm_challenge_ru/ru_with_types')\
#      .fillna('')
df['prev_prev'] = df['before'].shift(2)
df['prev'] = df['before'].shift(1)
df['next'] = df['before'].shift(-1)
df['next_next'] = df['before'].shift(-2)
df = df[~(df['before'] == df['after'])].fillna('')
df = df[df['class'] == 'CARDINAL']
df['before'] = df['prev_prev'].map(str) + ' '\
               + df['prev'].map(str) + ' '\
               + df['before'].map(lambda s: ' '.join(list(s))) + ' ' \
               + df['next'].map(str) + ' ' \
               + df['next_next'].map(str)
del df['prev_prev'], df['prev'], df['next'], df['next_next'],
df['before'] = df['before'].str.lower()
# df = df.sample(10000)
print(df.info())


result_df = train('cardinal', df,
                  INPUT_MAX_LEN, INPUT_VOCAB_SIZE, OUTPUT_MAX_LEN, OUTPUT_VOCAB_SIZE,
                  HIDDEN_DIM, LAYER_NUM, LEARNING_RATE, DROPOUT, EMBEDDING_DIM,
                  NB_EPOCH, MEM_SIZE, BATCH_SIZE)

print(result_df[~(result_df['actual'] == result_df['predict'])][['actual', 'predict']].sample(20))
