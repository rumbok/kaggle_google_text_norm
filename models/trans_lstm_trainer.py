from loaders.loading import load_train, load_external
from models.lstm.trainer import train


INPUT_MAX_LEN = 32
OUTPUT_MAX_LEN = 32
INPUT_VOCAB_SIZE = 27
OUTPUT_VOCAB_SIZE = 33

LAYER_NUM = 2
HIDDEN_DIM = 256
EMBEDDING_DIM = 0
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MEM_SIZE = 10000
NB_EPOCH = 100
DROPOUT = 0.01

df = load_train(['before', 'after'], input_path=r'../input/norm_challenge_ru').fillna('')
#df = load_external(['before', 'after'],
#                   only_diff=True,
#                   input_path=r'../input/norm_challenge_ru/ru_with_types')\
#      .fillna('')
df = df[~(df['before'] == df['after']) & (df['after'].str.contains('_trans'))]
df['after'] = df['after'].str.replace('_trans', '')
df['before'] = df['before'].str.lower().map(lambda s: ' '.join(list(s)))
print('drop {0} urls from strings'.format(len(df[df['before'].str.contains('\.')].index)))
df = df[~df['before'].str.contains('\.')]
#df = df.sample(3000000)
print(df.info())

result_df = train('trans', df,
                  INPUT_MAX_LEN, INPUT_VOCAB_SIZE, OUTPUT_MAX_LEN, OUTPUT_VOCAB_SIZE,
                  HIDDEN_DIM, LAYER_NUM, LEARNING_RATE, DROPOUT, EMBEDDING_DIM,
                  NB_EPOCH, MEM_SIZE, BATCH_SIZE)

print(result_df[~(result_df['actual'] == result_df['predict'])][['actual', 'predict']].sample(20))
