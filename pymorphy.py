import pymorphy2
import re

morph = pymorphy2.MorphAnalyzer()


def flection(lex_neighb, tags):
    tags = str(tags)
    tags = re.sub(',[AGQSPMa-z-]+? ', ',', tags)
    tags = tags.replace("impf,", "")
    tags = re.sub('([A-Z]) (plur|masc|femn|neut|inan)', '\\1,\\2', tags)
    tags = tags.replace("Impe neut", "")
    tags = tags.split(',')
    tags_clean = []
    for t in tags:
        if t:
            if ' ' in t:
                t1, t2 = t.split(' ')
                t = t2
            tags_clean.append(t)
    tags = frozenset(tags_clean)
    prep_for_gen = morph.parse(lex_neighb)



