from mecab import MeCab
# try:
mecab=MeCab(dictionary_path="C:/Users/user/Desktop/RNE_21/data_aug/mecab/share/mecab-ko-dic")
# except ValueError as e:
#     print(e)

text='공용 컴퓨터'
print(mecab.morphs(text))