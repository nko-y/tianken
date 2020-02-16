import keras
import os
import numpy as np
import re
import operator
from functools import reduce

texts = []
data_dir = '/Users/lufeiyang/Desktop/pycharm/fill_pit/passage.txt'
f = open(data_dir)
texts.append(f.read())
texts = reduce(operator.add, texts)
f.close()

f = open("twelveth.txt", "w")                                                          #需要修改第一处
input_data = "灰原哀此时此刻面临一个选择：向前还是向后。\n这并不是一个很好做的选择题。\n如果此时向前去上病原生物学课，那么她已经迟到一个半小时了，似乎没有这个必要；而如果向后回寝室顶着黑眼圈继续睡觉，那么她就等于白白来回爬了五层楼，这听上去就很让人沮丧。\n"
f.write(input_data)
print("input data:", input_data)
print("-------------------------")
print("len:", len(input_data))

lines_of_text = texts.split('\n')   #分割成为行

lines_of_text = [lines.strip() for lines in lines_of_text]
print(len(lines_of_text))
print(lines_of_text)    #删除头尾空格

lines_of_text = [lines for lines in lines_of_text if len(lines)>0 ] #去掉空行
print(len(lines_of_text))
print(lines_of_text)

pattern = re.compile(r'\[.*\]')
lines_of_text = [pattern.sub("", lines) for lines in lines_of_text] #去掉[]
pattern = re.compile(r'<.*>')
lines_of_text = [pattern.sub("", lines) for lines in lines_of_text] #去掉<>
pattern = re.compile(r'\.+')
lines_of_text = [pattern.sub("。", lines) for lines in lines_of_text]    #省略号替换为。
print(len(lines_of_text))
print(lines_of_text)
pattern = re.compile(r'\\r')
lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]
print(len(lines_of_text))
print(lines_of_text)    #去掉\r的内容

print('————————————————————————————————————————')
str_out = "\n".join(lines_of_text)
print(str_out)
print('————————————————————————————————————————')

maxlen = 121
step = 3
sentences = []
next_chars = []
for i in range(0, len(str_out)-maxlen, step):
    sentences.append(str_out[i:i+maxlen])
    next_chars.append(str_out[i+maxlen])
print('Number of sequence:', len(sentences))

chars = sorted(list(set(str_out+input_data)))
print('Unique characters:', len(chars))
char_indices = dict((char, chars.index(char)) for char in chars)

print('Vectorizeation...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

from keras import layers

model = keras.models.load_model('eleventh.h5')                                 #需要修改第二处

# model = keras.models.Sequential()
# model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
# model.add(layers.Dense(len(chars), activation='softmax'))
# optimizer = keras.optimizers.RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

import random
import sys

for epoch in range(1, 6):
    f.write("\n\n\n-------------------------------------------------")
    output_epoch = 'epoch' + str(epoch) + '\n'
    f.write(output_epoch)
    model.fit(x, y, batch_size=128, epochs=1)
    generated_text = input_data
    output_data = '--- the Input data:' + generated_text + '"'
    f.write(output_data)

    for temperature in [0.2, 0.5, 1.0 ,1.2]:
        output_temperature = '\n\n\n' + '-----temperature:' + str(temperature) + '\n'
        f.write(output_temperature)
        sys.stdout.write(generated_text)
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]]= 1.
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            f.write(next_char)
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)

f.close()
model.save('twelveth.h5')          #需要修改第三处
