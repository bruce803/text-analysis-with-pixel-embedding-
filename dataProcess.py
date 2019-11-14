import csv
import nltk
import numpy as np
from scipy import sparse
from PIL import Image, ImageFont, ImageDraw
try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_from_csv(filePath):
    '''
    input the csv file path, outputs feature, label
    '''
    samples = []
    with open(filePath) as file:
        reader = csv.reader(file, quotechar='"')
        for idx, line in enumerate(reader):
            text = ""
            for tx in line[1:]:
#                 print(tx)
                text += tx
                text += " "
            label = int(line[0]) - 1
            samples.append((label, text))
        return samples


def eng2img(sentence, max_word_len=16, fontsize=20):
    words_from_sen = wordTokener(sentence)

    height = int(max_word_len * 8.2)  # 8.2 is an empirical para

    img_3d = np.zeros((fontsize, height, len(words_from_sen)))
    for i in range(len(words_from_sen)):
        if len(words_from_sen[i]) <= max_word_len:  # deleting words whose len is to large

            font = ImageFont.truetype('times.ttf', fontsize)
            mask = font.getmask(words_from_sen[i], mode='L')
            mask_w, mask_h = mask.size
            img = Image.new('L', (height, fontsize), color=0)
            img_w, img_h = img.size
            d = Image.core.draw(img.im, 0)
            d.draw_bitmap(((img_w - mask_w) / 2, (img_h - mask_h) / 2), mask, 255)

            img_map = np.array(img.getdata()).reshape(img.size[1], img.size[0])
            img_3d[..., i] = img_map
    # print(img_map.shape)
    #         plt.imshow(img_map, cmap='hot')
    return sparse.COO(img_3d)


def wordTokener(sent):  # 将单句字符串分割成词
    wordsInStr = nltk.word_tokenize(sent)
    return wordsInStr


def write_pkl(X_list, file_dir='./train1/'):
    for i in range(len(X_list)):
        arr3d = eng2img(X_list[i][1])
        write_path = file_dir + str(i) + '.pkl'
        with open(write_path, 'wb') as curFile:
            pickle.dump([X_list[i][0], arr3d], curFile)



ag_test = './test-repaired.csv'
ag_train = './train-repaired.csv'

train_samples = read_from_csv(ag_train)
test_samples = read_from_csv(ag_test)

write_pkl(test_samples, file_dir='./test1/')
write_pkl(train_samples)