import cv2
from helper import preprocess
from model import Model
from dataloader import MiniBatch
from segment import prepareImg, wordSegmentation

def predict(model, word):
    ''' predict and return text recognized '''
    img = preprocess(word, Model.imgSize)
    minibatch = MiniBatch(None, [img])
    recognized = model.inferBatch(minibatch)
    return recognized[0]

def main():
    # restore model
    with open('../data/charList.txt', 'r', encoding='utf-8') as f:
        charList = f.read()
    charList = list(charList)
    model = Model(charList, restore=True)
    # text segment
    img = prepareImg(cv2.imread('../data/sample/sample_3.png'), 500)
    result = wordSegmentation(img, kernelSize=25, sigma=11, theta=4, minArea=500)

    recognized = str()
    draw = []
    for line in result:
        if len(line):
            for (_, w) in enumerate(line):
                (wordBox, wordImg) = w
                recognized += predict(model, wordImg) + ' '
                draw.append(wordBox)
            recognized += '\n'
    print('\n---------------------\nRecognized:\n' + recognized)

    for wordBox in draw:
        (x, y, w, h) = wordBox
        cv2.rectangle(img, (x, y), (x+w, y+h), 0, 1)
    cv2.imshow('result', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()