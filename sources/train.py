import sys
import numpy as np
import matplotlib.pyplot as plt
import editdistance
from dataloader import DataLoader
from model import Model


def train(model, loader):
    ''' training neural network '''
    epoch = 0 # epochs since start
    bestCharErrorRate = float('inf') # initialize character error rate
    noImprovement = 0 # number of epochs no improvement of character error rate
    earlyStopping = 5 # stop training after this number of epochs without improvement
    avgLoss, listCharErrorRate, listAcc = ([] for _ in range(3))

    while True:
        epoch += 1
        lossPerMiniBatch = []
        loader.trainSet()
        while loader.next():
            info = loader.getIterator()
            minibatch = loader.getMiniBatch()
            loss = model.trainBatch(minibatch)
            sys.stdout.write('\rEpoch: %02d | Minibatch: %04d/%d | Loss: %7.3f' % (epoch, info[0], info[1], loss))
            lossPerMiniBatch.append(loss)
        # test with test set
        charErrorRate, avgAcc = test(model, loader)
        
        avgLoss.append(np.mean(lossPerMiniBatch))
        listCharErrorRate.append(charErrorRate)
        listAcc.append(avgAcc)

        if charErrorRate < bestCharErrorRate:
            # save model if character error rate is improved
            bestCharErrorRate = charErrorRate
            noImprovement = 0
            model.save()
        else:
            # Character error rate not improved
            noImprovement+= 1
        # stop training if model no more improvement
        if noImprovement >= earlyStopping:
            print('No more improvement. Training stopped.')
            break
    # plot during training
    plt.plot(np.arange(1, epoch + 1, 1), avgLoss, label='Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.show()

    plt.plot(np.arange(1, epoch + 1, 1), listCharErrorRate, label='Character Error Rate')
    plt.plot(np.arange(1, epoch + 1, 1), listAcc, linestyle='--', label='Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.show()

def test(model, loader):
    ''' test neural network model '''
    loader.testSet()
    avgAccuracy = []
    numCharErr = 0
    numCharTotal = 0

    while loader.next():
        info = loader.getIterator()
        minibatch = loader.getMiniBatch()
        texts = model.inferBatch(minibatch)
        # calculate words accuracy
        texts = np.array(texts, dtype=object)
        minibatch.gtTexts = np.array(minibatch.gtTexts, dtype=object)
        avgAccuracy.append(np.array(texts == minibatch.gtTexts))

        for i, text in enumerate(texts):
            dist = editdistance.eval(text, minibatch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(minibatch.gtTexts[i])
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + minibatch.gtTexts[i] + '"', '->', '"' + text + '"')
        print('Test | Minibatch : %d/%d' % (info[0], info[1]))
    # calulate char error rate
    charErrorRate = numCharErr/numCharTotal
    print('Character error rate: %7.3f | Word accuracy: %7.3f' % (charErrorRate*100.0, np.mean(avgAccuracy)*100.0))
    return charErrorRate, np.mean(avgAccuracy)

def main():
    loader = DataLoader('../data/InkData_word_processed/', Model.batchSize, Model.imgSize)
    # save charList in data folder
    strChars = str()
    with open('../data/charList.txt', 'w', encoding='utf-8') as f:
        for c in loader.charList:
            strChars += c
        f.write(strChars)
    
    model = Model(loader.charList)
    train(model, loader)

if __name__ == '__main__':
    main()