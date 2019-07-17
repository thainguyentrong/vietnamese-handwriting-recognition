import os 
import cv2
from sklearn.utils import shuffle
from helper import preprocess

class Sample: 
    ''' sample from dataset '''
    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath

class MiniBatch:
    ''' minibatch containing images and ground truth texts '''
    def __init__(self, gtTexts, imgs):
        self.imgs = imgs
        self.gtTexts = gtTexts

class DataLoader:
    def __init__(self, dataPath, batchSize, imgSize):
        ''' loader for dataset at given location '''
        self.curr = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []

        chars = set()

        for folderName in os.listdir(dataPath):
            baseLabels = os.path.join(dataPath, folderName + '/labels/')
            baseImages = os.path.join(dataPath, folderName + '/images/')

            for fileName in os.listdir(baseLabels):
                filePath = os.path.join(baseLabels, fileName)
                # get ground truth text
                with open(filePath, 'r', encoding='utf-8') as f:
                    gtText = f.read()
                # get image file path
                image = os.path.join(baseImages, fileName.split('.')[0] + '.png')
                # get all characters in dataset
                chars = chars.union(set(list(gtText)))
                # put sample into list
                self.samples.append(Sample(gtText=gtText, filePath=image))
        
        # split into training and test set: 95% - 5%
        split = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:split]
        self.testSamples = self.samples[split:]
        # sort and put into list
        self.charList = sorted(list(chars))
    
    def trainSet(self):
        ''' initialize and shuffle training set '''
        self.curr = 0
        self.samples = shuffle(self.trainSamples, random_state=123)
    
    def testSet(self):
        ''' initialize test set '''
        self.curr = 0
        self.samples = self.testSamples
    
    def getIterator(self):
        ''' return current minibatch index and number of minibatches '''
        return self.curr//self.batchSize + 1, len(self.samples)//self.batchSize
    
    def next(self):
        ''' use in iterator '''
        return self.curr + self.batchSize <= len(self.samples)
    
    def getMiniBatch(self):
        ''' get minibatch '''
        batchRange = range(self.curr, self.curr+self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        imgs = [preprocess(img=cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), imgSize=self.imgSize) for i in batchRange]
        self.curr += self.batchSize
        return MiniBatch(gtTexts, imgs)