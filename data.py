import os
import time
from PIL import Image
from torchvision import transforms
import torch
import time
from torch.utils.data import Dataset, DataLoader

dataPath = "./data/archive"
trainPath = dataPath + "/train"
testPath = dataPath + "/test"

trainDir = os.listdir(trainPath)
testDir = os.listdir(testPath)

transforms = transforms.ToTensor()


class Data:
    def __init__(self, trainDir, testDir, device):
        self.device = device
        self.trainDir = trainDir
        self.testDir = testDir
        self.trainData = []
        self.trainData = self.trainData
        self.testData = []
        self.testData = self.testData
        self.label2id = {'lqs': 0, 'htj': 1, 'wzm': 2, 'fwq': 3, 'yzq': 4, 'hy': 5, 'mf': 6, 'sgt': 7, 'smh': 8, 'oyx': 9, 'lx': 10, 'wxz': 11, 'zmf': 12, 'yyr': 13, 'mzd': 14, 'bdsr': 15, 'csl': 16, 'gj': 17, 'shz': 18, 'lgq': 19}
        self.id2label = {0: 'lqs', 1: 'htj', 2: 'wzm', 3: 'fwq', 4: 'yzq', 5: 'hy', 6: 'mf', 7: 'sgt', 8: 'smh', 9: 'oyx', 10: 'lx', 11: 'wxz', 12: 'zmf', 13: 'yyr', 14: 'mzd', 15: 'bdsr', 16: 'csl', 17: 'gj', 18: 'shz', 19: 'lgq'}
        self.startTime = time.time()
    def loadData(self):
        for i, path in enumerate(self.trainDir):
            if path == '.DS_Store': continue
            self.loadImage(trainPath + "/" + path, path, self.trainData, channel=3)
        for i, path in enumerate(self.testDir):
            if path == '.DS_Store': continue
            self.loadImage(testPath + "/" + path, path, self.testData, channel=3)
    def loadImage(self,path:str, name:str,data, channel=3):
        limit = 0
        for img in os.listdir(path):
            tmp = dict()
            with Image.open(path + "/" + img) as im:
                if channel == 3:
                    tmp['pixel_values'] = transforms(im).to(self.device)
                else:
                    tmp['pixel_values'] = transforms(im.convert('L')).to(self.device)
            tmp['labels'] = torch.tensor(self.label2id[name], dtype=torch.long).to(self.device)
            data.append(tmp)

            limit += 1
            if limit >= 1500 and "train" in path:
                break
            elif limit >= 200 and "test" in path:
                break

        return data
    def checkData(self):
        print(f'train data: {len(self.trainData)}, test data: {len(self.testData)}')
        tmpTrain = dict()
        tmpTest = dict()
        for case in self.trainData:
            label = case['labels'].item()
            label = self.id2label[label]
            tmpTrain[label] = tmpTrain.get(label, 0) + 1
        for case in self.testData:
            label = case['labels'].item()
            label = self.id2label[label]
            tmpTest[label] = tmpTest.get(label, 0) + 1
        print(f'train data distribution: {tmpTrain}, test data distribution: {tmpTest}')
    
    def getData(self, type:str, batch_size):
        if type == 'train': return DataLoader(self.trainData, batch_size=batch_size, shuffle=True)
        if type == 'test': return DataLoader(self.testData, batch_size=batch_size, shuffle=False)

        print(f"Processing error: getData_type:{type}")
        return None
    
    def getCheckData(self, batch_size,channel=3):
        data = []
        for i, path in enumerate(self.testDir):
            if path == '.DS_Store': continue
            for img in os.listdir(trainPath + "/" + path):
                tmp = dict()
                with Image.open(trainPath + "/" + path + "/" + img) as im:
                    if channel == 3:
                        tmp['pixel_values'] = transforms(im).to(self.device)
                    else:
                        tmp['pixel_values'] = transforms(im.convert('L')).to(self.device)
                tmp['labels'] = torch.tensor(self.label2id[path], dtype=torch.long).to(self.device)
                data.append(tmp)
        return DataLoader(data, batch_size=batch_size, shuffle=True)

    def cost(self):
        print(f'load data total cost: {time.time() - self.startTime}')
        return time.time() - self.startTime

def InitData(device='cpu'):
    return Data(trainDir, testDir, device)
def LoadData(device='cpu'):
    data = Data(trainDir, testDir, device)
    data.loadData()
    data.checkData()
    data.cost()
    return data

if __name__ == '__main__':

    print('debug', trainDir)
    data = Data(trainDir, testDir)
    data.loadData()
    data.checkData()
    data.cost()
    test = data.getData('test', batch_size=3)
    for i, items in enumerate(test):
        print(items['pixel_values'].shape)
        print(items['labels'])
        break