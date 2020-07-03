import jieba.posseg
import pandas as pd
import os
import jieba.analyse
import numpy as np
import math

# generate stopwords list
def generateStopwords():
    stopwords = [line.strip() for line in open('cn_stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords

# get the names of all files under the path
def getFileName(path):
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            files.append((os.path.join(dirpath, filename)))
    return files

# get the keywords of a file
def getEachKeywords(file, description):
    wordsPostProcess = []
    data = pd.read_excel(file)
    workOrder = list(data[description].values)
    length = len(workOrder)
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']
    punc = "，。、【 】:“”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥"
    # remove punctuations
    for i in workOrder:
        for j in i:
            if j in punc:
                i.replace(j, "")
    # remove stopwords
    stopwords = generateStopwords()
    for i in range(length):
        words = jieba.posseg.cut(workOrder[i])
        for eachWord in words:
            if eachWord.word not in stopwords and eachWord.flag in pos:
                wordsPostProcess.append(eachWord.word)
    return wordsPostProcess

# get the characteristic words (allow repeating) and unique words of all the files under the path
def getCharacteristicAndUniqueWords(path, description):
    characteristicAndUniqueWords = []
    files = getFileName(path)
    # get characteristic words of all the files
    characteristicWords = []
    for i in files:
        processed = getEachKeywords(i, description)
        characteristicWords.append(processed)
    characteristicAndUniqueWords.append(characteristicWords)
    # get unique words of all the files
    uniqueWords = []
    for i in characteristicWords:
        for j in i:
            if j not in uniqueWords:
                uniqueWords.append(j)
    characteristicAndUniqueWords.append(uniqueWords)
    return characteristicAndUniqueWords


# def getFrequencyMatrix(characteristicAndUniqueWords):
#     characteristic = characteristicAndUniqueWords[0]
#     uniqueWords = characteristicAndUniqueWords[1]
#     num = len(characteristicAndUniqueWords[0])
#     frequencyMatrix = np.zeros((num, len(uniqueWords)))
#     for i in range(num):
#         counts = {}
#         for word in characteristic[i]:
#             counts[word] = counts.get(word, 0) + 1
#         for j in range(len(uniqueWords)):
#             word = uniqueWords[j]
#             if word in counts:
#                 frequencyMatrix[i, j] = counts[word]
#             else:
#                 frequencyMatrix[i, j] = 0
#     return frequencyMatrix


# get the tf value of each unique word
def getTF(frequencyVector):
    # get the number of all words in this vector
    eachFileWordNum = np.sum(frequencyVector)
    # get the number of unique words in this vector
    wordNum = len(frequencyVector)
    tf = np.zeros((wordNum))
    # tf = number of each unique words / number of all words
    for i in range(wordNum):
        tf[i] = frequencyVector[i] / eachFileWordNum
    return tf


# get the idf value of each unique word in train files and test files
def getIDF(frequencyVectorTest, frequencyVectorTrain):
    fileNum = 2
    wordNum = len(frequencyVectorTest)
    count = {}
    # get the number of each unique word in the vector
    for i in range(wordNum):
        if frequencyVectorTest[i] > 0:
            count[i] = count.get(i, 0) + 1
        if frequencyVectorTrain[i] > 0:
            count[i] = count.get(i, 0) + 1
    wordCount = list(count.values())
    # idf = log(number of files / (number of files the unique word shows up + 1))
    idf = np.zeros((wordNum))
    for i in range(wordNum):
        idf[i] = math.log(fileNum / (wordCount[i]) + 1)
    return idf


# get the tf-idf value
def getTF_IDF(tf, idf):
    wordNum = len(tf)
    # tf-idf = tf * idf
    tf_idf = np.zeros((wordNum))
    for i in range(wordNum):
        tf_idf[i] = tf[i] * idf[i]
    return tf_idf

# get the cosine similarity of 2 vectors
def calculateSimilarity(trainCorpus, testCorpus):
    typeNum = len(trainCorpus)
    testOrderNum = len(testCorpus)
    similarity = np.zeros((testOrderNum, typeNum))
    for i in range(testOrderNum):
        workOrder = testCorpus[i]
        characteristicTest = workOrder[0]
        for j in range(typeNum):
            characteristicTrain = trainCorpus[j]
            uniqueWords = []
            countTrain = {}
            countTest = {}
            # get the characteristic words and unique words of train+test corpus
            for word in characteristicTest:
                countTest[word] = countTest.get(word, 0) + 1
                if word not in uniqueWords:
                    uniqueWords.append(word)
            for word in characteristicTrain:
                countTrain[word] = countTrain.get(word, 0) + 1
                if word not in uniqueWords:
                    uniqueWords.append(word)
            uniqueWordsNum = len(uniqueWords)
            frequencyVectorTest = np.zeros((uniqueWordsNum))
            frequencyVectorTrain = np.zeros((uniqueWordsNum))
            # get the frequency vector of train + test corpus
            for k in range(uniqueWordsNum):
                word = uniqueWords[k]
                if word in countTest:
                    frequencyVectorTest[k] = countTest[word]
                if word in countTrain:
                    frequencyVectorTrain[k] = countTrain[word]
            # get the tf-idf value of train + test corpus
            tfTest = getTF(frequencyVectorTest)
            tfTrain = getTF(frequencyVectorTrain)
            idf = getIDF(frequencyVectorTest, frequencyVectorTrain)
            tf_idfTest = getTF_IDF(tfTest, idf)
            tf_idfTrain = getTF_IDF(tfTrain, idf)
            # get the cosine similarity of these 2 vectors
            thisSimilarity = cosineSimilarity(tf_idfTest, tf_idfTrain)
            similarity[i][j] = thisSimilarity
    # return the similarity of work orders in 2 test files and the 11 types in train files
    return similarity


# get the cosine similarity of 2 vectors
def cosineSimilarity(vectorA, vectorB):
    vectorProduct = np.dot(vectorA, vectorB)
    normProduct = np.linalg.norm(vectorA) * np.linalg.norm(vectorB)
    cosine = vectorProduct / normProduct
    return cosine


# get the charactersitic words and unique words in each work order of 2 test files
def getTestCorpus(path, description):
    files = getFileName(path)
    feature = []
    for file in files:
        data = pd.read_excel(file)
        workOrder = list(data[description].values)
        length = len(workOrder)
        pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']
        punc = "，。、【 】:“”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥"
        for i in workOrder:
            for j in i:
                if j in punc:
                    i.replace(j, "")
        stopwords = generateStopwords()
        totalCharacteristicAndUniqueWords = []
        for i in range(length):
            thisCharacteristicAndUniqueWords = []
            thisCharacteristicWords = []
            thisUniqueWords = []
            words = jieba.posseg.cut(workOrder[i])
            for eachWord in words:
                if eachWord.word not in stopwords and eachWord.flag in pos:
                    thisCharacteristicWords.append(eachWord.word)
            thisCharacteristicAndUniqueWords.append(thisCharacteristicWords)
            for j in thisCharacteristicWords:
                if j not in thisUniqueWords:
                    thisUniqueWords.append(j)
            thisCharacteristicAndUniqueWords.append(thisUniqueWords)
            totalCharacteristicAndUniqueWords.append(thisCharacteristicAndUniqueWords)
        feature.append(totalCharacteristicAndUniqueWords)
    return feature


# get the index([0,10]) of the max values in similarity of 1 work order and 11 types in 1 test file
def getComparisonResult(cosineSimilar):
    workOrderNum = cosineSimilar.shape[0]
    maxIndex = np.zeros(workOrderNum, dtype=np.int)
    for i in range(workOrderNum):
        maxIndex[i] = np.argmax(cosineSimilar[i])
    return maxIndex


# write the type with highest similarity into the excel
# calculate the correctness
def writeIntoExcel(comparisonaResult, id):
    files = getFileName("data_train/")
    typeNum = len(files)
    typeName = [1] * typeNum
    workOrderNum = len(comparisonaResult)
    result = [1] * workOrderNum
    for i in range(typeNum):
        typeName[i] = files[i][11:-4]
    for i in range(workOrderNum):
        index = comparisonaResult[i]
        result[i] = typeName[index]
    resultList = result
    resultDataFrame = pd.DataFrame(result)
    testFiles = getFileName("data_test/")
    targetFile = testFiles[id]
    data = pd.read_excel(targetFile)
    data['预测结果'] = resultDataFrame
    name = '第' + str(id+1) + '轮预测结果.xls'
    data.to_excel(name, index=False)
    rawWorkOrder = list(data['场景'].values)
    print(resultList)
    print(rawWorkOrder)
    count = 0
    for i in range(workOrderNum):
        if rawWorkOrder[i] == resultList[i]:
            count += 1
    correct = count / workOrderNum
    return correct


trainCorpus = getCharacteristicAndUniqueWords("data_train/", "工单内容")
testCorpus = getTestCorpus("data_test/", "内容")
cosineSimilar_01 = calculateSimilarity(trainCorpus[0], testCorpus[0])
cosineSimilar_02 = calculateSimilarity(trainCorpus[0], testCorpus[1])
comparisonResult_01 = getComparisonResult(cosineSimilar_01)
comparisonResult_02 = getComparisonResult(cosineSimilar_02)

correct_01 = writeIntoExcel(comparisonResult_01, 0)
print(correct_01)
print("--------------------------------")
correct_02 = writeIntoExcel(comparisonResult_02, 1)
print(correct_02)