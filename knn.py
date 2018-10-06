import csv
import math
import random
import statistics as stat
import matplotlib.pyplot as plt

def printData(data):
    for row in data:
        print(row)

def loadData(filename, split, x_train=[], x_test=[], y_train=[]):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        #remove rows with missing values
        dataset = [row for row in dataset if '?' not in row]
        #convert all values from str to int
        dataset = [[int(val) for val in row] for row in dataset]
        #remove serial numbers
        dataset = [row[1:11] for row in dataset]
        #split the data 80/20 train/test
        trainSize = math.ceil(len(dataset) * split)
        testSize = len(dataset) - trainSize
        topEighty = dataset[0:int(trainSize)]
        bottomTwenty = dataset[int(trainSize):len(dataset)]
        #now we have the 80/20 split
        #now put the class ID's from top eighty percent into y_train
        y_train[:] = [row[9] for row in topEighty]
        #now remove class ID's from training data
        x_train[:] = [row[0:9] for row in topEighty]
        #now fill x_test with bottom 20% (including ID's)
        x_test[:] = bottomTwenty

#find distance between a and b given p
def distance(x, y, p):
    sum = 0
    for val1, val2 in zip(x, y):
        sum += pow(abs(val1 - val2), p)
    return sum**(1/float(p))

#find the k nearest neighbors for a test instance
def get_neighbors(x_train, test_val, y_train, k, p):
    distances = []
    dist = 0
    for i in range(len(x_train)):
        dist = distance(test_val, x_train[i], p)
        ident = y_train[i]
        distances.append((dist, ident))
    distances.sort()
    #return only the k-nearest
    return distances[0:k]

def knn_classifier(x_test, x_train, y_train, k, p):
    y_pred = []
    for test_instance in x_test:
        neighbors = get_neighbors(x_train, test_instance[0:9], y_train, k, p)
        #predict class
        mal_count = 0
        ben_count = 0
        for t in neighbors:
            if t[1] == 2:
                ben_count += 1
            elif t[1] == 4:
                mal_count += 1
        y_pred.append(2) if ben_count > mal_count else y_pred.append(4)
    return y_pred
    

def part1():
    filename = 'breast-cancer-wisconsin.data'
    distances = []
    x_train = []
    x_test = []
    y_train = []

    loadData(filename, 0.80, x_train, x_test, y_train)

    y_pred = knn_classifier(x_test, x_train, y_train, 2, 2)
    y_actual = [row[9] for row in x_test]

    matches = 0.0
    for pred, actual in zip(y_pred, y_actual):
        if pred == actual:
            matches += 1

    total_size = float(len(y_pred))

    print("Matches: " + str(int(matches)) + " (" + str((matches/total_size)*100) + "%)")

part1()

def cross_validation():
    filename = 'breast-cancer-wisconsin.data'
    distances = []
    x_train = []
    x_test = []
    y_train = []
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        #remove rows with missing values
        dataset = [row for row in dataset if '?' not in row]
        #convert all values from str to int
        dataset = [[int(val) for val in row] for row in dataset]
        #remove serial numbers
        dataset = [row[1:11] for row in dataset]
        #iterate through list and create 10 folds
        foldSize = int(len(dataset) / 10)
        #shuffle data randomly
        random.shuffle(dataset)
        #here we actually create the folds
        start = 0
        #each iteration of this loop is a new fold
        p1_acc = []
        p1_sen = []
        p1_spec = []
        p2_acc = []
        p2_sen = []
        p2_spec = []
        for i in range(10):
            currentFold = dataset[start:start+foldSize]
            currentTraining = dataset[0:start] + dataset[start+foldSize:len(dataset)]
            #now that current fold is set and the rest is used for training, we make the necessary calls
            print("Fold "+str(i+1))
            y_train[:] = [row[9] for row in currentTraining]
            x_train[:] = [row[0:9] for row in currentTraining]
            x_test[:] = currentFold
            #P=1 FOR K=1-10
            print("\tp = 1")
            #for each k value
            for i in range(10):
                y_pred = knn_classifier(x_test, x_train, y_train, i+1, 1)
                y_actual = [row[9] for row in x_test]
                #get the accuracy
                #we must also find the true negatives and true positives
                matches = 0.0
                truePos = 0.0 
                trueNeg = 0.0
                for pred, actual in zip(y_pred, y_actual):
                    if pred == actual:
                        matches += 1
                        if pred == 2:
                            trueNeg += 1 
                        else: 
                            truePos += 1
                #now get the number of pos and neg
                negs = 0.0
                pos = 0.0
                for val in y_actual:
                    if val == 2:
                        negs += 1 
                    else:
                        pos += 1
                total_size = float(len(y_pred))
                accuracy = (matches/total_size)*100
                sensitivty = (truePos / pos)*100
                specificity = (trueNeg / negs)*100
                p1_acc.append(accuracy)
                p1_sen.append(sensitivty)
                p1_spec.append(specificity)
                #print("\t\tk = " + str(i+1) + ": (A/Sen/Spec) " + str(round(accuracy, 1)) + "%/" + str(round(sensitivty, 1)) + "%/" + str(round(specificity, 1)) + "%")
            #P=2 FOR K=1-10
            print("\tp = 2")
            for i in range(10):
                y_pred = knn_classifier(x_test, x_train, y_train, i+1, 2)
                y_actual = [row[9] for row in x_test]
                #get the accuracy
                #we must also find the true negatives and true positives
                matches = 0.0
                truePos = 0.0 
                trueNeg = 0.0
                for pred, actual in zip(y_pred, y_actual):
                    if pred == actual:
                        matches += 1
                        if pred == 2:
                            trueNeg += 1 
                        else: 
                            truePos += 1
                #now get the number of pos and neg
                negs = 0.0
                pos = 0.0
                for val in y_actual:
                    if val == 2:
                        negs += 1 
                    else:
                        pos += 1
                total_size = float(len(y_pred))
                accuracy = (matches/total_size)*100
                sensitivty = (truePos / pos)*100
                specificity = (trueNeg / negs)*100
                p2_acc.append(accuracy)
                p2_sen.append(sensitivty)
                p2_spec.append(specificity)
                #print("\t\tk = " + str(i+1) + ": (A/Sen/Spec) " + str(round(accuracy, 1)) + "%/" + str(round(sensitivty, 1)) + "%/" + str(round(specificity, 1)) + "%")

            start += foldSize

        #STATS PORTION
        x = [1,2,3,4,5,6,7,8,9,10]
        #WE AVERAGE AND FIND STD DEV FOR ACCURACY, PRECISION, AND SPECIFICITY ACROSS ALL 10 FOLDS FOR K=1-10
        #P1 STATS
        p1_k1_acc = p1_acc[0::10]
        p1_k2_acc = p1_acc[1::10]
        p1_k3_acc = p1_acc[2::10]
        p1_k4_acc = p1_acc[3::10]
        p1_k5_acc = p1_acc[4::10]
        p1_k6_acc = p1_acc[5::10]
        p1_k7_acc = p1_acc[6::10]
        p1_k8_acc = p1_acc[7::10]
        p1_k9_acc = p1_acc[8::10]
        p1_k10_acc = p1_acc[9::10]

        p1_k1_sen = p1_sen[0::10]
        p1_k2_sen = p1_sen[1::10]
        p1_k3_sen = p1_sen[2::10]
        p1_k4_sen = p1_sen[3::10]
        p1_k5_sen = p1_sen[4::10]
        p1_k6_sen = p1_sen[5::10]
        p1_k7_sen = p1_sen[6::10]
        p1_k8_sen = p1_acc[7::10]
        p1_k9_sen = p1_sen[8::10]
        p1_k10_sen = p1_sen[9::10]

        p1_k1_spec = p1_spec[0::10]
        p1_k2_spec = p1_spec[1::10]
        p1_k3_spec = p1_spec[2::10]
        p1_k4_spec = p1_spec[3::10]
        p1_k5_spec = p1_spec[4::10]
        p1_k6_spec = p1_spec[5::10]
        p1_k7_spec = p1_spec[6::10]
        p1_k8_spec = p1_spec[7::10]
        p1_k9_spec = p1_spec[8::10]
        p1_k10_spec = p1_spec[9::10]

        #PLOTS
        #accuracy
        y = [stat.mean(p1_k1_acc), stat.mean(p1_k2_acc), stat.mean(p1_k3_acc), stat.mean(p1_k4_acc), stat.mean(p1_k5_acc), stat.mean(p1_k6_acc), stat.mean(p1_k7_acc), stat.mean(p1_k8_acc), stat.mean(p1_k9_acc), stat.mean(p1_k10_acc)]
        yerr = [stat.stdev(p1_k1_acc), stat.stdev(p1_k2_acc), stat.stdev(p1_k3_acc), stat.stdev(p1_k4_acc), stat.stdev(p1_k5_acc), stat.stdev(p1_k6_acc), stat.stdev(p1_k7_acc), stat.stdev(p1_k8_acc), stat.stdev(p1_k9_acc), stat.stdev(p1_k10_acc)]
        fig = plt.figure()
        fig.suptitle('p = 1')
        plt.xlabel('k neighbors')
        plt.ylabel('Accuracy %')
        axes = plt.gca()
        axes.set_ylim([80,100])
        plt.xticks(x)
        plt.errorbar(x, y, yerr)
        plt.show()
        fig.savefig('fig1.png')
        #sensitivity
        y = [stat.mean(p1_k1_sen), stat.mean(p1_k2_sen), stat.mean(p1_k3_sen), stat.mean(p1_k4_sen), stat.mean(p1_k5_sen), stat.mean(p1_k6_sen), stat.mean(p1_k7_sen), stat.mean(p1_k8_sen), stat.mean(p1_k9_sen), stat.mean(p1_k10_sen)]
        yerr = [stat.stdev(p1_k1_sen), stat.stdev(p1_k2_sen), stat.stdev(p1_k3_sen), stat.stdev(p1_k4_sen), stat.stdev(p1_k5_sen), stat.stdev(p1_k6_sen), stat.stdev(p1_k7_sen), stat.stdev(p1_k8_sen), stat.stdev(p1_k9_sen), stat.stdev(p1_k10_sen)]
        fig = plt.figure()
        fig.suptitle('p = 1')
        plt.xlabel('k neighbors')
        plt.ylabel('Sensitivity %')
        axes = plt.gca()
        axes.set_ylim([80,100])
        plt.xticks(x)
        plt.errorbar(x, y, yerr)
        plt.show()  
        fig.savefig('fig2.png')
        #specificity
        y = [stat.mean(p1_k1_spec), stat.mean(p1_k2_spec), stat.mean(p1_k3_spec), stat.mean(p1_k4_spec), stat.mean(p1_k5_spec), stat.mean(p1_k6_spec), stat.mean(p1_k7_spec), stat.mean(p1_k8_spec), stat.mean(p1_k9_spec), stat.mean(p1_k10_spec)]
        yerr = [stat.stdev(p1_k1_spec), stat.stdev(p1_k2_spec), stat.stdev(p1_k3_spec), stat.stdev(p1_k4_spec), stat.stdev(p1_k5_spec), stat.stdev(p1_k6_spec), stat.stdev(p1_k7_spec), stat.stdev(p1_k8_spec), stat.stdev(p1_k9_spec), stat.stdev(p1_k10_spec)] 
        fig = plt.figure()
        fig.suptitle('p = 1')
        plt.xlabel('k neighbors')
        plt.ylabel('Specificity %')
        axes = plt.gca()
        axes.set_ylim([80,100])
        plt.xticks(x)
        plt.errorbar(x, y, yerr)
        plt.show() 
        fig.savefig('fig3.png')

        #P2 STATS 
        p2_k1_acc = p1_acc[0::10]
        p2_k2_acc = p1_acc[1::10]
        p2_k3_acc = p1_acc[2::10]
        p2_k4_acc = p1_acc[3::10]
        p2_k5_acc = p1_acc[4::10]
        p2_k6_acc = p1_acc[5::10]
        p2_k7_acc = p1_acc[6::10]
        p2_k8_acc = p1_acc[7::10]
        p2_k9_acc = p1_acc[8::10]
        p2_k10_acc = p1_acc[9::10]

        p2_k1_sen = p1_sen[0::10]
        p2_k2_sen = p1_sen[1::10]
        p2_k3_sen = p1_sen[2::10]
        p2_k4_sen = p1_sen[3::10]
        p2_k5_sen = p1_sen[4::10]
        p2_k6_sen = p1_sen[5::10]
        p2_k7_sen = p1_sen[6::10]
        p2_k8_sen = p1_acc[7::10]
        p2_k9_sen = p1_sen[8::10]
        p2_k10_sen = p1_sen[9::10]

        p2_k1_spec = p1_spec[0::10]
        p2_k2_spec = p1_spec[1::10]
        p2_k3_spec = p1_spec[2::10]
        p2_k4_spec = p1_spec[3::10]
        p2_k5_spec = p1_spec[4::10]
        p2_k6_spec = p1_spec[5::10]
        p2_k7_spec = p1_spec[6::10]
        p2_k8_spec = p1_spec[7::10]
        p2_k9_spec = p1_spec[8::10]
        p2_k10_spec = p1_spec[9::10]

        #PLOTS
        #accuracy
        y = [stat.mean(p2_k1_acc), stat.mean(p2_k2_acc), stat.mean(p2_k3_acc), stat.mean(p2_k4_acc), stat.mean(p2_k5_acc), stat.mean(p2_k6_acc), stat.mean(p2_k7_acc), stat.mean(p2_k8_acc), stat.mean(p2_k9_acc), stat.mean(p2_k10_acc)]
        yerr = [stat.stdev(p2_k1_acc), stat.stdev(p2_k2_acc), stat.stdev(p2_k3_acc), stat.stdev(p2_k4_acc), stat.stdev(p2_k5_acc), stat.stdev(p2_k6_acc), stat.stdev(p2_k7_acc), stat.stdev(p2_k8_acc), stat.stdev(p2_k9_acc), stat.stdev(p2_k10_acc)]
        fig = plt.figure()
        fig.suptitle('p = 2')
        plt.xlabel('k neighbors')
        plt.ylabel('Accuracy %')
        axes = plt.gca()
        axes.set_ylim([80,100])
        plt.xticks(x)
        plt.errorbar(x, y, yerr)
        plt.show()
        fig.savefig('fig4.png')
        #sensitivity
        y = [stat.mean(p2_k1_sen), stat.mean(p2_k2_sen), stat.mean(p2_k3_sen), stat.mean(p2_k4_sen), stat.mean(p2_k5_sen), stat.mean(p2_k6_sen), stat.mean(p2_k7_sen), stat.mean(p2_k8_sen), stat.mean(p2_k9_sen), stat.mean(p2_k10_sen)]
        yerr = [stat.stdev(p2_k1_sen), stat.stdev(p2_k2_sen), stat.stdev(p2_k3_sen), stat.stdev(p2_k4_sen), stat.stdev(p2_k5_sen), stat.stdev(p2_k6_sen), stat.stdev(p2_k7_sen), stat.stdev(p2_k8_sen), stat.stdev(p2_k9_sen), stat.stdev(p2_k10_sen)]
        fig = plt.figure()
        fig.suptitle('p = 2')
        plt.xlabel('k neighbors')
        plt.ylabel('Sensitivity %')
        axes = plt.gca()
        axes.set_ylim([80,100])
        plt.xticks(x)
        plt.errorbar(x, y, yerr)
        plt.show()  
        fig.savefig('fig5.png')
        #specificity
        y = [stat.mean(p2_k1_spec), stat.mean(p2_k2_spec), stat.mean(p2_k3_spec), stat.mean(p2_k4_spec), stat.mean(p2_k5_spec), stat.mean(p2_k6_spec), stat.mean(p2_k7_spec), stat.mean(p2_k8_spec), stat.mean(p2_k9_spec), stat.mean(p2_k10_spec)]
        yerr = [stat.stdev(p2_k1_spec), stat.stdev(p2_k2_spec), stat.stdev(p2_k3_spec), stat.stdev(p2_k4_spec), stat.stdev(p2_k5_spec), stat.stdev(p2_k6_spec), stat.stdev(p2_k7_spec), stat.stdev(p2_k8_spec), stat.stdev(p2_k9_spec), stat.stdev(p2_k10_spec)]
        fig = plt.figure()
        fig.suptitle('p = 2')
        plt.xlabel('k neighbors')
        plt.ylabel('Specificity %')
        axes = plt.gca()
        axes.set_ylim([80,100])
        plt.xticks(x)
        plt.errorbar(x, y, yerr)
        plt.show()
        fig.savefig('fig6.png')


cross_validation()










