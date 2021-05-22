# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 04:39:55 2021

@author: erent
"""

import cv2
import time
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5.uic import *
from PyQt5.Qt import QApplication, QUrl, QDesktopServices
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
import xlwt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import sys
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import random
import seaborn as sns
from pandas import DataFrame
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier 
from skimage.feature import daisy
import os,shutil
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler
import pathlib
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import pickle
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.datasets import mnist
import joblib




class window(QMainWindow):
    
    def __init__(self):
        super(window, self).__init__()
        loadUi("goruntu_arayuz.ui", self)
        self.show()
        self.dosyabtn.clicked.connect(self.OpenFileDialog)
        self.getirbtn.clicked.connect(self.getircek)
        self.uygulabtn.clicked.connect(self.holdorkfold)
        self.egitbtn.clicked.connect(self.egit)
        self.excelbtn.clicked.connect(self.veriseticek)
        self.kfoldk.currentTextChanged.connect(self.kfoldcb)
        self.cnnbtn.clicked.connect(self.cnn)
        self.dosyabtn.setVisible(False)
        self.daisyn=30
        self.cropsayisi=10
        self.deger="0"
        self.descs=[]
        self.ylist=[]
        self.Xlist=[]
        self.Xcrop=[]
        self.imagesay=0
        self.fileNameListKeypoint=[]
        self.groupBox_2.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.groupBox.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.groupBox_4.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.groupBox_6.setStyleSheet("QGroupBox { border: 1px solid red;}")
        
        
           
    
    def veriseticek(self):
       file_name,_= QFileDialog.getOpenFileName(self, 'Open Image File', r".\Desktop")
       self.data = pd.read_csv(file_name, index_col=[0])
       c=len(self.data.columns)
       r=len(self.data.values)
       self.veritb.setColumnCount(c)
       self.veritb.setRowCount(r)
       for i,row in enumerate(self.data):
             for j,cell in enumerate(self.data.values):
                  self.veritb.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
       self.Xdata = self.data.drop(['pollen_carrying'], axis=1)
       self.ydata = self.data['pollen_carrying'] 
       self.Xdata=np.array(self.Xdata)
       self.Xdata = DataFrame(self.Xdata)
       self.Xdata=np.array(self.Xdata)
       self.dosyabtn.setVisible(True)
       self.excelbtn.setVisible(False)
       print("Tamamlandı.")
       self.groupBox_6.setStyleSheet("QGroupBox { border: 1px solid green;}")
             

    def OpenFileDialog(self):
        self.f0_name = QFileDialog.getExistingDirectory(self, 'Image Files', "select Directory")
        self.files=os.listdir(str(self.f0_name))
        self.groupBox_2.setStyleSheet("QGroupBox { border: 1px solid yellow;}")
        

    

    def getircek(self):
        print("START")
        self.maxData=int(self.kptb_2.text())
        self.cropsayisi=int(self.kptb.text())
        self.daisyn=int(self.ntb.text())
        self.siftorsuft()
        if(self.radioButton_he.isChecked()):
            for i in range(0, self.maxData+1):
                self.imagesay=i+1
                file_name=self.Xdata[i,0]
                self.deger=self.ydata[i]
                self.img=cv2.imread(self.f0_name + "/"+file_name)
                print(self.deger, " - ", str(file_name))
                self.HE()        
        else:
            for i in range(0, self.maxData+1):
                self.imagesay=i+1
                file_name=self.Xdata[i,0]
                self.deger=self.ydata[i]
                self.img=cv2.imread(str(self.f0_name)+"/"+file_name)
                print(self.deger, " - ", str(file_name))
                self.clahe()
        print("Tamamlandı.")
        self.groupBox_2.setStyleSheet("QGroupBox { border: 1px solid green;}")
        self.groupBox.setStyleSheet("QGroupBox { border: 1px solid yellow;}")
                
         

    def siftorsuft(self):
        self.kpA=0
        if(self.seccb.currentText()=="Sift"):
            self.kpA=0
        elif(self.seccb.currentText()=="Surf"):
            self.kpA=1
            
            
                

    def holdorkfold(self):
        algorithm = self.seccb2.currentText()
        if(algorithm=="Hold Out"):
            self.holdout()
        else:
            self.kfold()  
        print("Tamamlandı.")
        self.groupBox.setStyleSheet("QGroupBox { border: 1px solid green;}")
        self.groupBox_4.setStyleSheet("QGroupBox { border: 1px solid yellow;}")
    
    
    
    def kfold(self):
        self.k_foldBool=True
        self.holdOutBool=False
        tut=0
        self.kf=0
        if(self.radioButton_5.isChecked()):
            kf=5
        else:
            kf=10
        self.Xlist=self.Xlist.reshape(self.Xlist.shape[0],self.Xlist.shape[1]*self.Xlist.shape[2])
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.Xlist = scaler.fit_transform(self.Xlist)
        self.Kf=[None, None, None, None, None, None, None, None, None, None]
        cv = KFold(n_splits=kf, random_state=1, shuffle=False)
        self.ylist=np.array(self.ylist)
        for train_index, test_index in cv.split(self.Xlist):
            bag=[]
            y_train, y_test=self.ylist[train_index], self.ylist[test_index]
            X_train, X_test = self.Xlist[train_index], self.Xlist[test_index]
            bag.append(X_train)
            bag.append(X_test)
            bag.append(y_train)
            bag.append(y_test)
            self.Kf[tut]=bag
            self.kfoldi(tut)
            tut+=1
            
            
        
    def kfoldi(self,tut):
        if(self.k_foldBool):
            self.KfoldCurrentX_train=self.Kf[tut][0]
            self.KfoldCurrentX_test=self.Kf[tut][1]
            self.KfoldCurrenty_train=self.Kf[tut][2]
            self.KfoldCurrenty_test=self.Kf[tut][3]
            self.xtraincekKfold(self.KfoldCurrentX_train)
            self.Ytraincek(self.KfoldCurrenty_train)
            self.xtestcekKfold(self.KfoldCurrentX_test)
            self.Ytestcek(self.KfoldCurrenty_test)
            
    def kfoldcb(self):
        if(self.k_foldBool):
            self.KfoldCurrentX_train=self.Kf[int(self.kfoldk.currentText())-1][0]
            self.KfoldCurrentX_test=self.Kf[int(self.kfoldk.currentText())-1][1]
            self.KfoldCurrenty_train=self.Kf[int(self.kfoldk.currentText())-1][2]
            self.KfoldCurrenty_test=self.Kf[int(self.kfoldk.currentText())-1][3]
            self.xtraincekKfold(self.KfoldCurrentX_train)
            self.Ytraincek(self.KfoldCurrenty_train)
            self.xtestcekKfold(self.KfoldCurrentX_test)
            self.Ytestcek(self.KfoldCurrenty_test)
    
    
        
    def holdout(self):
        self.holdOutBool=True
        self.k_foldBool=False
        content = self.trainper.currentText() 
        X_train, X_test, y_train, y_test = train_test_split(self.Xlist, self.ylist, test_size=float(content), random_state=10)
        self.HX_train, self.HX_test, self.Hy_train, self.Hy_test=X_train, X_test, y_train, y_test
        self.xtraincekHold(self.HX_train)
        self.xtestcekHold(self.HX_test)
        self.Ytestcek(y_test)
        self.Ytraincek(y_train)
        
        
    def Ytestcek(self, y_test):
        y_test = pd.DataFrame(y_test)
        c=len(y_test.columns)
        r=len(y_test.values)
        self.tv_yTest.setColumnCount(c)
        self.tv_yTest.setRowCount(r)
        for i,row in enumerate(y_test):
                 for j,cell in enumerate(y_test.values):
                     self.tv_yTest.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
        
    
    
    def Ytraincek(self, y_train):
        y_train = pd.DataFrame(y_train)
        c=len(y_train.columns)
        r=len(y_train.values)
        self.tb_yTrain.setColumnCount(c)
        self.tb_yTrain.setRowCount(r)
        for i,row in enumerate(y_train):
                 for j,cell in enumerate(y_train.values):
                     self.tb_yTrain.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
                     
                     
                     
        
    def xtestcekHold(self,X_test):
        nsamplesTest, nxTest, nyTest = X_test.shape
        X_test = X_test.reshape((nsamplesTest,nxTest*nyTest))
        X_test = pd.DataFrame(X_test)
        c=len(X_test.columns)
        r=len(X_test.values)
        self.tv_Xtest.setColumnCount(c)
        self.tv_Xtest.setRowCount(r)
        for i,row in enumerate(X_test):
                 for j,cell in enumerate(X_test.values):
                     self.tv_Xtest.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
                     
                     
                     
                     
    def xtestcekKfold(self,X_test):
        X_test = pd.DataFrame(X_test)
        c=len(X_test.columns)
        r=len(X_test.values)
        self.tv_Xtest.setColumnCount(c)
        self.tv_Xtest.setRowCount(r)
        for i,row in enumerate(X_test):
                 for j,cell in enumerate(X_test.values):
                     self.tv_Xtest.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
        
        
        
    def xtraincekHold(self,xtrain):
        nsamples, nx, ny = xtrain.shape
        xtrain = xtrain.reshape((nsamples,nx*ny))
        xtrain=pd.DataFrame(xtrain)
        c=len(xtrain.columns)
        r=len(xtrain.values)
        self.tb_Xtrain.setColumnCount(c)
        self.tb_Xtrain.setRowCount(r)
        for i,row in enumerate(xtrain):
                 for j,cell in enumerate(xtrain.values):
                     self.tb_Xtrain.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
                     
                     
                     
    def xtraincekKfold(self,xtrain):
        xtrain=pd.DataFrame(xtrain)
        c=len(xtrain.columns)
        r=len(xtrain.values)
        self.tb_Xtrain.setColumnCount(c)
        self.tb_Xtrain.setRowCount(r)
        for i,row in enumerate(xtrain):
                 for j,cell in enumerate(xtrain.values):
                     self.tb_Xtrain.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
        
    
    
            
            
    def egit(self):
        if(self.holdOutBool):
            if(self.algseccb.currentText()=="KNN"):
                self.KNN(0, self.HX_train, self.HX_test, self.Hy_train, self.Hy_test)
            elif(self.algseccb.currentText()=="DT"):
                self.dt(0, self.HX_train, self.HX_test, self.Hy_train, self.Hy_test)
            elif(self.algseccb.currentText()=="SVM"):
                self.svm(0, self.HX_train, self.HX_test, self.Hy_train, self.Hy_test)
            elif(self.algseccb.currentText()=="LR"):
                self.logreg(0, self.HX_train, self.HX_test, self.Hy_train, self.Hy_test)
        else:
            if(self.algseccb.currentText()=="KNN"):
                self.KNN(1, self.KfoldCurrentX_train, self.KfoldCurrentX_test, self.KfoldCurrenty_train, self.KfoldCurrenty_test)
            elif(self.algseccb.currentText()=="DT"):
                self.dt(1, self.KfoldCurrentX_train, self.KfoldCurrentX_test, self.KfoldCurrenty_train, self.KfoldCurrenty_test)
            elif(self.algseccb.currentText()=="SVM"):
                self.svm(1, self.KfoldCurrentX_train, self.KfoldCurrentX_test, self.KfoldCurrenty_train, self.KfoldCurrenty_test)
            elif(self.algseccb.currentText()=="LR"):
                self.logreg(1, self.KfoldCurrentX_train, self.KfoldCurrentX_test, self.KfoldCurrenty_train, self.KfoldCurrenty_test)
        print("Tamamlandı.")  
        self.groupBox_4.setStyleSheet("QGroupBox { border: 1px solid green;}")
        
            

    def HE(self):
        g_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        hist = cv2.equalizeHist(g_img)
        if(self.kpA==0):
            self.siftkp(hist)
        elif(self.kpA==1):
            self.surfkp(hist)



    def clahe(self):
        g_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        clh = cv2.createCLAHE(2.0, (8, 8))
        cl1 = clh.apply(g_img)
        if(self.kpA==0):
            self.siftkp(cl1)
        elif(self.kpA==1):
            self.surfkp(cl1)
    

    def siftkp(self, image):
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(image, None)
        img_kp = cv2.drawKeypoints(image, kp, self.img)
        img_kp = cv2.cvtColor(img_kp, cv2.COLOR_BGR2GRAY)
        pathlib.Path('./Keypoints').mkdir(parents=True, exist_ok=True)
        cv2.imwrite(("./Keypoints/"+str(self.imagesay)+".KeypointImage.jpg"), img_kp)
        self.fileNameListKeypoint.append((str(self.imagesay)+".KeypointImage.jpg"))
        self.daisy(img_kp, kp)
    
    
    def surfkp(self, image):
        surf = cv2.xfeatures2d.SURF_create()
        kp, descriptors = surf.detectAndCompute(image, None)
        img_kp = cv2.drawKeypoints(image, kp, self.img)
        img_kp = cv2.cvtColor(img_kp, cv2.COLOR_BGR2GRAY)
        pathlib.Path('./Keypoints').mkdir(parents=True, exist_ok=True)
        cv2.imwrite(("./Keypoints/"+str(self.imagesay)+".KeypointImage.jpg"), img_kp)
        self.fileNameListKeypoint.append((str(self.imagesay)+".KeypointImage.jpg"))
        self.daisy(img_kp, kp)
        
        
    def daisy(self, img_kp, kp):
        count=0
        img_width = img_kp.shape[1]
        img_height = img_kp.shape[0]
        self.descs.clear()
        random.shuffle(kp)
        XListe=[]
        YListe=[]
        descs=[]
        for s1,i in enumerate(kp):
            if(count<=self.cropsayisi):
                x,y = int(i.pt[0]), int(i.pt[1])
                if  (x-self.daisyn)>0 and (y-self.daisyn)>0 and (x+self.daisyn)<img_width and (y+self.daisyn)<img_height:
                    a=x-self.daisyn
                    b=x+self.daisyn
                    c=y-self.daisyn
                    d=y+self.daisyn
                    crop_image = img_kp[c:d, a:b] 
                    desc, desc_img = daisy(crop_image, step=90, radius=3, rings=2, histograms=5,
                                          orientations=5, visualize=True)
                    desc=desc.reshape(desc.shape[0],desc.shape[1]*desc.shape[2])
                    desc=resize(desc, (28, 28))
                    desc=desc.flatten()
                    descs.append(desc)
                    count+=1
            else:
                break

        XListe.append(descs)
        XListe=np.array(XListe)
        XListe=XListe.reshape(XListe.shape[0],XListe.shape[1]*XListe.shape[2])
        YListe.append(self.deger)
        self.Xcrop.append(XListe)
        self.Xlist=np.array(self.Xcrop)
        self.ylist.append(YListe)
        print("---------------")
    
        
        
    def KNN(self, inf, X_train, X_test, y_train, y_test):
        if(inf==0):
            nsamples, nx, ny = X_train.shape
            X_train = X_train.reshape((nsamples,nx*ny))
            nsamplesTest, nxTest, nyTest = X_test.shape
            X_test = X_test.reshape((nsamplesTest,nxTest*nyTest))  
        k_range = list(range(1,26))
        scores = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            scores.append(metrics.accuracy_score(y_test, y_pred))
        acc=(metrics.accuracy_score(y_test, y_pred))*100
        self.confmat(y_test, y_pred, "KNN")
        self.label_acc.setText("ACC : {:.2f}%".format(acc))
        self.pltRoc(y_test,y_pred,"CNN ROC")
        
        
        
    def logreg(self, inf, X_train, X_test, y_train, y_test):
        if(inf==0):
            nsamples, nx, ny = X_train.shape
            X_train = X_train.reshape((nsamples,nx*ny))
            nsamplesTest, nxTest, nyTest = X_test.shape
            X_test = X_test.reshape((nsamplesTest,nxTest*nyTest))       
        log = LogisticRegression()
        log.fit(X_train,y_train)
        prediction=log.predict(X_test)
        print('ACC : ',metrics.accuracy_score(prediction,y_test)) 
        self.label_acc.setText("ACC : {:.2f}".format(metrics.accuracy_score(prediction,y_test)))
        self.confmat(y_test, prediction, "Logisitic")
        self.pltRoc(y_test,prediction,"CNN ROC")
        
        
    def dt(self, inf, X_train, X_test, y_train, y_test):
        if(inf==0):
            nsamples, nx, ny = X_train.shape
            X_train = X_train.reshape((nsamples,nx*ny))
            nsamplesTest, nxTest, nyTest = X_test.shape
            X_test = X_test.reshape((nsamplesTest,nxTest*nyTest))
        tree=DecisionTreeClassifier()
        tree.fit(X_train,y_train)
        prediction=tree.predict(X_test)
        print('ACC : ',metrics.accuracy_score(prediction,y_test))
        self.label_acc.setText("ACC : {:.2f}".format(metrics.accuracy_score(prediction,y_test)))
        self.confmat(y_test, prediction, "DT")
        self.pltRoc(y_test,prediction,"CNN ROC")
        

    def svm(self, inf, X_train, X_test, y_train, y_test):
        if(inf==0):
            nsamples, nx, ny = X_train.shape
            X_train = X_train.reshape((nsamples,nx*ny))
            nsamplesTest, nxTest, nyTest = X_test.shape
            X_test = X_test.reshape((nsamplesTest,nxTest*nyTest))
        model = SVC()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        print('ACC : ',metrics.accuracy_score(prediction,y_test))
        self.label_acc.setText("ACC : {:.2f}".format(metrics.accuracy_score(prediction,y_test)))
        self.confmat(y_test, prediction, "SVM")
        self.pltRoc(y_test,prediction,"SVMROC")
        
    

    def cnn(self):
        self.ylist=np.array(self.ylist)
        self.Xlist=self.Xlist.reshape(self.Xlist.shape[0],self.Xlist.shape[1]*self.Xlist.shape[2])
        scaler = StandardScaler()
        self.Xlist = scaler.fit_transform(self.Xlist)
        X_train, X_test, y_train, y_test = train_test_split(self.Xlist, self.ylist, test_size=0.2, random_state=2)
        import tensorflow as tf
        inputs = tf.keras.Input(shape=(self.Xlist.shape[1]))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc')
            ]
        )

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            batch_size=int(self.eptb_3.text()),
            epochs=int(self.bstb_4.text()),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
        y_true = np.array(y_test)
        y_pred = np.squeeze(np.array(model.predict(X_test) ,dtype=np.int))
        self.textEdit_2.setText(str(history.history['accuracy']) + "\n" + str(history.history['loss']))
        self.textEdit.setText(str(round(accuracy_score(y_true, y_pred)*100, 2)))
        cm = confusion_matrix(y_true, y_pred)
        self.textEdit_3.setText(str(cm))
        self.pltRoc(y_test,y_true,"CNN ROC")
        print("Tamamlandı.")
        
    
    
    def confmat(self,y_test,y_pred,isim):
        cm = confusion_matrix(y_test, y_pred)
        cm_data = pd.DataFrame(cm)
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        sensitivity = (tp/(tp+fn))*100
        specificity = (tn/(tn+fp))*100
        plt.figure(figsize = (5,5))
        sns.heatmap(cm_data, annot=True,fmt="d")
        plt.title(isim)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        pathlib.Path('./PCAs').mkdir(parents=True, exist_ok=True)
        plt.savefig("./cnfmat.png")
        self.pixmap = QPixmap("./cnfmat.png")
        self.label_18.setPixmap(self.pixmap)
        self.textEdit_4.setText(str(sensitivity))
        self.textEdit_5.setText(str(specificity))
        plt.show()
    
    
    
        
    def pltRoc(self,y_test,y_pred,baslik):    
        from sklearn import preprocessing
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from sklearn.linear_model import LogisticRegression
        from sklearn import metrics
        le = preprocessing.LabelEncoder()      
        y_test = le.fit_transform(y_test)
        y_pred = le.fit_transform(y_pred)
        y_test=np.array(y_test)
        y_pred=np.array(y_pred)
        postotal=0
        for i in range(2):
            if np.count_nonzero(y_pred == i)!=0:
                postotal+=1
        postotal1=0
        for i in range(2):
            if np.count_nonzero(y_test == i)!=0:
                postotal1+=1
                
        if postotal==postotal1:
            lr_fpr, lr_tpr, thresholds  =metrics.roc_curve(y_test, y_pred, pos_label=postotal)
            plt.plot(lr_fpr, lr_tpr, marker='.', label='baslik')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.savefig('roc.png')
            plt.show()   
            photo_path2 = "./roc.png"
            self.label_12.setPixmap(QPixmap(photo_path2))
        else:
            lr_fpr, lr_tpr, thresholds  =metrics.roc_curve(y_test, y_pred, pos_label=postotal1)
            plt.plot(lr_fpr, lr_tpr, marker='.', label='baslik')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.savefig('roc.png')
            plt.show()   
            photo_path2 = "./roc.png"
            self.label_12.setPixmap(QPixmap(photo_path2))    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = window()
    window.show()
    sys.exit(app.exec())