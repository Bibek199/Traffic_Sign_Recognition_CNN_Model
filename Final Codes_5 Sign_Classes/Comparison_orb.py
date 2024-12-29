import cv2
import pandas as pd

Train00=pd.read_csv("Sh00.csv", usecols = ['Path','ClassId'])
Train03=pd.read_csv("Sh03.csv", usecols = ['Path','ClassId'])
Train14=pd.read_csv("Sh14.csv", usecols = ['Path','ClassId'])
Train33=pd.read_csv("Sh33.csv", usecols = ['Path','ClassId'])
Train34=pd.read_csv("Sh34.csv", usecols = ['Path','ClassId'])
TestSet=pd.read_csv("Test_Robot_orb.csv", usecols = ['Path','ClassId'])

Reference=pd.read_csv("Meta.csv", usecols = ['Path','ClassId'])
Ref00=Reference.loc[1][0]
Ref00_ID=Reference.loc[1][1]

Ref03=Reference.loc[23][0]
Ref03_ID=Reference.loc[23][1]

Ref14=Reference.loc[7][0]
Ref14_ID=Reference.loc[7][1]

Ref33=Reference.loc[27][0]
Ref33_ID=Reference.loc[27][1]

Ref34=Reference.loc[28][0]
Ref34_ID=Reference.loc[28][1]

Path="D:\# NU Master\\2nd sem\Spring2023\Robotics and vision\GSRTB_Datasets\\"

ref_img00=cv2.resize(cv2.imread(Path+Ref00),(500,500))
ref_img03=cv2.resize(cv2.imread(Path+Ref03),(500,500))
ref_img14=cv2.resize(cv2.imread(Path+Ref14),(500,500))
ref_img33=cv2.resize(cv2.imread(Path+Ref33),(500,500))
ref_img34=cv2.resize(cv2.imread(Path+Ref34),(500,500))

def Calculate_Threshold(ref_img,Train,Path):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher()
    kp_ref, des_ref = orb.detectAndCompute(ref_img, None)
    matches_sum = 0
    for n in range(len(Train)):
        img_path = Train.loc[n][1]
        img = cv2.resize(cv2.imread(Path + img_path), (500,500))
        kp_img, des_img = orb.detectAndCompute(img, None)
        matches = bf.match(des_ref, des_img)
        matches_sum = matches_sum + len(matches)
    Avg = matches_sum / len(Train)

    return Avg

def Calculate_Confusion_Matrix(ref_img,Ref_ID,TestSet,Path,Thr):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher()
    kp_Ref, des_Ref = orb.detectAndCompute(ref_img, None)

    TP=0
    TN=0
    FP=0
    FN=0

    for n in range(len(TestSet)):
        imgTest_path=TestSet.loc[n][1]
        imgTest_ID=TestSet.loc[n][0]
        imgTest=cv2.resize(cv2.imread(Path+imgTest_path),(500,500))
        kp_imgTest,des_imgTest=orb.detectAndCompute(imgTest, None)
        matches=len(bf.match(des_Ref, des_imgTest))

        if (imgTest_ID==Ref_ID) & (matches>=Thr):
            TP=TP+1
        elif (imgTest_ID==Ref_ID) & (matches<Thr):
            FN=FN+1
        elif (imgTest_ID!=Ref_ID) & (matches>=Thr):
            FP=FP+1
        elif (imgTest_ID!=Ref_ID) & (matches<Thr):
            TN=TN+1

    Accuracy = (TP + TN) * 100 / (TP + TN + FP + FN)
    Precision = TP * 100 / (TP + FP)
    Recall = TP * 100 / (TP + FN)
    F1_score = (2 * Precision * Recall) / (Precision + Recall)
    Confusion_Matrix="TP, TN, FP, FN"
    Comparison_Parameters="Accuracy,Precision,Recall,F1_score"

    return Confusion_Matrix,TP,TN,FP,FN,Comparison_Parameters,Accuracy,Precision,Recall, F1_score

'''
print("For Class 0 (Limit Speed 20)")
Thr00=Calculate_Threshold(ref_img00,Train00,Path)
print(Thr00)
print(Calculate_Confusion_Matrix(ref_img00,Ref00_ID,TestSet,Path,Thr00))

print("For Class 3 (Limit Speed 60)")
Thr03=Calculate_Threshold(ref_img03,Train03,Path)
print(Thr03)
print(Calculate_Confusion_Matrix(ref_img03,Ref03_ID,TestSet,Path,Thr03))

print("For Class 14 (Stop)")
Thr14=Calculate_Threshold(ref_img14,Train14,Path)
print(Thr14)
print(Calculate_Confusion_Matrix(ref_img14,Ref14_ID,TestSet,Path,Thr14))

print("For Class 33 (Turn Right Ahead)")
Thr33=Calculate_Threshold(ref_img33,Train33,Path)
print(Thr33)
print(Calculate_Confusion_Matrix(ref_img33,Ref33_ID,TestSet,Path,Thr33))

print("For Class 34 (Turn Left Ahead)")
Thr34=Calculate_Threshold(ref_img34,Train34,Path)
print(Thr34)
print(Calculate_Confusion_Matrix(ref_img34,Ref34_ID,TestSet,Path,Thr34))
'''
print("For Class 34 (Turn Left Ahead)")
Thr34=Calculate_Threshold(ref_img34,Train34,Path)
print(Thr34)
print(Calculate_Confusion_Matrix(ref_img34,Ref34_ID,TestSet,Path,Thr34))

print("For Class 33 (Turn Right Ahead)")
Thr33=Calculate_Threshold(ref_img33,Train33,Path)
print(Thr33)
print(Calculate_Confusion_Matrix(ref_img33,Ref33_ID,TestSet,Path,Thr33))

print("For Class 14 (Stop)")
Thr14=Calculate_Threshold(ref_img14,Train14,Path)
print(Thr14)
print(Calculate_Confusion_Matrix(ref_img14,Ref14_ID,TestSet,Path,Thr14))

print("For Class 0 (Limit Speed 20)")
Thr00=Calculate_Threshold(ref_img00,Train00,Path)
print(Thr00)
print(Calculate_Confusion_Matrix(ref_img00,Ref00_ID,TestSet,Path,Thr00))

print("For Class 3 (Limit Speed 60)")
Thr03=Calculate_Threshold(ref_img03,Train03,Path)
print(Thr03)
print(Calculate_Confusion_Matrix(ref_img03,Ref03_ID,TestSet,Path,Thr03))