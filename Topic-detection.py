"""
Created on Mon Mar 11 01:14:03 2019

@author: dell


"""
from  tkinter import *
from tkinter import filedialog
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.svm import *




##################### redaing files pathes ##########################################
root = Tk()
root.filename =  filedialog.askopenfilenames(initialdir = "/",title = "please enter your files for classification",filetypes = (("txt files","*.txt"),("all files","*.*")))

# hold the pathes of each file
files=list(root.filename)
root.withdraw()
root.destroy()

############################reading files content ##################################################
ds=pd.DataFrame()
list_content=[]

for j in files:
            f = open(j, 'r')
            message = f.read()
            message = message.lower()
            list_content.append(message)
            
            
ds['file']=pd.Series(list_content)
################################### load the model ####################################################################

filename_model = 'C:\\Users\\dell\\Desktop\\Senior project\\finalized_model2.pickle'
filename_vectorizer= 'C:\\Users\\dell\\Desktop\\Senior project\\vec3.pickle'
vectorizer = pickle.load(open(filename_vectorizer, 'rb'))
loaded_model = pickle.load(open(filename_model, 'rb'))

###############################################################################


################################### predict the labels ####################################################################

vectorize_text = vectorizer.transform(ds.file)
ynew = loaded_model.predict(vectorize_text)
ypep= loaded_model.predict_proba(vectorize_text)

# reduse the digits 
for i in range(len(ypep)):
    for j in range(len(ypep[i])):
        ypep[i][j]="%.3f" % ypep[i][j]
data=pd.DataFrame(data=ypep,columns=['business', 'entertainment', 'politics', 'sport', 'tech'])
data=data*100

data = data.astype(str) + '%'
data['predicted']=ynew


##################################saving the result in CSV file ##########################################################3
filename_Result=r'C:\\Users\\dell\\Downloads\\Results.csv'

data.to_csv(filename_Result, index = None, header=True)

print("************* Your Result is saved in "+filename_Result+ "*************")






            
            

