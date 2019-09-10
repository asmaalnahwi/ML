import os
import pandas as pd
import nltk as n
import math
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import operator



prefix_list = ["a-", "an-", "ab-", "ad-", "ac-", "as-", "ante-", "anti-", "auto-", "ben-", "bi-", "circum-", "co-",
               "com-", "con-", "contra-", "counter-", "de-", "di-", "dis-", "eu-", "ex-", "e-", "exo-", "ecto-",
               "extra-", "extro-", "fore-", "hemi-", "hyper-", "hypo-", "il-", "im-", "in-", "ir-", "inter-", "intra-",
               "macro-", "mal-", "micro-", "mis-", "mono-", "multi-", "non-", "ob-", "o-", "oc-", "op-", "omni-",
               "over-", "peri-", "poly-", "post-", "pre-", "pro-", "quad-", "re-", "semi-", "sub-", "sup-", "sus-",
               "super-", "supra-", "sym-", "syn-", "trans-", "tri-", "ultra-", "un-", "uni-"]

file = "C:\\Users\\dell\\Documents\\bbc"
file2="C:\\Users\\dell\\Documents\\BBC test"







def OpenAndRead(file):
    frames = {}  ### count per file
    frames_2 = {}  ### count per catgory
    all_files = os.listdir(file)
    groups = all_files  ### names of training catgory
    for grp in groups:
        ds = pd.DataFrame()  #### count per file
        dc = pd.DataFrame()  ### count per catgory
        dx = pd.Series()  ### count per catgory
        da = pd.Series()  ### count per file
       
        k = os.listdir(file + "\\" + grp)
        for j in k:
            f = open(file +"\\" + grp + "\\" + j, 'r')
            message = f.read()
            message = message.lower()
            dashTest = n.word_tokenize(message)
            df = pd.Series(dashTest)
            dx = pd.concat([dx, df], axis=0)
            ############ count per file
            df = df.value_counts()
            da = pd.concat([da, df], axis=0)
            da = da.sort_index()
            count_file = da.values
            word_file = da.index
            count_file = pd.Series(count_file)
            word_file = pd.Series(word_file)
            
            #############
            ds.fillna(" ", inplace=True)
        dx = dx.value_counts()  # -------------------------------------------------- count per  cartgory
        dx = dx.sort_index()
        word_cat = dx.index
        count_cat = dx.values
        count_cat = pd.Series(count_cat)
        word_cat = pd.Series(word_cat)
        dc["word"] = word_cat  ##per catgory words
        dc["count"] = count_cat  # per catgory counts
        ds["word"] = word_file  ##per file words
        ds["count"] = count_file  # per file counts
        frames[grp] = ds
        frames_2[grp] = dc
    wc = WeightCalc(groups, frames , file)
    
    return wc,frames




def OpenAndReadAndClean(file):
    frames = {}  ### count per file
    frames_2 = {}  ### count per catgory
    all_files = os.listdir(file)
    groups = all_files  ### names of training catgory
    for grp in groups:
        ds = pd.DataFrame()  #### count per file
        dc = pd.DataFrame()  ### count per catgory
        dx = pd.Series()  ### count per catgory
        da = pd.Series()  ### count per file
       
        k = os.listdir(file + "\\" + grp)
        for j in k:
            f = open(file +"\\" + grp + "\\" + j, 'r')
            message = f.read()
            message = message.lower()
            lists = DashHandel(message)
            string_text = ""
            dashTest = pd.Series(lists)  ## convert list to Series
#             ********* convert series of words to text to tokenize *****#
            for dash in dashTest:
                string_text = string_text + dash + " "

            dashTest = RemoveNoise(string_text)
            x = Lemma(dashTest)
            df = RemoveStopwords(x)
            dx = pd.concat([dx, df], axis=0)
            ############ count per file
            df = df.value_counts()
            da = pd.concat([da, df], axis=0)
            da = da.sort_index()
            count_file = da.values
            word_file = da.index
            count_file = pd.Series(count_file)
            word_file = pd.Series(word_file)
             
            #############
            ds.fillna(" ", inplace=True)
        dx = dx.value_counts()  # -------------------------------------------------- count per  cartgory
        dx = dx.sort_index()
        word_cat = dx.index
        count_cat = dx.values
        count_cat = pd.Series(count_cat)
        word_cat = pd.Series(word_cat)
        dc["word"] = word_cat  ##per catgory words
        dc["count"] = count_cat  # per catgory counts
        ds["word"] = word_file  ##per file words
        ds["count"] = count_file  # per file counts
        frames[grp] = ds
        frames_2[grp] = dc
    wc = WeightCalc(groups, frames , file)
    
    return wc,frames




def DashHandel(message):
    lists = []  ## contain words without dashes in middle
    dashTest = []
    dashTest = n.word_tokenize(message)
    dashTest = pd.Series(dashTest)

    ###################### condtion to make sure thers is no dashes in the middle of two words   ######################
    for dash in dashTest:

        if "-" in dash:
            a, b, c = dash.partition("-")
            dash = a + b
            for w in prefix_list:
                if dash == w:
                    lists.append(dash.replace("-", ""))
                    break
                else:
                    dash = dash.replace("-", " ")
                    lists.append(dash + c)
                    break
        else:
            lists.append(dash)
    ######################
    return lists


def RemoveNoise(string_text):
    string_text = ''.join(c for c in string_text if not c.isnumeric())  # ---------- digits

    dashTest = n.word_tokenize(string_text)

    dashTest = [w for w in dashTest if w.isalnum()]  # --------------- lower + remove punch

    return pd.Series(dashTest)


def Lemma(dashTest):
    result = []
    lemmatizer = WordNetLemmatizer()
    result = (
    [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in dashTest])  ### call the function and lemmatie each word
    return result


def get_wordnet_pos(word):  ### function to know the POS of the word
    """Map POS tag to first character lemmatize() accepts"""
    tag = n.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def RemoveStopwords(x):
    f_StopWords = open('C:\\Users\\dell\\Documents\\Stop\\stopwords.txt','r')  # custome file for stopwords
    add_StopWords = f_StopWords.read()  # contain the text from each file
    add_StopWords = add_StopWords.lower()  # all characters in lower case
    add_StopWords = n.word_tokenize(add_StopWords)  # tokenize
    add_StopWords = [w for w in add_StopWords if w.isalnum()]  # remove punch

    stopwords = n.corpus.stopwords.words('english')
    filtered_sentence = []  # new list to save all the words that are not in the stopwords list
    stopwords.extend(add_StopWords)  # combine the custom stopwords with the predefinded one
    for w in x:
        if w not in stopwords:
            filtered_sentence.append(w)  # add the word to the new list
    return pd.Series(filtered_sentence)  # convert the list to series


def WeightCalc(groups, frames, file):
    frames_3 = {}  #### wights
    zero = pd.Series([0])
    for grp in groups:
        k = os.listdir(file+ "\\" + grp)
        k = len(k)
        dw = pd.DataFrame()
        currentIndex = 0  # ------ cuttent index
        currentword = frames[grp]['word']
        currentword = currentword.append(zero, ignore_index=True)
        word = currentword[0]
        currentcount = frames[grp]['count']
        currentcount = currentcount.append(zero, ignore_index=True)
        count = currentcount[0]
        listOfword = []
        evar = [count]
        evartotal = count
        weightWithinCats = []
        weightAcrossCats = []
        for i in range(currentword.size):
            if currentIndex == 0:
                currentIndex = 1
                continue
            newword = currentword[currentIndex]
            newcount = currentcount[currentIndex]
            if newword == word:
                evar.append(newcount)
                evartotal = evartotal + newcount
                currentIndex = currentIndex + 1
                word = newword
                count = newcount
            else:  # ------- the algorithm startes
                # ------- Count the average
                listOfword.append(word)
                avrg = 0.0
                summ = 0.0
                eta = 0.0
                weightwithen = 0.0
                weightacross=0.0
                evesize = len(evar)
                sub = k - evesize
                evar.extend([0] * sub)
                for a in range(len(evar)):
                    avrg = avrg + evar[a]
                avrg = avrg / float(len(evar))
                # ------ Sum of the frequency minus the average all square
                for b in range(len(evar)):
                    summ = summ + pow(evar[b] - avrg, 2)

                eta = (1 / (len(evar) * (avrg))) * (math.sqrt(summ))
                weightwithen = evartotal * (1 - eta)
                weightacross=evartotal*eta
                # ----- Save the result in list
                weightWithinCats.append(weightwithen)
                weightAcrossCats.append(weightacross)
                # ----- Reset the loop variables

                word = currentword[currentIndex]
                count = currentcount[currentIndex]
                evar.clear()
                evar = [count]
                evartotal = count
                currentIndex = currentIndex + 1
        dw['words'] = pd.Series(listOfword)
        dw['weightwithen'] = pd.Series(weightWithinCats)
        dw['weightacross']=pd.Series(weightAcrossCats)
        frames_3[grp] = dw
    return frames_3

def semantic_words(frame3):
    frame_s=frame3
    cat=list(frame_s.keys())
    for catgory in cat:
        for w in frame_s[catgory]["words"]:
            wieght_s=0.0
            ind2=frame_s[catgory]["weightacross"][frame_s[catgory]["words"]== w].index[0]
            we=frame_s[catgory]["weightacross"][ind2]
            synonyms = []
            for syn in wordnet.synsets(w):
                for l in syn.lemmas():
                   synonyms.append(l.name())
            synonyms=list(set(synonyms))
            
            for s in synonyms:
                if s.lower()==w: 
                    synonyms.remove(s)
            wieght_s=we*0.50
            
            
            if(len(synonyms)!=0):
                for s in synonyms:
                   frame_s[catgory].at[len(frame_s[catgory]["words"])+1,"words"]=s
                   frame_s[catgory].at[len(frame_s[catgory]["weightacross"]),"weightacross"]=wieght_s
    return frame_s
            
            
 
def innerproduct(train,file):
    frame_final={}
    cat=list(train.keys())
    all_files = os.listdir(file)

    groups = all_files  ### names of training catgory
    for grp in groups:
        k = os.listdir(file + "\\" + grp)
        for j in k:
            dataframe=pd.DataFrame(index=cat,columns=['weight'])
            f = open(file +"\\" + grp + "\\" + j, 'r')
            message = f.read()
            message = message.lower()
            lists = DashHandel(message)
            string_text = ""
            dashTest = pd.Series(lists)  ## convert list to Series
            # ********* convert series of words to text to tokenize *****#
            for dash in dashTest:
                string_text = string_text + dash + " "

            dashTest = RemoveNoise(string_text)
            x = Lemma(dashTest)
            df = RemoveStopwords(x)
            count_first = df.value_counts()#seprated
            count_s = count_first.values
            word_s = count_first.index
            count_s = pd.Series(count_s)
            word_s = pd.Series(word_s)
            
            
            for catgory in cat:
                evar = 0.0
                for word in word_s:
                    for wordC in train[catgory]["words"]:
                        
                        if word==wordC:
                            ind1=word_s[word_s== word].index[0]
                            count=count_s[ind1]
                            ind2=train[catgory]["weightacross"][train[catgory]["words"]== wordC].index[0]
                            W=train[catgory]["weightacross"][ind2]
                            evar = evar + (count * W)
                            break

                            
                    dataframe.at[catgory,'weight']=evar
                    
            frame_final[j]=dataframe
    return frame_final


def countTrue (result):
    files=list(result.keys())
    count=0
    cati=list(result[files[0]].index.values)
    for f in files :
        max_index, max_value = max(enumerate(result[f]['weight']), key=operator.itemgetter(1))
        if cati[max_index] in f:
            count=count+1
    return count
            
        



#m= OpenAndRead(file)
train_3,train = OpenAndReadAndClean(file)
#train_3_semantic=semantic_words(train_3)
result=innerproduct(train_3,file2)
#count=countTrue(result)




