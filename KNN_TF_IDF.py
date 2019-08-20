import operator
import re
import nltk
import math
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.append("'s")

###### open the data file 
fp1 = open('data_train_200.txt')
data = fp1.readlines()

stemmer = nltk.PorterStemmer()
   
###### delete the stop words and creat a normal list of article words by stemming 
all_features_data = []
for line in data:
    article_words = [word for word in line.split() if word not in stop_words]
    #article_words = [word for word in article_words if len(word) > 2]
    ''' stemming the words with extracting features '''
    features = [stemmer.stem(word) for word in article_words]
    all_features_data.append(features)
#print(len(all_features_data))
    

###### get the word types in features 
type_features0 = {}
for feature in all_features_data:
    for word in feature:
        try:
            type_features0[word] += 1
            #print('add value')
        except:
            type_features0[word] = 1
            #print ('add word')
type_features1 = {}
for k in type_features0:
    if type_features0[k] > 3:
        type_features1[k] = type_features0[k]
type_features = list(type_features1)
ftu = len(type_features)
print('number of features = ' ,ftu)



###### open the labales of train file 
fp2 = open('labels_train_200.txt')
txt = fp2.read()
lable = txt.split()

###### open the labales of test file
fp4 = open('labels_valid_100.txt')
txt1 = fp4.read()
lable_test = txt1.split()

###### compute the norm of a vector 
def normalize(article_vector):
    normal = 0
    for i in range(len(article_vector)):
        normal = normal + (article_vector[i]) ** 2
    normal = math.sqrt(normal)
    return normal


###### we want to see howmany of articles contain the word 
def num_docs_containing(word, articles):
    count = 0
    for line in articles:
        if word in line:
        #if line.count(word) > 0:
            count += 1
    return 1+ count


###### vectorize the data 
m = len(all_features_data)
data_matrix = [[0 for i in range(ftu)] for j in range(m)]
j = 0
for line in all_features_data:
    i = 0
    for word in type_features:
        count = 0
        count = line.count(word)
        data_matrix[j][i] = count
        i+= 1
    j+= 1
###### tf matrix constructed 
#print(data_matrix[0])     
tf = 0
idf = 0
idf_list = []
for j in range(m):
   # print ('j=',j, end = '\t')
    n = normalize(data_matrix[j])
    #print(n)
    i = 0
    for f in data_matrix[j]:
    #    print ('i=',i, end = '\t')
        tf = f / n
        idf = math.log(m/ num_docs_containing(type_features[i], all_features_data))
        #print(idf)
        idf_list.append(idf)
        data_matrix[j][i] = tf * idf
     #   print ('tf-idf=',tf*idf)
        i +=1
   # print(data_matrix[j])
    #print()
   
print ('tf-idf matrix for train data built!')
    
###### open rhe test file 
fp3 = open('data_valid_100.txt')
test = fp3.readlines()
all_features_test = []
for line in test:
    article_words = [word for word in line.split() if word not in stop_words]
    features = [stemmer.stem(word) for word in article_words]
    all_features_test.append(features)
    
###### create test_matrix 
h = len(all_features_test)
test_matrix =[[0 for i in range(ftu)] for j in range(h)]
j = 0
for line in all_features_test:
    i = 0
    for word in type_features:
        count = 0
        count = line.count(word)
        test_matrix[j][i] = count
        i+= 1
    j+= 1
      
######crate tf_idf matrix for data_valid
tf = 0
idf = 0
for j in range(h):
    N = normalize(test_matrix[j])
    i = 0
    for f in range(ftu):
        test_matrix[j][i] = test_matrix[j][i] * idf_list[i]/N
        i +=1
   # print(test_matrix[j])
    #print()

print ('tf-idf matrix for test data built!')


def manhattan(vector1, vector2):
        """Computes the Manhattan distance."""
        return sum(map(lambda v1, v2: abs(v1 - v2), vector1, vector2))

k = int(input("enter K for KNN: "))

######compute the distances for each test data with whole train data and select the k nearest articles foe each test article
#def nearest_neighbours(train_data, test_data, type_features, k, data_lables):
print("computing the class of test docs ...")
all_nearest_neighbours = []
distances = {}
diatance = 0
b = 0
x = 0
for test in test_matrix:
    a = 0
    for neighbour in data_matrix:
        distance = manhattan(test , neighbour)
        #article_num = data_matrix.index(neighbour)
        distances[a] = distance
        a += 1

    #######find the nearest neighbours indices 
    dist_list = []
    sorted_distances = sorted(distances.items(), key=operator.itemgetter(1))
    dist_list = sorted_distances[0:k]
    NN = dict(dist_list)
    NN_indices = list(NN.keys())
    KNN = []
    for article in NN_indices:
        KNN.append(lable[article])
        
    ########find the major class for tested articles
    major = {}
    test_article_class = []
    for cls in KNN:
        if cls in major:
            major[cls] += 1
        else:
            major[cls] = 1
    test_article_class = [str(k) for k, v in major.items() if v == max(major.values())]
    print(test_article_class[0])
    if test_article_class[0] == lable_test[b]:
        x += 1
    b += 1
print ("accuracy = %", float(x)*100/b)
''' result = ''
#for i in range(len(test_matrix)):
    #result += str(test_article_class[i][0])+'\n'         
        #return result

#m = nearest_neighbours(all_features_data, all_features_test, type_features, 3, lable)

#fp4 = open('d:\\labels_test_man.txt' , 'w')
#fp4.write(m)

'''


fp1.close()
fp2.close()
fp3.close()
fp4.close()

