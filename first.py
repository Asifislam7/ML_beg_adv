import random
# import numpy as np is used to import numpy library to make arrays
import numpy as np # type: ignore
import pandas as pd
from io import StringIO
import json
import seaborn as sns
import pickle
# # print("hello Asif\nArif")
# # name=input("What is your name") 
# # # length= len(name)
# # # print(length)
# # # x=input("Enter a number: ")
# # # first_dig= x[0]
# # # second_dig= x[1]
# # # print(int(first_dig)+ int(second_dig))
# # print(4/2)
# # print(10/3)
# # print(4//52)
# # print(2**3)
# a,b,c= 5,5,6
# print(a,b,c)
 
# # print(a is not b)
# # print(a==b)
# # print(id(a))
# # print(id(b))

# # # identity operator

# # # python is OOP so every object has its own id 
# # # membership operator
# # l= [1,10,-1,0,12]
# # print(10 not in l)
# # name= "Asif"
# # age= 23
# # height= 0.5
# # print(f"My name is {name}, I am {age} years old. My height is {height} meters")

# # # using if else statement

# # if(age> 18):
# #     print("Adult")
# #     print("another line")
# #     print("Out of if")

# # # nested if 

# # # if(age > 18):
# # #     print("now entering the real world")
# # #     if(height> 1):
# # #         print("taller person")
# # #         print("this is the same height if part")
# # #     else:
# # #         print("Not a taller person. need Complan bro")
# # # else:
# # #     print("Underage..let him grow 18")

# # #     # elif is also used as else if is used in other PL 


# #  learning list now in python 
# # roll_no= [1,2,3,4,5,6,7,8,9,10]
# # # print(roll_no[-1])  #starts from the last
# # # print(roll_no[0:3])   #here 0 is the index and 3 is the length number (not the index number)
# # # print(roll_no[0:10:3]) #this is the slicing of the array where 0 will print the values of the starting index
# # # #and :10 means till what value we have to go for that it should be and 3 means the no of values it will skip and print 
# # # print(roll_no[0::2])

# # # #print the numbers now 
# # # roll_no.sort()
# # # print(roll_no)

# # # roll_no.append(20)
# # # print(roll_no)
# # # roll_no.reverse()
# # # print(roll_no)
# # # roll_no.remove(10)
# # # print(roll_no)
# # # roll_no.extend([45,46,47])
# # # print(roll_no)

# # # #random module in python 

# # # a=random.randint(1,3)
# # # print(a)

# # # b=random.randrange(1,3)
# # # print(b)

# # # c=random.random()
# # # print(c)



# # # d=random.uniform(1,3)
# # # print(d)
# # # random.shuffle(roll_no)
# # print(roll_no)

# # # names=input("Enter the name of th people to pick randomly with spaces between them ")
# # # name_list = names.split()
# # # print(name_list)
# # # rand_no= random.randint(0,len(name_list)-1)
# # # print(name_list[rand_no])



# # #nested list
# # newlist= [1,2,3,[6,7,8],9,10]
# # # print(newlist[3])


# # # tuples in python 
# # tuple1=(10,1,-10,15,20)
# # tuple2=("jenny","Ram")
# # # print(type(tuple1))

# # # these tuples are immutable
# # # tuple1[0]=5   #this will give an error because we can't change

# # # print(tuple1[1:4])
# # # print(tuple1[1:4:2])

# # # nesting of tuples is also allowed 
# # tuple3= (tuple1,tuple2)
# # # print(tuple3[1])


# # # sets in python 
# # # set1={10, True, 'Asif', 1.11}
# # set1={10, True, 'Asif', 1.11}
# # set2={20,"Arif", "Ruby", 30,"Asif"}

# # # duplicate itesmae not allowed in sets
# # # sets are unordered. Items does not have an order
# # # indexing is not allowed in sets


# # # operations on sets- union 
# # # print(set1.union(set2))
# # #we can use operators also 
# # # print(set2 | set1)

# # set3={"Asif","harsh" }
# # # print(set1.union(set3))

# # # update operation i sets
# # set1.update(set2)
# # # print(set1)

# # # python is case sensitive so if we add same names with different cases it accepts
# # # intersection
# # # print(set1.intersection(set3))
# # # difference between two sets gives elements present in first set and not in the second one
# # # print(set1.difference(set3))
# # # symmetric difference returns element which are there in either of the sets not in both, it tracks from both the sides
# # # check if a specific element exists or not
# # # print("harsh" in set3)
# # # symmetric difference is also checked
# # # print(set1.symmetric_difference(set2))

# # # range 
# # a= range(5,10,2)
# # # print(a[1])

# # for i in a:
# #     print(i)

# # #  -----------------------------------------?  # 

# # # functions in python 
# # # built in function and user defined functions
# # def welcome():
# #     return "Welcome to Python Programming!"
# # print (welcome())
# # # parameter passing - by value /by reference

# # def Asif():
# #     print("Asif islam is a coder")

# # # arguements in functions
# # def greet(name):
# #     print(f"My name is {name}")

# # greet("Asif") 
# # # these are actual parameters

# # def add(*args,name):
# #     sum=0
# #     print(args)
# #     for i in args:
# #         sum+=i
# #     print(f"Sumis {sum}")
# #     print(name)
# # add(2,3,4,5,name="Asif")





# # # now if we want to add nne more arguemnt to the function of other type we have to use = sign
# # add(2,3,name="Asif")

# # # dictionaries in python
# # # objects in python are called dictionaries

# # phone_no={
# #     "Asif": 1234,
# #     "Farhan": 2345,
# #     "Moteen": 6865
# #     # it uses key value pair for the data to be stored
# # }
# # # values are mutable for the key value pairs
# # print(phone_no['Farhan'])
# # phone_no["Farhan"]=9999
# # print(phone_no['Farhan'])
# # # deleting an element from dictionary using del keyword
# # del(phone_no['Farhan'])
# # print(phone_no)
# # phone_no['Farhan']={'Home': 64543, 'work': 54435}
# # print(phone_no['Farhan']['work'])
# # phone_no.pop('Farhan')
# # print(phone_no)
# # # clear will delete all the items
# # # phone_no.clear()
# # print(phone_no)
# # # keys are used to access only the keys  
# # print(phone_no.keys())
# # print(phone_no.items())
# # # nested dictionar is dic witihn dic

# # details={
# #     "dict1": {"Name": "Asif","Age": 22},
# #     "dict2": {"Name": "Asma","Age": 21},
# # }
# # print(details['dict1']['Age'])


# #functions with return values 
# # def add(a,b):
# #     c=a+b
# #     return c
# # print(add(7,5))


# # diff between print and return in python 
# # return will not show any value to the user but print will show the value ot the user
# # return is used to return the value to the calling function
# # print is used to print the value to the console


# #local vs global scope
# # local scope is the scope of the variable in the function
# # global scope is the scope of the variable outside the function
# # global variable is used to access the variable outside the function
# # local variable is used to access the variable inside the function

    
# #debugging in python
# # we can use print statements to debug the code
# # we can use breakpoints in the code to debug the code
# # we can use the debugger in the code to debug the code

# # dice_number=["one","two","three","four","five"]
# # dice_no= random.randint(1,5)
# # print(dice_no)


# #difference between bug and error
# # bug is the error in the code which is not working properly


# # numpy  library useage and importance
# # numpy is used for the numerical calculations in python
# # numpy is used for the array operations in python
# # numpy is used for the matrix operations in python
# # numpy is used for the linear algebra operations in python
# # numpy is used for the statistical operations in python
# # numpy is used for the random number generation in python
# # numpy is used for the fourier transformation in python
# # numpy is used for the image processing in python
# # numpy is used for the mathematical operations in python

# # used in eda for data science

# # lst=[1,2,3,4]
# # arr=np.array(lst)
# # print(arr)
# # print(type(arr)) #it gives n dimesnional array

# #making two dimensional array
# # arr1=np.array([[1,2,3],[4,5,6]])
# # print(arr1.shape) #two rows and three columns gives output as 2 and 3
# # print(arr1.size) #total number of elements in the array
# # print(arr1.ndim) #number of dimensions in the array

# #creating an array of zeros

# #arr[:-1] gives all the elements in the list except the last element

# arr2=np.zeros((2,3))
# #print(arr2)
# #creating an array of ones
# arr3=np.ones((2,3))
# #print(arr3)
# #creating an array of random numbers
# arr4=np.random.random((2,3))
# # print(arr4)
# #creating an array of random integers
# arr5=np.random.randint(1,10,(2,3))
# # print(arr5)
# #creating an array of random numbers with normal distribution
# arr6=np.random.randn(2,3)
# # print(arr6)
# #creating an array of random numbers with uniform distribution
# arr7=np.random.uniform(1,10,(2,3))
# # print(arr7)


# # array([1, 2, 3, 5])
# # arr[arr<2]
# # array([1])

# # array([[1, 2, 3, 4, 5],
# #        [2, 3, 4, 5, 6],
# #        [3, 4, 5, 6, 7]])
# # arr1.reshape(5,3)

# # array([[1, 2, 3],
# #        [4, 5, 2],
# #        [3, 4, 5],
# #        [6, 3, 4],
# #        [5, 6, 7]])



# # arr8=np.arange(1,11,1).reshape(5,2)
# # print(arr8)

# # print(arr8 * 2 );

# # Pandas library learning 
# # pandas is used for data manipulation and data analysis
# # pandas is used for data cleaning and data wrangling
# # pandas is used for data visualization
# # pandas is used for data transformation
# # pandas is used for data aggregation



# # if we write the data in the excel sheet in the form of rows and columns 
# # then it is called as dataframe in pandas
# # dataframe is a two dimensional data structure in pandas
# # dataframe is a collection of series
# # dataframe is a collection of rows and columns

# #creating Dataframe
# # arr10=np.arange(0,20).reshape(5,4)
# # #  this is to be converted to data frame
# # df=pd.DataFrame(data=arr10, index=['A','B','C','D','E'], columns=['W','X','Y','Z'])
# # print(df) 
# # df.info gives the datatype of the column 
# # df.describe gives the details of count mean std min 25% etc and only works on int and floating types

# # indexing in dataframe
# # columnanme, rowindex[loc], rowindex columnindex number [.iloc] - three methods are there

# # print(df[['W','X']]) #this will give the columns as mentioned
# # this was by using column name

# # difference between dataframe and series 
# # series is a one dimensional data structure in pandas( it has only one row or one column) where as dataframe has multiple rows and columns
# # series is a collection of elements where as dataframe is a collection of series


# # using row index
# # print(df.loc['A']) #this will give the row with index A 


# # using row index and column index iloc
# # print(df.iloc[2:4,0:2]) #this will give the element of the 2nd and 3rd row with 0th and 1st column element name   
# # print(df.iloc[0,1]) #this will give the element at 0th row and 1st column
# # print(df.iloc[2:4,0:2].values) #this will give the element of the 2nd and 3rd row with 0th and 1st column element name   
# #  and values will remove the row and column name 


# # operations 
# # print(df.isnull()) shows whether null is present or not (true or false)
# # print(df.dropna()) remove missing values
# # print(df.fillna(0)) fill the missing values with 0
# # print(df.drop_duplicates()) remove the duplicate values
# # print(df.shape) gives the shape of the dataframe
# # print(df.columns) gives the columns of the dataframe
# # print(df.index) gives the index of the dataframe
# # print(df.dtypes) gives the data types of the dataframe
# # print(df.head()) gives the top 5 rows of the dataframe
# # print(df.tail()) gives the last 5 rows of the dataframe



# # reading the csv file in pandas



# # making the in-memory file format from the data 
# # data=('col1,col2,col3\n''x,y,1\n''a,b,2\n','c,d,3\n')
# # data_str = ''.join(data)
# # print(type(data))
# # s=StringIO(data_str)
# # print(pd.read_csv(StringIO(data_str)))
# # read_csv needs a file  format so we convert the string to store into a file format and then read it

# # print(pd.read_csv(StringIO(data_str)),usecols=['A','B']) we can give column names here if we want to fetch only specific columns from the sheet
# # print(pd.read_csv(StringIO(data_str)),dtype={'A':int,'B':str}) we can give the data types of the columns here, this is done using dictonaries(colons)
# # print(pd.read_csv(StringIO(data_str))) print to check


# # data1= ('a,b,c\n','4,apple,bat\n','8,orange,cow')
# # data1_str=''.join(data1)
# # print(pd.read_csv(StringIO(data1_str)).values)

# # print(pd.read_csv('https://download.bls.gov/pub/time.series/cu/cu.item',sep='\t')) this is used to fetch the data from a paricular url and read it in form of tables 

# # working with JSON files
# # JSON is a lightweight data interchange format
# # JSON is used to store and transport data
# # JSON is used to store data in key value pairs
# # JSON is used to store data in arrays
# # JSON is used to store data in objects

# data3 = '{"employee_name": "James", "email": "james@gmail.com", "job_profile": [{"title1":"Team Lead", "title2":"Sr. Developer"}]}'
# print(type(data3))

# # we will convert this json into datafram to operate
# df1=pd.read_json(StringIO(data3))
# print(df1)
# # df1=pd.read_json(StringIO(data3),orient='records') above two are same column name will be there and below tha column name data will be displayed
# # print(df1)
# # df1=pd.read_json(StringIO(data3),orient='index') this changes the column name to row index and prints the data besides it
# # print(df1)
# # df1=pd.read_json(StringIO(data3),orient='split') 


# df4 = pd.DataFrame([['a', 'b'], ['c', 'd']], 
#                   index=['row 1', 'row 2'],
#                   columns=['col 1', 'col 2'],dtype='str')
# # remember the sequence of input for the dataframe
# # print(df4)

# # print(df4.to_json()) this will make the column as key and the row values as values
# # print(df4.to_json(orient='index'))  this will make the row values as the key and column values as their values

# # print(df4.to_json(orient='split')) gives the info of diffrent parameters

# # df5 = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
# # print(df5)
# # print(df5.to_json(orient='index'))

# data10 = [{"employee_name": "James", "email": "james@gmail.com", "job_profile": {"title1":"Team Lead", "title2":"Sr. Developer"}}]
# # Convert list of dictionaries into a JSON string
# data_str = json.dumps(data10)
# print(type(data10))
# df10=pd.read_json(StringIO(data_str))
# # print(df10)

# # normalize the data
# df10=pd.json_normalize(data10)
# # print(df10)


# # pickling and unpickling 
# # pickling is used to store the python objects
# # pickling is used to store the python objects in the form of byte stream
# # pickling is used to store the python objects in the form of binary format
# # pickling is used to store the python objects in the form of serialized format

# # stores in the formof 0s and 1s to store in the database 
 
# df=sns.load_dataset('tips')
# print(df.head)

# filename='file.pkl'
# ##serialize process
# print(pickle.dump(df,open(filename,'wb')))

# ##unsereliaze
# df=pickle.load(open(filename,'rb'))
# print(df.head)


# dic_example={'first_name':'Krish','last_name':'Naik'}
# pickle.dump(dic_example,open('test.pkl','wb'))


# little bit of OOPS, constructors necessary for modular programming
# class Car: 
#     pass

# car1= Car()
# car2= Car()
# car1.speed= 200
# car1.color= 'Black'
# car2.speed= 220
# car2.color= 'Red'
# print(car1.speed)
# print(car1.color)
# print(car2.speed)
# print(car2.color)

# constructor in python
# class Car:
#     def __init__(self,speed,color,engine):
#         self.speed=speed
#         self.color=color
#         self.engine=engine
    
#     def self_driving(self):
#         print("the car type is {}" .format(self.engine))

# car1= Car(200,'Black','petrol')
# car2= Car(220,'Red','diesel')
# print(car1.speed)
# print(car1.color)

# car1.self_driving()


# inheritance in python 
# inheritance is used to inherit the properties of the parent class
# inheritance is used to inherit the methods of the parent class
# inheritance is used to inherit the attributes of the parent class

# class Car:
#     def __init__(self,speed,color):  
#         self.speed=speed
#         self.color=color
#     def drive(self):
#         print("Driving")
# class BMW(Car):
#     def __init__(self,speed,color,make):
#         super().__init__(speed,color)
#         self.make=make
#     def display(self):
#         print("The car is {}" .format(self.make))
# car1= BMW(200,'Black','BMW') 


# car1.display()

# # python list comprehension
# # list comprehension is used to create lists in python
# # list comprehension is used to create lists in a single line
# # list comprehension is used to create lists with less code



# print(type([1,2,3,4,5,6,7]))  

# # list comprehension
# lst=[1,2,3,4,5,6]
# lst2=[]
# for i in lst:
#     lst2.append(i**2)
    
# print(lst2)

# numbers=[1,2,3,4,5]
# squard_numbers=[i**2 for i in numbers]
# print(squard_numbers)

# ## Filtering even numbers from a list:
# numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# even_number=[n for n in numbers if n%2==0]
# print(even_number)


# # Generating a list of the first letters of words in a list:
# words = ['apple', 'banana', 'cherry', 'date']
# first_letters=[word[0] for word in words]
# print(first_letters)

# numbers=[1,2,3,4,5,6,7,8,9,10]
# square=[n**2 for n in numbers if n%2==0]
# print(square)



# lambda function 
# lambda function is used to create an anonymous function
# lambda function is used to create a function in a single line
# lambda function is used to create a function with less code



# f=lambda x,y:x+y
# print(f(5,6))


# multiply= lambda x,y:x*y
# print(multiply(5,6))

# # lambda function with filter
# numbers=[1,2,3,4,5,6,7,8,9,10]
# even_numbers= list(filter(lambda x:x%2==0,numbers))

# print(even_numbers)







# numbers =[1,2,3,4,5,6]
# squares= list(map(lambda x:x**2,numbers))
# map function maps all the elements of the 2nd parameter and applies the function mentioned in the first parameter and then 
# we are typecasting it into list


# filter out even numbers from a list
# numbers=[1,2,3,4,5,6,7,8,9,10]
# print(list(filter(lambda x:x%2==0,numbers)))


# fruits = ['apple', 'banana', 'cherry', 'date', 'elderberry']
# sorted(fruits,key= lambda x:len(x))

# people = [
#     {'name': 'Alice', 'age': 25, 'occupation': 'Engineer'},
#     {'name': 'Bob', 'age': 30, 'occupation': 'Manager'},
#     {'name': 'Charlie', 'age': 22, 'occupation': 'Intern'},
#     {'name': 'Dave', 'age': 27, 'occupation': 'Designer'},
# ]

# sorted(people, key=lambda x:(x["age"]))
# # this will sort the dictionaries on the basis of age in ascending order


# data = {'a': 10, 'b': 20, 'c': 5, 'd': 15}
# max(data,key= lambda x:data[x])
# this will give the max of all the dictionaries 





# from sklearn.datasets import make_blobs # type: ignore
# from matplotlib import pyplot as plt 
# from matplotlib import style
# style.use('ggplot')

# X,y=make_blobs(n_samples=100,centers=3,cluster_std=1.0)

# plt.scatter(X[:,0],X[:,1],c=y)
# plt.show()



# Data Preprocessing in Python
# Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data preprocessing is a technique that is used to convert the raw data into a clean data set.

# import pandas as pd
# import scipy
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import seaborn as sns
# import matplotlib.pyplot as plt 

# # Load the dataset
# df = pd.read_csv('./diabetes.csv')
# print(df.head())
 
    
# df.info()
# # df.isnull().sum()
# print(df.describe())

# fig, axs = plt.subplots(9,1,dpi=95, figsize=(7,17))
# i = 0
# for col in df.columns:
#     axs[i].boxplot(df[col], vert=False)
#     axs[i].set_ylabel(col)
#     i+=1
# plt.show()


# # now form the boxplot visualization we have seen outliers which needs to removed
# # thus we remove them 
# q1, q3 = np.percentile(df['Insulin'], [25, 75])
# # Calculate the interquartile range
# iqr = q3 - q1
# # Calculate the lower and upper bounds
# lower_bound = q1 - (1.5 * iqr)
# upper_bound = q3 + (1.5 * iqr)
# # Drop the outliers
# clean_data = df[(df['Insulin'] >= lower_bound) 
#                 & (df['Insulin'] <= upper_bound)]
 
 
# # Identify the quartiles
# q1, q3 = np.percentile(clean_data['Pregnancies'], [25, 75])
# # Calculate the interquartile range
# iqr = q3 - q1
# # Calculate the lower and upper bounds
# lower_bound = q1 - (1.5 * iqr)
# upper_bound = q3 + (1.5 * iqr)
# # Drop the outliers
# clean_data = clean_data[(clean_data['Pregnancies'] >= lower_bound) 
#                         & (clean_data['Pregnancies'] <= upper_bound)]
 
 
# # Identify the quartiles
# q1, q3 = np.percentile(clean_data['Age'], [25, 75])
# # Calculate the interquartile range
# iqr = q3 - q1
# # Calculate the lower and upper bounds
# lower_bound = q1 - (1.5 * iqr)
# upper_bound = q3 + (1.5 * iqr)
# # Drop the outliers
# clean_data = clean_data[(clean_data['Age'] >= lower_bound) 
#                         & (clean_data['Age'] <= upper_bound)]
 
 
# # Identify the quartiles
# q1, q3 = np.percentile(clean_data['Glucose'], [25, 75])
# # Calculate the interquartile range
# iqr = q3 - q1
# # Calculate the lower and upper bounds
# lower_bound = q1 - (1.5 * iqr)
# upper_bound = q3 + (1.5 * iqr)
# # Drop the outliers
# clean_data = clean_data[(clean_data['Glucose'] >= lower_bound) 
#                         & (clean_data['Glucose'] <= upper_bound)]
 
 
# # Identify the quartiles
# q1, q3 = np.percentile(clean_data['BloodPressure'], [25, 75])
# # Calculate the interquartile range
# iqr = q3 - q1
# # Calculate the lower and upper bounds
# lower_bound = q1 - (0.75 * iqr)
# upper_bound = q3 + (0.75 * iqr)
# # Drop the outliers
# clean_data = clean_data[(clean_data['BloodPressure'] >= lower_bound) 
#                         & (clean_data['BloodPressure'] <= upper_bound)]
 
 
# # Identify the quartiles
# q1, q3 = np.percentile(clean_data['BMI'], [25, 75])
# # Calculate the interquartile range
# iqr = q3 - q1
# # Calculate the lower and upper bounds
# lower_bound = q1 - (1.5 * iqr)
# upper_bound = q3 + (1.5 * iqr)
# # Drop the outliers
# clean_data = clean_data[(clean_data['BMI'] >= lower_bound) 
#                         & (clean_data['BMI'] <= upper_bound)]
 
 
# # Identify the quartiles
# q1, q3 = np.percentile(clean_data['DiabetesPedigreeFunction'], [25, 75])
# # Calculate the interquartile range
# iqr = q3 - q1
# # Calculate the lower and upper bounds
# lower_bound = q1 - (1.5 * iqr)
# upper_bound = q3 + (1.5 * iqr)
 
# # Drop the outliers
# clean_data = clean_data[(clean_data['DiabetesPedigreeFunction'] >= lower_bound) 
#                         & (clean_data['DiabetesPedigreeFunction'] <= upper_bound)]

    
    
# corr = df.corr()
 
# plt.figure(dpi=130)
# sns.heatmap(df.corr(), annot=True, fmt= '.2f')
# plt.show()

# to check the proportionality we can also see the percentage of the people dibeteic and non-diabetic



# concept of tensor with examples
# tensor is a multi-dimensional array
# tensor is a generalization of matrices
# tensor is used in deep learning
# tensor is used in neural networks
# tensor is used in image processing



# [1,2] is a 1D tensor ndim gives 1 value but the vector is 2.
# [[1,2],[3,4]] is a 2D tensor 
# [[[1,2],[3,4]],[[5,6],[7,8]]] is a 3D tensor


# shape means no of elements in a particular axes (row & column)
# rank means no of axes in a tensor also equal to the number of dimensions to easily remember
# vector is 1D tensor but it has its own dimesnions
# matrix is 2D tensor







    
    




 












