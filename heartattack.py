import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

st.write("""
# Heart attack detection: Dectect if someone has heart stroke using machine learning and python !
""")
image=Image.open('heartt.jpg')
st.image(image,caption='ML',use_column_width=True)

df = pd.read_csv('heart.csv')
st.subheader('Data information:')
st.dataframe(df)
st.write(df.describe())
corr_matrix=df.corr()
CR=corr_matrix["output"].sort_values(ascending=False)
st.subheader('Cross co-relation:')
st.write(CR)
#col=df.columns
#df= scaler.fit_transform(df)
#df=pd.DataFrame(df)
#df.columns=col
#st.write(df)
chart= st.bar_chart(df)
x= df.iloc[:,0:13].values
y= df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.3,shuffle=True)
#x_train ,x_test, y_train ,y_test= train_test_split(x,y, test_size=0.25,random_state=0)
def get_user_input():
    age = st.sidebar.slider('age',20,100,55)
    sex = st.sidebar.slider('sex',0,1,0)
    cp = st.sidebar.slider('cp', 0, 3 , 1)
    trtbps = st.sidebar.slider('trtbps',90.0 , 200.0 ,100.0)
    chol = st.sidebar.slider('chol',120.0 ,564.0 ,32.0)
    fbs = st.sidebar.slider('fbs',0 , 1 , 1)
    restecg = st.sidebar.slider('restecg' , 0 , 2 ,1)
    thalach = st.sidebar.slider('thalach' ,50.0 , 220.0, 100.0)
    exng = st.sidebar.slider('exng' ,0 ,1 , 0)
    oldpeak = st.sidebar.slider('oldpeak' ,0.0 ,6.5 ,1.0)
    slp = st.sidebar.slider('slp', 0, 2 , 1)
    caa = st.sidebar.slider('caa' , 0 , 4 ,2)
    thall = st.sidebar.slider('thall' , 0 , 3 ,2)

    ##Store a dictionary into avariable 
    user_data = {'age': age,
                    'sex': sex,
                    'cp': cp,
                    'trtbps': trtbps,
                    'chol': chol ,
                    'fbs': fbs ,
                    'restecg': restecg,
                    'thalach': thalach,
                    'exng': exng,
                    'oldpeak': oldpeak,
                    'slp': slp,
                    'caa': caa,
                    'thall': thall,}
    features= pd.DataFrame(user_data, index = [0])
    return features
user_input  = get_user_input()
st.subheader('User Input:')
st.write(user_input)
#coll=user_input.columns
#user_input= scaler.fit_transform(user_input)
#user_input=pd.DataFrame(user_input)
#user_input.columns=coll
#st.write(user_input)
RandomForstClassifier = RandomForestClassifier()
#RandomForstClassifier.fit(x_train , y_train)
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
regr = linear_model.LinearRegression()

regr.fit(x_train,y_train)
prediction = regr.predict(x_test)
st.subheader('Model Test Accuracy Score:')
st.write(str('Mean squared error: %.2f'% mean_squared_error(y_test, prediction)))
st.write(str('Coefficient of determination: %.2f'
      % r2_score(y_test,prediction)))
#accuracy_dt=accuracy_score(y_test,prediction)
#st.write(str("Accuracy score"))
predictionn = regr.predict(user_input)
st.subheader("Classification: ")
a=predictionn
if (a>0.5):
    st.write('Risk')
    #st.write(predictionn)
else:
    st.write('No Risk')
    #st.write(predictionn)
