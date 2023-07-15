from django.shortcuts import render,HttpResponse,redirect
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from dia_2.models import Peopleinfo
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 


def view(request):
    return render(request, 'index.html')
def get_user_input_dia(request):
# Load and preprocess the diabetes dataset
    data2 = pd.read_csv('PIMA workbook Maths.csv')
# preprocess the dataset as needed
    columns_drop = ['Outcome', 'SkinThickness']
    x = np.array(data2.drop(columns_drop, axis=1))
    y = np.array(data2.Outcome)
# Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a logistic regression model
    LR = LogisticRegression()
    logreg=BaggingClassifier(estimator=LR,n_estimators=100,max_samples=0.8,oob_score=True,random_state=0)
    logreg.fit(x_train, y_train)

# Define a function to take user input

    if request.method == "POST":
        name = request.POST.get('name')
        preg = float(request.POST.get('preg', 0))
        glu = float(request.POST.get('glu', 0))
        if glu == 0:
            glu = float(data2['Glucose'].mean())
        bp = float(request.POST.get('bp', 0))
        if bp == 0:
            bp = float(data2['BloodPressure'].mean())
        insulin = float(request.POST.get('insulin', 0))
        if insulin == 0:
            insulin = float(data2['Insulin'].mean())
        height =(request.POST.get('height', '0'))
        weight = (request.POST.get('weight', '0'))

        dpf = float(request.POST.get('dpf', 0))
        if dpf == 0:
            dpf =float (data2['DiabetesPedigreeFunction'].mean())
        age = float(request.POST.get('age', 0))
        gender = float(request.POST.get('Gender', 0))
        
    height=float(height)
    weight=float(weight)
    # Return None if the request method is not POST
    bmi = weight / (height * height)
# Call your input function to get user input
    user_input =  np.array([[preg, glu, bp, insulin, bmi, dpf, gender, age]])
    prediction = logreg.predict(user_input)
    my_user=Peopleinfo(name=name,preg=preg,glu=glu,bp=bp,insulin=insulin,height=height,weight=weight,dpf=dpf,age=age,gender=gender,bmi=bmi)
    my_user.save()
    if prediction == [0]:
            result = "Good news! You are safe. Take care"
            is_affected = False
    else:
            result = "Oops! You may be diabetic. Take care"
            is_affected = True

    context = {
            'result' : result,
            'is_affected': is_affected
        }
    return render(request, 'disease_result.html', context)
    
    
    
def get_user_input_park(request):
# Load and preprocess the diabetes dataset
    data2 = pd.read_csv('parkinsons.csv')
# preprocess the dataset as needed
    X = data2.drop(columns=['name','status'], axis=1)
    Y = data2['status']
# Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=2)
    # Data standardization
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

# Train a  model
    clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, Y_train)


# Define a function to take user input

    if request.method == "POST":
        MDVP_Fo_Hz = float(request.POST.get('MDVP:Fo(Hz)', 0))
        MDVP_Fhi_Hz = float(request.POST.get('MDVP:Fhi(Hz)', 0))
        MDVP_Flo_Hz = float(request.POST.get('MDVP:Flo(Hz)', 0))
        MDVP_Jitter_percent = float(request.POST.get('MDVP:Jitter(%)', 0))
        MDVP_Jitter_Abs =(request.POST.get('MDVP:Jitter(Abs)', '0'))
        MDVP_RAP = (request.POST.get('MDVP:RAP', '0'))
        MDVP_PPQ = float(request.POST.get('MDVP:PPQ', 0))
        Jitter_DDP = float(request.POST.get('Jitter:DDP', 0))
        MDVP_Shimmer = float(request.POST.get('MDVP:Shimmer', 0))
        MDVP_Shimmer_dB = float(request.POST.get('MDVP:Shimmer(dB)', 0))
        Shimmer_APQ3 = float(request.POST.get('Shimmer:APQ3', 0))
        Shimmer_APQ5 = float(request.POST.get('Shimmer:APQ5', 0))
        MDVP_APQ = float(request.POST.get('MDVP:APQ', 0))
        Shimmer_DDA = float(request.POST.get('Shimmer:DDA', 0))
        NHR = float(request.POST.get('NHR', 0))
        HNR = float(request.POST.get('HNR', 0))
        RPDE = float(request.POST.get('RPDE', 0))
        DFA = float(request.POST.get('DFA', 0))
        spread1 = float(request.POST.get('spread1', 0))
        spread2 = float(request.POST.get('spread2', 0))
        D2 = float(request.POST.get('D2', 0))
        PPE = float(request.POST.get('PPE', 0))
        
 
# Call your input function to get user input
    user_input =  np.array([[MDVP_Fo_Hz,MDVP_Fhi_Hz,MDVP_Flo_Hz,MDVP_Jitter_percent,MDVP_Jitter_Abs,MDVP_RAP,MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB,Shimmer_APQ3,
                              Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
    prediction = clf.predict(user_input)
    if prediction == [0]:
            result = "Good news! You are safe. Take care"
            is_affected = False
    else:
            result = "Oops! You may be affected Parkinson. Take care"
            is_affected = True

    context = {
            'result' : result,
            'is_affected': is_affected
        }
    return render(request, 'disease_result.html', context)


def get_user_input_heart(request):
# Load and preprocess the diabetes dataset
    data = pd.read_csv('heart.csv')
# preprocess the dataset as needed
    X = np.array(data.drop(columns=['target'], axis=1))
    Y = np.array(data['target'])
# Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
    
    rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=100, n_jobs=-1, oob_score=False,
                       random_state=123, verbose=0, warm_start=False)
    
    rf.fit(X_train,Y_train)


# Define a function to take user input

    if request.method == "POST":
        age = float(request.POST.get('age', 0))
        sex = float(request.POST.get('sex', 0))
        cp = float(request.POST.get('cp', 0))
        trestbps = float(request.POST.get('trestbps', 0))
        chol =float(request.POST.get('chol', '0'))
        fbs = float(request.POST.get('fbs', '0'))
        restecg = float(request.POST.get('restecg', 0))
        thalach = float(request.POST.get('thalach', 0))
        exang = float(request.POST.get('exang', 0))
        oldpeak = float(request.POST.get('oldpeak', 0))
        slope = float(request.POST.get('slope', 0))
        ca = float(request.POST.get('ca', 0))
        thal = float(request.POST.get('thal', 0)) 
# Call your input function to get user input
    user_input =  np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                              exang, oldpeak, slope, ca, thal]])
    prediction = rf.predict(user_input)
    if prediction == [0]:
            result = "Good news! You are safe. Take care"
            is_affected = False
    else:
            result = "Oops! You may be affected Heart disease. Take care"
            is_affected = True

    context = {
            'result' : result,
            'is_affected': is_affected
        }
    return render(request, 'disease_result.html', context)

    



# Create your views here.
