from flask import Flask,render_template,request,session,redirect,flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from flask_mail import Mail
import os
from werkzeug.utils import secure_filename
import math
import pickle
import numpy as np
from joblib import dump,load
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

housing = pd.read_csv('data.csv')

from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)

# # Stratify Sampling

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    stratify_train_set = housing.loc[train_index]
    stratify_test_set = housing.loc[test_index]

housing = stratify_train_set.copy()

# # Handling Missing Values
from sklearn.impute import SimpleImputer

housing = stratify_train_set.drop("MEDV",axis=1)
housing_labels = stratify_train_set["MEDV"].copy()


# # Creating Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler()),
])
housing_num_tr = my_pipeline.fit_transform(housing)

app = Flask(__name__)
model = pickle.load(open('real_estates.pkl','rb'))

with open('static/config.json','r') as c:
    params= json.load(c)["params"]
    
# connection with database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/real_estates'
db = SQLAlchemy(app)
app.secret_key = 'super-secret-key'



#Creating Database MOdel (Table)


class Contact(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120),nullable=False)
    phone = db.Column(db.Integer, nullable=False)
    message=db.Column(db.String(120), nullable=False)
    location = db.Column(db.String(120), nullable=False)
    money=db.Column(db.String(120), nullable=False)
    date= db.Column(db.String(150), nullable=False)
    
class Prediction(db.Model):
    # CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDV

    sno = db.Column(db.Integer, primary_key=True)
    crim = db.Column(db.String(80), nullable=False)
    zn = db.Column(db.String(120),nullable=False)
    indus=db.Column(db.String(120), nullable=False)
    chas = db.Column(db.String(120),nullable=False)
    nox = db.Column(db.String(120), nullable=False)
    rm=db.Column(db.String(120), nullable=False)
    age= db.Column(db.String(150), nullable=False)
    dis = db.Column(db.String(80), nullable=False)
    rad = db.Column(db.String(120),nullable=False)
    tax=db.Column(db.String(120), nullable=False)
    ptratio = db.Column(db.String(120), nullable=False)
    b=db.Column(db.String(120), nullable=False)
    lstat= db.Column(db.String(150), nullable=False)
    medv = db.Column(db.String(80), nullable=False)
    

@app.route("/")
def home():
    return render_template("index.html", params=params)
  
  
  
@app.route("/about")
def about():
      return render_template("about.html", params=params)
  
  
@app.route("/services")
def services():
      return render_template("services.html", params=params)
  
  
  
@app.route("/dashboard",methods=['POST','GET'])
def dashboard():
    #if user already login
    if('user' in session and session['user']==params['user_name']):
        return render_template('dashboard.html',params=params)
    
    if(request.method=='POST'):
        username=request.form.get('uname')
        password=request.form.get('pass')
        if(username==params['user_name'] and password==params['user_password']):
            session['user']=username
            # posts = Posts.query.all()
            return render_template('dashboard.html',params=params)
            # return render_template('dashboard.html',params=params,posts = posts)
    return render_template('login.html',params=params)


@app.route('/customer')
def customer():
    customers = Contact.query.all();
    return render_template('customer.html',customers=customers,params=params)

 
@app.route('/predict_value')
def predict_data():
    return render_template('predict_value.html',params=params)

@app.route('/predictions')
def predictions():
    predictions = Prediction.query.all();
    return render_template('predictions.html',predictions=predictions,params=params)


@app.route('/predict', methods=['POST'])
def predict():
    # CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDV

    data1 = request.form.get('a',type=float)
    data2 = request.form.get('b',type=float)
    data3 = request.form.get('c',type=float)
    data4 = request.form.get('d',type=float)
    data5 = request.form.get('e',type=float)
    data6 = request.form.get('f',type=float)
    data7 = request.form.get('g',type=float)
    data8 = request.form.get('h',type=float)
    data9 = request.form.get('i',type=float)
    data10 = request.form.get('j',type=float)
    data11 = request.form.get('k',type=float)
    data12 = request.form.get('l',type=float)
    data13 = request.form.get('m',type=float)
    arr = np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13]])
    prepared_data = my_pipeline.transform(arr)
    pred = model.predict(prepared_data)
    pred = round(pred[0],3)
    entry = Prediction(crim=data1,zn=data2,indus=data3,chas=data4,nox=data5,rm=data6,age=data7,dis=data8,rad=data9,tax=data10,ptratio=data11,b=data12,lstat=data13,medv=pred);
    db.session.add(entry)
    db.session.commit()
    return render_template('after.html',data=pred,params=params)


@app.route('/logout')
def logout():
    session.pop('user')
    return redirect('/dashboard')
  
  
@app.route("/contact", methods=['GET','POST'])
def contact():
    if(request.method=='POST'):
        name=request.form.get('name')
        phone=request.form.get('phone')
        email=request.form.get('email')
        message=request.form.get('message')
        location=request.form.get('location')
        money=request.form.get('money')
        entry = Contact(name=name,phone= phone,email=email,message=message,location=location,money=money);
        db.session.add(entry)
        db.session.commit()
        # flash("Thanks for contact with us","success")
        # flash("we will soon contact with you","primary")
    return render_template("contact.html", params=params)

app.run(debug=True)