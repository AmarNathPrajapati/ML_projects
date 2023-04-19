from flask import Flask, render_template
app = Flask(__name__) # it return name if you are in python script

@app.route('/')
def home():
    return render_template('home.html')

if __name__=="__main__":
    app.run(debug=True)
#The above code is enough to render a html file