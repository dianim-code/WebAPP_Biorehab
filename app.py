import os
from datetime import datetime
from cs50 import SQL
from flask import Flask, flash, jsonify, redirect, url_for
from flask import render_template, request, session
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash
from helpers import apology, login_required, allowed_file
from werkzeug.utils import secure_filename
from pathlib import Path
import webbrowser
import matplotlib.pyplot as plt
import pandas
import numpy
import jinja2
import io
from random import seed
from random import randint, choice
from flask_mail import Mail, Message
from flask import render_template
from itsdangerous import URLSafeTimedSerializer

# seed random number generator
seed(1)

from SIIR_Q_Learning_v4 import siirqt
from SIIR_Q_Test_v4 import siirqr
from covid_19_dqn_simple_stratified_with_vaccine_v3 import dqn

plt.style.use('ggplot')


# -------------------------------------------- Configure application --------------------------------------------
app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = '/static'

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
os.environ['EMAIL_USER'] ='infowebappg5@gmail.com'
os.environ['EMAIL_PASSWORD'] = 'webapp93@@'

mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    "MAIL_USERNAME": os.environ['EMAIL_USER'],
    "MAIL_PASSWORD": os.environ['EMAIL_PASSWORD']
}
app.config.update(mail_settings)
mail = Mail(app)
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///users.db")

# -------------------------------------------- ROUTES --------------------------------------------
# -------------------------------------------- HOME --------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")

# -------------------------------------------- SIIR Q --------------------------------------------
# ---------------------- SIIR Q TRAIN ----------------------
@app.route("/siirqt", methods=["GET", "POST"])
# ROUTE NAME/FUNCION: Run SIIR Model Q Learning
@login_required
def siirqtrainfun():

    # SIIRQT PATH GENERATION
    my_path = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(my_path,'static\\images\\SIIRQT')

    if request.method == "POST":
        #GET VARIABLES FROM SLIDERS
        episodes = int(request.form["siirqtamountRange1"])*1000
        beta1 = float(request.form["siirqtamountRange2"])/100
        beta2 = float(request.form["siirqtamountRange3"])/100

        # CHECK FROM ZERO START
        if request.form.get("checksiirqt") == "fromZero":
            from0 = True
        else:
            from0 = False

        # PROGRAM RUN
        siirqt(from0,episodes,beta1,beta2)
        flash("Elaborazione Conclusa")
        return render_template("siirq_get.html")

    else:
        return render_template("siirq_get.html")

# ---------------------- SIIR Q TEST ----------------------
@app.route("/siirqr", methods=["GET", "POST"])
# ROUTE NAME/FUNCION: Run SIIR Model Q Learning
@login_required
def siirqrunf():

    # SIIRQR PATH GENERATION
    my_path = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(my_path,'static\\images\\SIIRQR')

    if request.method == "POST":
        #GET VARIABLES FROM SLIDERS
        episodes = int(request.form["siirqramountRange1"])
        beta1 = float(request.form["siirqramountRange2"])/1000
        beta2 = float(request.form["siirqramountRange3"])/1000

        newTable = False

        # PROGRAM RUN
        siirqr(newTable,episodes,beta1,beta2)
        flash("Elaborazione Conclusa")
        return render_template("siirq_get.html")

    else:
        return render_template("siirq_get.html")

# ---------------------- SIIR Q SHOW ----------------------
@app.route("/siirqs", methods=["GET", "POST"])
# ROUTE NAME/FUNCION: Show SIIR Model Q Learning
@login_required
def siirqshowfun():

    # SIIR 2DIR PATH GENERATION
    my_path = os.path.dirname(os.path.abspath(__file__))
    images_dir_r = os.path.join(my_path,'static\\images\\SIIRQR')
    images_dir_t = os.path.join(my_path,'static\\images\\SIIRQT')

    if request.method == "POST":
        # CHOOSE DIR
        radio = request.form["siirqtable"]
        if radio == "new":
            path_to_dir = images_dir_t
            dir_chosen = 'SIIRQT'
        else:
            path_to_dir = images_dir_r
            dir_chosen = 'SIIRQR'

        # GET ELEMENT FROM DIR
        entries = os.listdir(path_to_dir)

        names = []

        # BUILD PATH FOR EACH ELEMENT
        for file in entries:
            names.append(os.path.join('..\\static\\images\\',dir_chosen,file))

        # SHOW ONLY 10 EPISOEDS
        names2 = []
        if len(names) > 10:
            for _ in range(10):
                names2.append(choice(names))
                names = names2

        return render_template("siirq.html", names = names)
    else:
        return render_template("siirq_get.html")       


# -------------------------------------------- dqnv2 --------------------------------------------
# ---------------------- dqnv2 RUN ----------------------
@app.route("/dqnv2r", methods=["GET", "POST"])
@login_required
def dqnv2r():

    # PATH GENERATION
    my_path = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(my_path,'static\\images\\Simple Stratified 3 Layers Vaccine')

    if request.method == "POST":

        #GET VARIABLES FROM SLIDERS
        weight_socioecon = int(request.form["dqnv2ramountRange1"])/100
        weight_infected = int(request.form["dqnv2ramountRange2"])/100
        weight_time = int(request.form["dqnv2ramountRange3"])/100

        # VACCINE YES/NO
        if int(request.form["dqnv2ramountRange4"]) == 1:
            Vaccine = True
        else:
            Vaccine = False

        # NEW NET YES/NO
        if request.form["dqnv2rtable"] == "new":
            preTrained = False
        else:
            preTrained = True
        
        # ONLY EXPLOIT YES/NO
        if request.form["onlyExploit"] == "Only":
            onlyExploit = True
        else:
            onlyExploit = False

        dqn(weight_socioecon,weight_infected,weight_time,Vaccine,preTrained,onlyExploit)
        flash("Elaborazione Conclusa")
        return render_template("dqnv2_get.html")

    else:
        return render_template("dqnv2_get.html")

# ---------------------- dqnv2 SHOW ----------------------
@app.route("/dqnv2s", methods=["GET", "POST"])
@login_required
def dqnv2s():

    my_path = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(my_path,'static\\images\\Simple Stratified 3 Layers Vaccine')
    dir_chosen = 'Simple Stratified 3 Layers Vaccine'

    if request.method == "POST":

        if request.form.get("Susceptible") == "suc":
            suc_checked = True
        else:
            suc_checked = False

        if request.form.get("Infected") == "inf":
            inf_checked = True
        else:
            inf_checked = False

        if request.form.get("Recovered") == "rec":
            rec_checked = True
        else:
            rec_checked = False

        if request.form.get("Hospitalized") == "hos":
            hos_checked = True
        else:
            hos_checked = False

        if request.form.get("Death") == "dea":
            dea_checked = True
        else:
            dea_checked = False

        entries = os.listdir(images_dir)
        names = []

        # SHOW ONLY REQUIRED PLOT
        for file in entries:

            # CHECK SUC
            if 'suc.png' in file and suc_checked:
                names.append(os.path.join('..\\static\\images\\',dir_chosen,file))

            # CHECK INF
            if 'inf.png' in file and inf_checked:
                names.append(os.path.join('..\\static\\images\\',dir_chosen,file))

            # CHECK HOS
            if 'hos.png' in file and hos_checked:
                names.append(os.path.join('..\\static\\images\\',dir_chosen,file))

            # CHECK REC
            if 'rec.png' in file and rec_checked:
                names.append(os.path.join('..\\static\\images\\',dir_chosen,file))

            # CHECK DEA
            if 'dea.png' in file and dea_checked:
                names.append(os.path.join('..\\static\\images\\',dir_chosen,file))
    
        # SHOW ONLY 10 EPISOEDS
        #names2 = []
        #if len(names) > 10:
        #    for _ in range(10):
        #        names2.append(choice(names))
        #        names = names2

        return render_template("dqnv2.html", names = names)
    else:
        return render_template("dqnv2_get.html")

# -------------------------------------------- prod --------------------------------------------  
@app.route("/prod", methods=["GET", "POST"])
def prod():

    if request.method == "POST":
        return render_template("prod.html")

    else:
        return render_template("prod.html")

# -------------------------------------------- mission -------------------------------------------- 
@app.route("/mission", methods=["GET", "POST"])
def mission():

    if request.method == "POST":
        return render_template("mission.html")

    else:
        return render_template("mission.html")

# -------------------------------------------- who --------------------------------------------
@app.route("/who", methods=["GET", "POST"])
def who():

    if request.method == "POST":
        return render_template("who.html")

    else:
        return render_template("who.html")

# -------------------------------------------- where --------------------------------------------   
@app.route("/where", methods=["GET", "POST"])
def where():

    if request.method == "POST":
        return render_template("where.html")

    else:
        return render_template("where.html")

# -------------------------------------------- Contact --------------------------------------------  
@app.route("/contact", methods=["GET", "POST"])
def contact():

    if request.method == "POST":
        mailAdd = request.form["mail"]
        mailObj = request.form["obj"]
        mailBody = request.form["body"]

        with app.app_context():
            msg = Message(subject=mailObj,
                        sender='infowebappg5@gmail.com',
                        recipients=[mailAdd], # replace with your email for testing
                        body=mailBody)
            mail.send(msg)
        flash("Sent")
        return render_template("contact.html")

    else:
        return render_template("contact.html")

# -------------------------------------------- final --------------------------------------------
@app.route("/final", methods=["GET", "POST"])
def final():

    if request.method == "POST":
        return render_template("final.html")

    else:
        return render_template("final.html")

# -------------------------------------------- thanks -------------------------------------------- 
@app.route("/thanks", methods=["GET", "POST"])
def thanks():

    if request.method == "POST":
        return render_template("thanks.html")

    else:
        return render_template("thanks.html")

# -------------------------------------------- CHANGE PASSWORD --------------------------------------------
@app.route("/changepsw", methods=["GET", "POST"])
@login_required
def changepsw():
    if request.method == "POST":
        # Ensure USERNAME was submitted
        if not request.form.get("passwordOld"):
            return apology("must provide old password", 408)

        # Ensure NEW PASSWORD was submitted
        if not request.form.get("passwordNew"):
            return apology("must provide new password", 408)
        
        # Ensure NEW PASSWORD CONFIRMATION was submitted
        if not request.form.get("passwordNewCon"):
            return apology("must provide new password", 408)
        
        # Ensure PASSWORD is match
        if request.form.get("passwordNew") != request.form.get("passwordNewCon"):
            return apology("Passwords don't match", 408)

        # Query database for username
        rows = db.execute("SELECT * FROM users WHERE id = :id",
                            id = session["user_id"])

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("passwordOld")):
            return apology("invalid username and/or password", 408)

        db.execute("UPDATE users SET hash = :newHash WHERE id = :id",
                    newHash = generate_password_hash(request.form.get("passwordNewCon")),
                    id = session["user_id"])
        flash("Password Changed")
        return redirect("/")
    else:
        return render_template("changepsw.html")

# -------------------------------------------- LOGIN --------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 403)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 403)

        # Query database for username
        rows = db.execute("SELECT * FROM users WHERE username = :username",
                            username=request.form.get("username"))

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
            return apology("invalid username and/or password", 403)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")

# -------------------------------------------- LOG OUT --------------------------------------------
@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")

# -------------------------------------------- REGISTER --------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    if request.method == "POST":
        # Check if the username is not null
        if not request.form.get("usernameReg"):
            return apology("You must provide a valid username", 401)

        # Check if the password is not null
        if not request.form.get("passwordReg"):
            return apology("You must provide a valid password", 401)

        # Check if the password match
        if request.form.get("passwordReg") != request.form.get("passwordRegCon"):
            return apology("Passwords don't match", 401)

        rows = db.execute("SELECT * FROM users WHERE username = :username",
                            username=request.form.get("usernameReg"))
        if len(rows) != 0:
            return apology("Username Already taken", 401)

        # Check if the username is already taken
        # snippet taken from login route
        # u2i = username to insert
        # p2i = password to insert
        key2pass = db.execute("INSERT INTO users (username, hash) VALUES (:username, :hash)",
                                username = request.form.get("usernameReg"),
                                hash = generate_password_hash(request.form.get("passwordReg")))

        # check for username
        if key2pass == None:
            return apology("Generic registration error. Try Later!", 401)
        session["user_id"] = key2pass
        # Redirect user to home page
        return redirect("/")
    # same order of action as login route
    # different html output
    else:
        return render_template("register.html")

def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return apology(e.name, e.code)

# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)
