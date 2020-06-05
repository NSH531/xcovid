from flask import Flask, flash, redirect  , request , render_template, url_for
from test import TestCovidNet
#from flask.ext.login import LoginManager
import flask_login 
app = Flask(__name__, template_folder='static')
app.secret_key='8d378ebec57b6805a493901ecc6ed926'
app.config['UPLOAD_FOLDER'] = 'u'
@app.route('/login1', methods=['POST'])
def login1():
    user = get_user(request.form['username'])

    if user.check_password(request.form['password']):
        login_user(user)
        app.logger.info('%s logged in successfully', user.username)
        return redirect(url_for('index'))
    else:
        app.logger.info('%s failed to log in', user.username)
        abort(401)

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Here we use a class of some kind to represent and validate our
    # client-side form data. For example, WTForms is a library that will
    # handle this for us, and we use a custom LoginForm to validate.
    form = flask_login.LoginForm()
    if form.validate_on_submit():
        # Login and validate the user.
        # user should be an instance of your `User` class
        login_user(user)

        flask.flash('Logged in successfully.')

        next = flask.request.args.get('next')
        # is_safe_url should check if the url is safe for redirects.
        # See http://flask.pocoo.org/snippets/62/ for an example.
        if not is_safe_url(next):
            return flask.abort(400)

        return flask.redirect(next or flask.url_for('index'))
    return flask.render_template('login.html', form=form)


@app.route('/xray')
def upload_file():
   return render_template('xray.html')
	
@app.route('/')
def test():
    # fix for "'thread._local' object has no attribute 'value'" bug
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True

    predictions = TestCovidNet.test()
    table=''
    for x in predictions:
        table=table+x+'    -     '
    return render_template ('covidnet.html',table=table,predictions=predictions)

if __name__ == "__main__":
#	from flask.ext.login import LoginManager
#	login_manager = LoginManager()
#	login_manager.init_app(app)

	from flaskext.mysql import MySQL
	mysql = MySQL()
	mysql.init_app(app)

	app.run(debug = False, host ='45.83.43.194', port = 5050)
