from flask import Flask, request, render_template, flash
from flask_sqlalchemy import SQLAlchemy
import os

basedir = os.path.abspath(os.path.dirname(__file__))  # /home/maxdev/Documents/flaskExamples/pr1


app = Flask(__name__)
app.secret_key = 'apple_pie'
app.config['SQLALCHEMY_DATABASE_URI'] = \
    'sqlite:///' + os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    users = db.relationship('User', backref='role')

    def __repr__(self):
        return '<Role %r>' % self.name

class User(db.Model):
    __talblename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True)
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))

    def __repr__(self):
        return '<User %r>' % self.username


@app.route('/')
def index():
    flash('You are on the index page!')
    return render_template('index.html')


@app.route('/roledb')
def roledb_repr():
    return '<h1>%r</h1>' % roledb


@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)


@app.route('/agentinfo')
def agentinfo():
    user_agent = request.headers.get('User-Agent')
    return '<p>Your browser is %s</p>' % user_agent


@app.route('/statuscode')
def statuscode():
    return '<h1>Bad request</h1>', 403  # Forbidden


@app.route('/render/structure')
def render_structure():
    user = ''
    users = ['Ivan', 'Igor', 'Gennadiy', 'Olexey']
    return render_template('structexample.html', user=user, users=users)


# Custom error page
@app.errorhandler(404)
def page_not_found(e):
    return '<h1>Woops, page not found!</h1>'




if __name__ == '__main__':
    db.create_all()

    admin_role = Role(name='Admin')
    mod_role = Role(name='Moderator')
    user_role = Role(name='User')

    user_john = User(username='john', role=admin_role)
    user_susan = User(username='susan', role=mod_role)
    user_david = User(username='david', role=user_role)

    #db.session.add(admin_role)
    #db.session.add(mod_role)
    #db.session.add(user_role)
    #db.session.add(user_john)
    #db.session.add(user_susan)
    #db.session.add(user_david)

    # Or, more concisely:
    db.session.add_all(
        [
            admin_role, mod_role, user_role,
            user_john, user_susan, user_david
        ]
    )
    # Write all obects to the database
    db.session.commit()

    print(admin_role.id)
    print(mod_role.id)
    print(user_role.id)

    app.run()
    #manager.run()
