from flask import Flask  # flask server
from flask import request  # how the user requested a resource
from flask import render_template  # to use templates/layouts

app = Flask(__name__)

# Something i can do in templates:
#  {{ var_name }}
#  {% """kind of python code on flask""" %}

# [@] signifies a decorator - way to wrap a function and modifying its behavior.
# In Flask, we are mapping url to return value.
@app.route('/')
def index():
    return 'This is the homepage'
@app.route('/about')
def about():
    return 'This is the about page.<br />brrr'


# Variables in routing. google query: flask converters
# Variable in URL - in brackets
# Example1 - strings. Strings are default type.
@app.route('/profile/<username>')
def profile(username):
    return 'Hello, %s' % username
# Example2 - integers
# In case of exceptions (if post_id is string) 404 page is already implemented.
@app.route('/post/<int:post_id>')
def post(post_id):
    return 'post_id is %d' % post_id


# HTTP methods. Here: GET and POST methods
# Example of simple GET method
@app.route('/method')
def check_method():
    return 'Method used: %s' % request.method
# Example of different methods
@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        return 'You are using POST method'
    elif request.method == 'GET':
        return 'You are using GET'
    return 'You are using something else...'


# HTML Templates
# [templates] and [static] folders are necessary
@app.route('/tmp/<name>')
def tmp_name(name):
    return render_template("profile.html", name=name)

# Mapping multiple URLs
@app.route('/mult/')
@app.route('/mult/<user>')
def mult(user=None):
    return render_template("user.html", user=user)

# Passing lists to the template
@app.route('/shopping')
def shopping():
    food = ["cheese", "eggs", "ham"]
    return render_template('shopping.html', food=food)


if __name__ == '__main__':
    app.run(debug=True)
