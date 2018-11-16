#!/usr/bin/env python3

from flask import Flask, render_template, redirect, url_for, session
from flask_wtf import Form
from wtforms.fields import RadioField, SubmitField, StringField
from wtforms.validators import DataRequired
from guess import Guess, GuessError

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'  # later generates a secret token.

game = Guess('Python')
game.expand('Python', 'C++', 'Is it interpreted?', False)
game.expand('C++', 'Java', 'Does it run on a VM?', True)


class YesNoQuestionForm(Form):
    answer = RadioField('Your answer', choices=[('yes', 'Yes'), ('no', 'No')])
    submit = SubmitField('Submit')


class LearnForm(Form):
    language = StringField('What language did you pick?',
                           validators=[DataRequired(message='Data is required in language field!')])
    question = StringField('What is a question that describes your language?',
                           validators=[DataRequired(message='Data is reqired in question field!')])
    answer = RadioField('What is the answer for your language?',
                        choices=[('yes', 'Yes'), ('no', 'No')])
    submit = SubmitField('Submit')


@app.route('/')
def index():
    session['question'] = 0
    return render_template('index.html')


@app.route('/question', methods=['GET', 'POST'])
def question():
    if 'question' not in session:
        return redirect(url_for('index'))
    id = session['question']
    question = game.get_question(id)
    if question is None:
        return redirect(url_for('guess'))
    form = YesNoQuestionForm()
    if form.validate_on_submit():
        session['question'] = game.answer_question(form.answer.data == 'yes', id)
        return redirect(url_for('question'))
    return render_template('question.html', question=question, form=form)


@app.route('/guess', methods=['GET', 'POST'])
def guess():
    if 'question' not in session:
        return redirect(url_for('index'))
    id = session['question']
    guess = game.get_guess(id)
    if guess is None:
        return redirect(url_for('index'))
    form = YesNoQuestionForm()
    if form.validate_on_submit():
        if form.answer.data == 'yes':
            return redirect(url_for('index'))
        return redirect(url_for('learn', id=id))
    return render_template('guess.html', guess=game.get_guess(id), form=form)


@app.route('/learn', methods=['GET', 'POST'])
def learn():
    if 'question' not in session:
        return redirect(url_for('index'))
    id = session['question']
    guess = game.get_guess(id)
    if guess is None:
        return redirect(url_for('index'))
    # 1. Create a form
    # 2. Validate on submit
    # 3. Inside the [if] - handling of the data
    # 4. Outside the [if] - rendering
    form = LearnForm()
    if form.validate_on_submit():
        game.expand(guess, form.language.data, form.question.data, form.answer.data == 'yes')
        return redirect(url_for('index'))
    return render_template('learn.html', guess=guess, form=form)


@app.errorhandler(GuessError)
@app.errorhandler(404)
def runtime_error(e):
    return render_template('error.html', error=str(e))


if __name__ == '__main__':
    # Simple running in debug mode:
    # app.run(debug=True)

    # To make application visible, use following:
    app.run(host='0.0.0.0', port=5000, debug=True)
    # Also to run on port 80, program needs a administrator permissions.
