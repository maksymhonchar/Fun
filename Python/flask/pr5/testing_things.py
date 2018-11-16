@app.route('/test/browser/')
def testbrowser():
    user_agent = request.headers.get('User-Agent')
    return '<p>Your browser is %s</p>' % user_agent
@app.route('/test/bad/')
def badrequest():
    return '<h1>Bad request. Check your console to view a response code! (400)</h1>', 400
@app.route('/test/cookie/')
def makeresponse():
    response = make_response('<h1>This document carries cookie!</h1>')
    response.set_cookie('answer', '42')
    return response
@app.route('/test/googleredirect/')
def googleredirect():
    return redirect('http://www.google.com')
@app.route('/test/abort404/<int:value>/')
def abort404(value):
    if value != 1:
        abort(404)  # Exception raising.
    return '<h1>You guessed it right! Now type smt else except of [1]</h1>'
