"""
The Connexion module allows a Python program to use the Swagger specification. This provides a lot of functionality: validation of input and output data to and from your API, an easy way to configure the API URL endpoints and the parameters expected, and a really nice UI interface to work with the created API and explore it.
"""

from flask import Flask, render_template
import connexion

# Create application instance.
# Internally, the Flask app is still created with additional functionality added to it.
app = connexion.App(__name__, specification_dir='./')

# Read the swagger.yml config file to setup API endpoints.
app.add_api('swagger.yml')

@app.route('/')
def home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

"""
http://127.0.0.1:5000/api/ui
http://127.0.0.1:5000/api/people
"""
