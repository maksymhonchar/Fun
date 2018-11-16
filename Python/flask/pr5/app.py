# Mailing service in Flask
# Example shows how to configure the application to send email through a Google Gmail account.

# If no configuration is given, Flask-Mail connects to localhost
# at port 25 and sends email without authentication.

from flask import Flask
from flask_mail import Mail, Message
import os

app = Flask(__name__)
mail = Mail(app)

# Mailing configuration
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')

print(os.environ.get('MAIL_USERNAME'))


def send_mail():
    msg = Message('test subject', sender=os.environ.get('MAIL_USERNAME'), recipients='maxgonchar8@gmail.com')
    msg.body = 'hello from app.py!'
    msg.html = '<b>HTML</b> body'
    with app.app_context():
        mail.send(msg)
        print(msg)


if __name__ == '__main__':
    app.run()
    send_mail()
