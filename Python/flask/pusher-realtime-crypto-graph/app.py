# App will get the prices for Bitcoin every 10 seconds.
# Pusher broadcasts an event, along with the new prices every time data is retrieved.

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from flask import Flask, render_template
from pusher import Pusher
import plotly, plotly.graph_objs as plotlygo
import requests, json, atexit, time

# Create Flask application.
app = Flask(__name__)

# Configure pusher object.
pusher_client = Pusher(
  app_id='',
  key='',
  secret='',
  cluster='',
  ssl=True
)

# Define variables for data retrieval.
times = []  # values of the time when we retrieve price data in a list.
currencies = ["BTC"]  # list of currencies we will be fetching data for.
prices = {"BTC": []}  # dictionary that holds the list of prices for currency defined.

@app.route("/")
def index():
    return render_template("index.html")

# Function to retrieve Bitcoin prices and then broadcast that data in graph and chart form.
def retrieve_data():
    # Create dict for saving current prices.
    current_prices = {}
    for currency in currencies:
        current_prices[currency] = []
    times.append(time.strftime('%H:%M:%S'))

    # Make request to API and get Bitcoin prices.
    api_url = "https://min-api.cryptocompare.com/data/pricemulti?fsyms={}&tsyms=USD".format(",".join(currencies))
    response = json.loads(requests.get(api_url).content)

    # Append new price to list of prices for graph and set current price for bar chart.
    for currency in currencies:
        price = response[currency]['USD']
        current_prices[currency] = price
        prices[currency].append(price)

    # Create an array of traces for graph data.
    graph_data = [
        plotlygo.Scatter(
            x=times,
            y=prices.get(currency),
            name="{0} Prices".format(currency)
        ) 
        for currency in currencies
    ]

    # Create an array of traces for bar chart data.
    bar_chart_data = [
        plotlygo.Bar(
            x=currencies,
            y=list(current_prices.values())
        )
    ]

    data = {
        'graph': json.dumps(list(graph_data), cls=plotly.utils.PlotlyJSONEncoder),
        'bar_chart': json.dumps(list(bar_chart_data), cls=plotly.utils.PlotlyJSONEncoder)
    }

    # Trigger event.
    # pusher_object.trigger('a_channel', 'an_event', {'some': 'data'})
    pusher_client.trigger('crypto', 'data-updated', data)


# Register the job for retrieving prices and run retrieve_data() function every 10 seconds.
scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(
    func=retrieve_data,
    trigger=IntervalTrigger(seconds=10),
    id='prices_retrieval_job',
    name='Retrieve prices every 10 seconds',
    replace_existing=True
)

# Shut down the scheduler when exiting the app.
atexit.register(lambda : scheduler.shutdown)

# Note: disabling auto reloader so as to prevent our scheduled function from running twice at every interval.
app.run(debug=True, use_reloader=False)
