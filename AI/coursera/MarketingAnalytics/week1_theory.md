# week 1 theory

# Types of Marketing Analytics
- First calculate metrics, then do analytics!
- Marketing analytics is a process where you use data to make better marketing decisions.
- There are 3 different kinds of marketing analytics:
    - Descriptive
    - Predictive
    - Prescriptive
- Descriptive analytics
    - Looking at the past to see what happened.
    - Most commonly used type of analytics.
    - ad hoc reports and standard reports
        - We are looking at "what happened", "how long", "how often"
    - Looking at alerts, which say something abnormal is happening and what should we do to address 
    this abnormality == "What actions are needed?"
    - query/drill down "What exactly is the problem?"
    - ad hoc reports "How many, how often, where?"
    - standard reports "What happened?"
- Predictive Analytics
    - Uses forecasts what might happen.
    - Randomized testing, A/B testing.
    - Example: what will happen to the company, if we reduce the price of a product?
    - Example: insurance companices use data to predict the life expectancy of potential clients.
- Prescriptive Analytics
    - Answers "What is the best that can happen of all the options out there?"
    - "Assure customers that the salt content in chips product is at an acceptable level"
    - These are the most complex and least used type of marketing analytics
    - Solving optimization problems.

# Marketing process
- Default steps of marketing process:
    - Setting objectives (5C) - what is it that we are trying to accomplish:
        - customer
        - company: capability of the company
        - competitors
        - collaborators: suppliers
        - context
        - Example: "I want to address a customer paying point using a new product"
    - Strategy:
        - segmentation: divide customer base
        - targeting: which segment to focus on?
        - positioning: what is the Value proposition?
    - Tactics (4P):
        - product
        - price
        - place
            - channels and distributions
        - promotion
    - Financials (show me the money!):
        - gross margin
        - return on investments
        - customer lifetime value
- Airbnb marketing process
    - Objective: "How to improve customer experience under the context of sharing economy?"
        - customers: people around us, probably even me == everyone who is looking to travel
        - company: the website, which allows customers to look for the vacation rental places
        - competitors: hotels
        - collaborators: the renters
        - context: sharing economy that allows customers to transact with each other and share something that they have (in this case, apartment) == to be able to be rented by someone else using this portal
    - Strategy:
        - Segmentation:
            - by location: where people want to travel
            - what they look for: adventure / food / ...
            - by price: cheaper / higher OR price sensivity
            - by purpose (vacation / business)
            - composition of who is travelling: family / single person
            - students / working professionals
            ...
        - Targeting:
            - does Airbnb want to really focus on people "by location", or "by price"?
            - could be combining previously defined segments
        - Positioning:
            - if we chose to focus on families: Value proposition might be "home away from home"
    - Tactics
        - Product: certain room/house/apartment with bed, furniture etc
        - Price: price in USD (or local currency) on the product page
        - Place: google maps on the product page
        - Promotion: reviews in comment section
    - Financials:
        - mainly margin and ROI

# Airbnb's strategic challenge
- How does Airbnb make money?
    - guests: 6-12% resevration subtotal
    - 3% service fee
- Airbnb's strategic challenge: "How do we improve the rental prospects for our hosts and identify better rental options for our guests?"
    - How to use all of the user generated content on the website? (reviews, price etc)
    - Is there value in improving the pricing of properties?

# Airbnb's marketing strategy with data
- step 1: construct mental model
    - identify "Profit per property"
        - price
        - # of rentals
            - Star rating
            - Review
            - Property Attributes
        - minimum stay
        - gross margin (%)
- How are mental models used in marketing analytics?
    - As the first step in the analytics process, to outline the factors that influence the target metric
    - To get an intuitive sense of what is happening in the market
    - As the basis of a predictive model
    - As a hypothesis to test with data
- What data is necessary to test Airbnb's mental model?
    - Reviews (text + 5star rating)

# Text analytics
- Review sentiment: positiveness and negativeness of the review.
    - One way to get sentiment:
        - Calculate "good" and "bad" words.
        - Output: positive/negative number that identifies sentiment of the text.
- Text could contain errors and slang.

# Utilizing data to improve marketing strategy
- What is the rest of the data (apart from the Review Sentiment):
    - 1-5 star rating
    - price of the apartment, in USD
    - property attributes
- All that goes into a predictive model to predict the number of times a property is saved on Airbnb.
    - Predictive model could be regression model
- Example results:
    - Paris: people like low rent and high reviews
    - Miami: people like low rent, high reviews and low cleaning fees

# week 1
- Analytics provides marketing managers the opportunity to test intuition about marketing.
- Insights from analytics can challenge widely held assumptions.
- Confidence in decisions that flow out of marketing process is higher when it is informed by analytics.
