# week 3 theory

# week 3 overview
- Delve into customer loyalty
- Explore the phases of a customers relationship life cycly with the firm
- Learn how to compute and apply the value of the relationshipp life cycle (CLV)
- evaluate marketing alternatives or levers like sales call or price promotions are effective in improving customer retention and lifetime value

# CLV: Customer Lifetime Value
- Customer retention - extremely important factor of the subscription model
    - == keep the customer for a longer time 
- Over time, any company as they grow would want:
    - RETENTION RATE of existing customers goes up;
    - ACQUISITION RATES going down;
    - Amount of new subscribers goes up;

- Customer Lifetime Value (CLV):
    - It computes the dollar value of an individual customer relationship
    - It is both backward looking and forward looking:
        - Computing value of past customers
        - Using that information to project forward

- For what is CLV used for?:
    - determine how much to spend to acquire a customer
    - determine how aggreessively to spend to retain a customer or a group of customers
    - sometimes: even value a company

# CLV: Netflix
- Mertics:
    - Expected customer lifetime (in months): 20
        - == how many months the customer will stay with Netflix
    - Average Gross Margin per Month per Customer (in USD): $50
    - Average Marketing Costs per Month per Customer (in USD): $0
    - Average Net Margin per Month per Customer (in USD): (2)-(3)=$50
    - Customer Lifetime Value (in USD): (1)*(4)=$1000
- Current conclusion of this $1000 number - Netflix SHOULD NOT spend more than $1000 for acquiring a new customer

- CLV IS DEFINED AS the discounted sum of all future customer revenue streams minus product and servicing costs and remarketing costs.

- Note: CLV is a good relative comparison tool, rather than an accurate predictive tool.

# Calculating CLV
- Assume that:
    - Net Marking per Netflix Customer = M-R=$50
        - M - gross marging
        - R - retention spending
    - Retention rate = r = 80%
    - Number of customers who joined Netflix in June 2014 = 100
- Build a table by month:
    - June 2014:
        - customers: 100 customers; net profit: (M-R)*100
    - July 2014:
        - customers: r * 100 = 80 customers; net profit: (M-R) * r * 100
    - August 2014:
        - customers: r * (r * 100); net profit: (M-R) * r * r * 100
    - September:
        - customers: r * (r * (r * 100)); net profit: (M-R) * r * r * r * 100

- The Base CLV Model:
    - $M - contribution per period from activate customers.
        - Contribution = SalesPrice - VariableCosts
    - $R - retention spending per period per active customer
    - r - retention rate
        - == fraction of current customers retained each period
    - d - discount rate per period
        - discount rate == the rate used to calculate the present value of future cash flows
        - Example: if I give you a $1 for a year  - I want $1.10.
            - 0.10 - money I couldve made if I invested in market or somewhere else
            - 0.10 == discount rate of 10%

- Present Value of net profit is extended up to infinity
    - reason: because there will always be a remainder of a customer when it comes to customer retention

- CLV formula:
    - CLV = ($M-$R) * ( (1+d) / (1+d-r) )

# Understanding the CLV Formula
- ($M-$R) - short-term recurring margin - margin you get each period
- ( (1+d) / (1+d-r)) - long-term multiplier - expected lifetime

- If retention rate increases:
    - discount rate stays the same
    - long-term multiplier goes up
    - CLV goes up

# Applying the CLV formula: Netflix
- Initial variables:
    - Charge: $19.95 per month
    - Variable costs: $1.50 per account per month
    - Marketing spending: $6 per year
    - Attribution: 0.5% per month
    - Monthly discound rate: 1%
- $M = 19.95 - 1.50 = 18.45
- $R = 6/12 = 0.5
- r = 0.995
- d = 0.01

- CLV = ($M - $R) * ( (1+d)/(1+d-r) )
- CLV = (18.45 - 0.5) * ( (1+0.1) / (1+0.1-0.995)) = 17.95 * 67.33 = 1.209

- Question: if Netflix cuts retention spending from $6 to $3 per year, they expect attribution will go up to 1% per month. Should they do it?
    - todo: recalculate CLV
        - is the new CLV is higher - do it! otherwise - don't
- New CLV:
    - retention spending: $3
    - attribution: 1%/month
    - new CLV = (18.45 - 0.25) * ( (1+0.01)/(1+0.01-0.99)) = 18.20 * 50.5 = $919
    - Conclusion: because before it was 1209 - Don't do that!

# Extending the CLV formula
- What is the time horizon? == how do we know, how far ahead to look?
- Ways to get money from the customer? == whether they first ask the money and then provide the service, or provide the service and ask for the money - that makes a difference in the CLV formula.

- Time Horizon
    - Build a table of "Percent of CLV accruing in first 5 years"
        - most of the CLV comes within the first 5 years
        - so it makes sense to consider just the first 5 years
    - Conclusions:
        - RETENTION rate GOES UP -> % of CLV accruings in the first 5 years DECREASES
        - DISCOUNT rate GOES UP -> % of CLV accruings in the first 5 years INCREASES

- Initial margin
    - there are 2 types of customers:
        - customers pay before using the service (Netflix, Hulu, apartment services (Airbnb))
            - use prev formula for CLV = ($M - $R) * ( (1+d)/(1+d-r) )
        - customers pay after using the service (Credit card)
            - new formula for CLV = ($M - $R) * ( r/(1+d-r) )
            - == so we basically remove first margin: ($M - $R) * ( (1+d)/(1+d-r) ) - m_first

- What about cohort and incubate?
- What about contractual vs noncontractual business models?

- Cohort and Incubate
    - Retention rate depends on time since customer acquisition - typically, retention rates start increasing and steadying over time
        - So the more time a customer stays with a firm, the longer their retention rate is going to be and over time it flattens out.

- You have to calculate CLVs among cohorts
    - cohort == customers acquired at the same time period (month, quarter, year)
    - Since retention changes with time since acquisition, CLV calculations are better if they are done separately for each cohort
        - if you mix customers - the retention on time graph is going to be different == incorrect

- CLV - contractual vs non-contractual
    - Contractual: Netflix has a contract with customers
        - So a customer has to call the firm and cancer his/her subscription
            - == companies KNOW when a customer unsubscribes to the service
            - This helps to estimating lifetime duration and retention rate
    - Non contractual: grocery stores, others
        - use empirical models, regression models - use historic data to predict expected retention rates.

# CLV application: make decisions in IBM
- context: allocate sales force for IBM's enterprise software customers in the mid-market segment
- evaluate: how profitable their custommers are.
- what was done:
    - divide all customers in 10 groups;
    - create several customer segments out of these 10 groups: super high clv, high clv, medium clv, low clv. 
- Result: they found out that ver-high-clv customers were never called, while low-clv customers were called despite sometimes doing a loss for IBM.
- By calling these very-high-clv cusomters and surveying them, IBM did 10x profits in the next yearss

# CLV: forward-looking measure
- Customer lifecycle: what are the metrics that identify that customers are going to be profitable in the future? :
    - backward-looking metrics:
        - share of wallet
        - past customer value
        - past period revenue
    - forward-looking metrics:
        - customer lifetime duration
        - customer lifetime value

- Marketing actions:
    - customer mindset: awareness, association => affect the brand equity
    - customer behavior: acquisition, retention => affect the customer lifetime value

- Link between brand equity and CLV:
    - Marketing actions (adverts, innovation, promotions, market presence, price) affect brand equity AND behavior
    - Brand equity (differentiation, relevance, esteen, knowledge) affects behavior
    - Behavior (acquisition, retention, profit contribution) affects CLV
