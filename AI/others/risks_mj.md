# Financial risks
- Some types:
    - market risk
    - credit risk
    - liquidity risk
    - operational risk
    - relative performance risk

- Risk?
- Situation 1: flipping a coin - there is nor risk. Just uncertainty.
    - Uncertainty != risk.
- Situation 2: flipping a coin, heads-paying a guy 100$, tails-earning 100$. Now there is a risk - I can potentially risk 100$ -- possibility of loss
- Situatioon 3: flipping a coin, heads-earn 200$, tails-pay 100$. Downside risk - losing 100$, upside risk - earning 200$.
    - Mistake - looking at risk for only downside situation.
    - Risk != Possibility of loss

- Risk is formed only if we take some actions.
    - Building a house of uncertain land = creating a risk

- Businesses take on risk in order to seek return.
    - Add value to society and seeks a reward for doing so

- Business <-> Risk

- Mistake: the bigger the risk, the bigger the return - that is not necessarily true!
    - You can optimise your exposure to the risk and maximize the returns.
    - There is some point where getting MORE risk DOES NOT gives you MORE return. 


# Market risk
- Market risk - risk of share price to go down
    - Note: it could be NOT a market risk. For example, it could be manager's fault or there was some operational flaw.
- Market risk is linked with Operational risk.

- Build models! Output of the model - Value at Risk (VaR)

- Problem of VaR - a little bit of artificial
    - Regulators tell "you can't have too much of risk"
    - Is not a very good measure

- There is also a MODEL RISK
    - it depends of how inflation has been modelled, how interest rates have been modeled etc
    - risk models give us a false confidence that we can understand the market risks.

- Risk Control Cycle (RCC):
    1. Identify what risks we have
    2. Measure those risks
    3. Manage the risk
    4. Monitor the risk
    5. Modify the risk
    6. -> 1

- Managing the risk:
    - transfering the risk away (f.e. to insurance company - pay $ for it!)
    - mitigating it
    - holding on to it (do that to earn money)

- Monitoring: what are the desirable features?
    1. up to date reporting
    2. as regular as possible
    3. automatic data update
    4. predictive - responsive to actions before they are made - ???
    5. regular reporting / intelligent reports
    6. Factors in the model should be good understood (i.e. understand how inflantion influences your model)
    7. Independent for each manager
    8. Output should be quantifiable - risk is VERY difficult to be quantifiable.

- Modify if things in the market / in the environment:
    - restructure your portfolio

- Strategy vs Tactics
    - Strategy - "Our strategy is to use tanks and soldiers to take over this town" - LOOK AT THE FOREST
    - Tactics - "Our tactics is to come in from 45 angle and aim at X tank" - LOOK AT THE TREES
- Asset Strategy vs Asset Tactics
    - Investment strategy: hold 30-50% property, 50-60% bonds, 10-20% stock
    - Tactics: which properties to buy, which bonds/equity to buy...

# Liquidity risk
- One of the most popular risks that the majority of businesses has to face.

- 3 Types of Liquidity Risk:
    1. Funding liqudity risk - credit crunch
        - when market lacks the supply of money - so that people cannot borrow
            - result of world recession - banks stop lending money
            - very much dependant of environment and economic climate
    2. Market liquidity risk
        - when market lacks capacity to handle the transaction
            - == marketability
        - example: when we sell a house; the problem - it takes a long time to sell house -> there is not enough capacity in market (not enough buyers)
        - example: bitcoin arbitrage
    3. Operational liquidity risk
        - not having enough cash to handle operations

- In your business, you have profits and expenses:
    - profits: selling, operating services etc
    - expenses: paying salaries, paying rent, disaster ...
- We have CASH INFLOWS and CASH OUTFLOWS

- Profitability: if inflows>outflows

- Liquidity risk:
    - look at total amount of cash in business as time flows by
    - look at how total amount of cash changes
    - Liquidity risk - IF OUR OUTFLOWS IS MORE THAN TOTAL CASH ON HAND over time
        - same as a car running out of gas - can't drive no more
- Solution:
    - some businesses keep cash on hand - in case of cash outflows
    - problem with keeping cash on hand - it weakens your returns
        - == money are not working for you

- Other strategies for dealing with liquidity risk:
    - Bank Overdraft
        - contracts to get emergency loans
    - Trade Agreements
- Problem with Bank Overdraft and Trade Agreements - they cost money
- Problem #2 - sometimess Bank Overdraft or Trade Agreements don't work (example: fin crisis)

- Scenario analysis
    - analyze probability, amount and other metrics for each of the out/inflows
    - "if we miss 2 inflows" - example of scenario

# Operational risk
- Internal fraud
- Mismanagement (computer system, people, thefts etc)

- Operational risks are DIFFICULT to identify, but EASY to manage

- Managing operational risks:
    - communication
    - chain of reporting
    - clear responsibilities and accountabilities for each person

- Main problem with Operational risk -> a lot of bureaucracy
- Also key personal are encouraged to leave

- Your task: balancing between "what defence mechanisms do we put in" and "are defence mechanisms aggravating factors to the other thigs?"

- Big part of managing operational risk - early warning systems

- What consultant should do:
    - identify
    - asses/evaluate
    - response & action

- Managing operational risk is about art and judgement, not pure math models

# Credit risk
- Credit risk != Default risk or Counterparty risk

- Counterparty risk:
    - whenever we relay on any counterparty and they fail to meet their obligations
        - supplier: fails to get the goods

- Default risk
    - we give some money to borrower and they fail to repay it
        - they default on their obligation

- Main terms in default risk:
    - probability of failing to repay
    - amount of loss
- By multiplying these 2 main termrs we get EXPECTED LOSS

- High probability of defaulting - they should be charged with higher interest
- Interesting thing - the higher interest we charge, the higher is the probability of the default

- Another type of Credit Risk: Late payment
    - It could be actually categorized as liquidity risk
- Another type of Credit Risk: change of credit spread
    - example: aaa rating -> aa rating
    - It could be categorized as market risk
- Systemic risk, contagion risk:
    - imagine supply chain 1->2->3; If 3 fails to repay 2 (defaults) -> this causes 1 to also have risk.

- Collateral
    - bank takes home/car if you fail to repay your dept
    - not so simple: a lot of hussle with laws
    - people could also lie about posessing house/car etc

# Credit risk management
- History: what procedure was earlier:
    - Defensive & Offensive teams
        - defensive team: the fewer defaults there are - the better
            - their incentive - have no defaults
            - this team catches all the "bad risk"
        - offensive team: the more loans they issue - the better
            - this team tells "why not taking a bigger loan?"
            - this strategy increases the risk of default

- Procedure that works TOO WELL:
    - Credit portfolio management:
        - lets say bank A is not allowed to rent >10**6$ to individual
            - reason - to maintain diversification - it is much safer to lend 5mln to 5 people rather to a single person
        - What if the best client wants 2*10**6 ?
        - The answer: bank A goes to bank B and give 1 mln of client from bank A in return of 1 mln from the client from the similar situation, but from the bank B

# Quantitative Credit Models
- Quantitative == look at financial data
- Main idea: get some sort of credit measure AND get probability of default

- Having this model we could answer:
    - should we transfer or hold this risk?
    - how much should we charge someone depending on probability of default?

- Fundamental flaws of these credit models:
    1. Lack of data
        - each credit situation is kind of unique
    2. Distributions non-normality
        - tails are way more fat AND skewed
    3. Correlations
        - if one defaults -> other might also default (because of inflation f.e.)

- Groups of quantitative CM:
    1. Credit Scoring Models
        - create measure from fundamentals and fin reports
        - problem: different industries are affected by different accounting ratios
    2. Structural model
        - starts with your SHARE PRICE and it's VOLATILITY
        - determine credit rating JUST USING Share price and Volatility
        - company would default if assets < liabilities
    3. Reduced form models
        - use statistical processes
        - check a general economical viewpoint and see how is the environment going around
    4. Credit migration model
        - see how the rating credit changes over time
        - uses Markow Model - calculate AAA->aa->a->bbb->default->bbb - simulate these processes
    5. Credit portfolio model
        - look at credit risk holistically (as the whole)
        - benefit: you can take into effect diversification

- Best approach: blend all of these models!
    - use quantitative AND qualitative models!

# Qualitative Credit Risk models
- Qualitative - very first models 

- Main idea: get the most information about the person BEFORE lending money
- Ask a bunch of questions to make sure our credit risk is we think t is

- Things to look at: Seniorty
    - senior dept - pay first
    - junior dept - pay next
    - unsecured dept (no collateral) - pay the least
- Collateral:
- Parental Guarantee
    - business is owned by government - it could pay its debts 
- Nature - what type of business the client does, what are they going to use money for
    - startup - high prob of a failure
    - utility, chocolate factory - good one to lend

- Economic indicators:
    - inflation rate
        - high inflation - higher interest rate
        - long-term loans - could be a fluctual interest rate

- Financial Ratio
    - find some interesting ratios between various parameters

- Face 2 Face meetings
    - gut feeling about the person (good shoes?!) ! 
    - is really subjective - BAD approach
