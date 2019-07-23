# week 4 theory

# week 4 overview
- AB testing - experiments
    - It allows to evaluate the consequences of marketer's different marketing actions like advertising or sales promotion.
- Correlation vs causation 
- How to define effective experiments to test the marketing hypothesis
- How to conduct basic experiments to assess the effectiveness of your marketing efforts
- It is challenging to evaluate marketing options because arketing usually DOES NOT occur in isolation.

# Determining Cause and Effect through Experiments
- Consider headline from a news article "Does skipping breakfast cause obesity?"
    - we can say, that people who skip breakfast are also obese - that there's a positive correlation
        - does that mean that the act of skipping breakfast for some reason causes obesity?
            - there are so many other reasons! example: people are ill or just don't do exercise
- You always need to dig deeper to understand if there is some causation (not correlation)
- Example of causation: poison ivy exposure and rash
    - Exposure to poison ivy causes a rash, but the rash itself is correlated with poison ivy (and many other things that cause a rash).
- Why is it important for marketing:
    - you want to know what is the effect of marketing on sales.
    - are marketing and sales correlated or does one cause the other?
    - can marketing cause sales and which part of marketing causes sales?
- This is exactly the reason WHY do marketers perform experiments - to reveal the causal effect of their marketing efforts on sales.
    - Experiments allow marketers to measure marketing return on investments, or the causal effect of their marketing campaigns. This is difficult to determine given the variety of factors that come into play between the time money is spent on advertising and when a sale is made.
- BASIC ISSUE: 
    - would you have achieved the same sales incresae WITHOUT the increased advertising spend
- Chief Marketing Officer needs to understand what is the causal effect on sales
    - Experiments ARE A WAY TO DETERMINE that causal effect.

# Designing basic experiments
- What establishes causality? 4 rules
    - Change in marketing mix produces change in sales:
        - increasing advertising budget => leads to increased sales
    - No sales increase when there is no change in the marketing mix:
        - no increase in advertising budget => same sales
    - Time Sequence:
        - increasing advertising today => higher sales tomorrow
    - No other external factor:
        - When advertising was increased, one of the competitors left the market. So sales increased because of lesser competition not because of increased advertising - WRONG

- Experiments:
    - we take one or more independent variable(s) (ex: advertising budget) are manipulated to observe changes in the dependent variable (Sales or Brand awareness)

- Origin of experiment - lab rats & pharmacy
    - give 1 group (test) - real drugs and the other group (control) - no pharmacy
    - observe the groups independently

- TEST GROUP - NEW DATA
- CONTROL GROUP - OLD DATA

- Construction of experiment: example:cereal
    - Choose 1,000 customers
        - test group: 500
        - control group: 500
    - Control group is exposed to OLD advertisement for one month
    - Test group exposed to advertisement highlighting new packaging for one month
    - Observe the results (sales):
        - test group: 1200 units
        - control group: 1000 units
    - Sales Lift: test - (minus) control = 200 units

# HOW to assign customers to Experiment and Control group?
- Answer - randomization
- With such an approach, we can match, test and control groups on all dimensions given you have a sufficient sample size.
- Example of randomization:
    - get all 1000 customers in 1 group
    - odd IDs: test group
    - even IDs: control group
- 1000 - magic number! If we have >1000 data points - we can achieve randomization

- ToNote factor: what are the pre-existing differences between the test and control group?
    - == are the test and control group different, even after randomization?

- Before-After design:
    - Choose 1000 customers
    - Divide into test and control group:
        - test group - 500
        - control group - 500
    - OLD ADVERTISEMENT (!):
        - test group: old ads
        - control group: old ads
    - Count the total sales:
        - test: 1100 sales
        - old: 1000 sales
    - NEW ADVERTISEMENT (!):
        - test group: new ads
        - control group: old ads
    - Count the total sales:
        - test group: 1200 units
        - control group: 1000 units
    - Sales Lift:
        - test-control = (1200-1000) - (1100 - 1000) = 100
- What does before-after design reveal:
    - there was a pre-existing differente of 100 units between the test and control group
        - conclusion: recalculate final sales lift regarding to this fact
- So before-after experiment takes into account DIFFERENCES between test and control groups

# Designing full factorial web experiments
- Why to choose web experiments:
    - cheap
    - fast to execute and change
    - you can change a lot of variables at the same time
- Example: manipulating PRIZE and AD COPY - FULL FACTORIAL DESIGN == FACTORIAL EXPERIMENT
    - price: now: $1.89
        - question: how sales change if price decreases to $1.59 or increases to $2.15?
    - Advertisement copy: now: "Good for you"
        - question: "Lasts longer", "Tastes Better" - how sales change
    - Having these variables, build a simple table to calculate EVERY CONBINATION of attributes

# Designing experiment: Etch-A-Sketch
- Question marketers pose to themselves: can they boost sales of Etch-A-Sketch through TV ads?
    - Awareness is high. What they really want is getting people to remember Etch-A-Sketch again and giving them a reason to go buy this product in the store.
    - One thing that marketers should be aware of is that their product is highly seasonal.

- What they did:
    - take Cincinnati as the TEST city.
    - take Charleston-South Carolina, Cleveland-Ohio and bunch of other cities as CONTROL cities
- Run new ads to TEST cities; Don't run new ads in CONTOL cities.

- PRE-EXISTING CONDITION
    - number of weeks: 12
    - in cincinnati - 162 units
    - all control units - 1526 units
    - cincinnati share- 9.6%
- RUNNING YOUTUBE ADS:
    - number of weeks: 3
    - cincinnati - 240 units
    - all control units - 1598
    - cincinnati shares - 13.1%

- Lift - 136.1%

- Let's say OTHER company didn't run ads - CONTROL PRODUCT
    - Etch-a-Sketch ran it, so it is a TEST product
- Let's say OTHER company share had 96.7%
- Net lift: 136.1-96.7 = 39.4%

- Reason to add OTHER company - It helped establish the fourth rule of causality by negating the influence of any external factor such as the impending holiday season.
    - Since no ad campaign was launched for OTHER COMPANY, it acted as the control product with no marketing, thus providing the baseline for the holiday season against which Etch a Sketch's sales could be compared.

- Question - does this lift make economic sense?
    - to do that, calculate some numbers for margins and retailer prices (see spreadsheet)

- retail price = $10
- retail margin = 36%
- manufacturer selling price = $6.4
- manufacturer contribution margin = 58% == $3.712
- national budget - $5,000,000
- units break even - 1,346,983 == how many etch-of-sketch they need to sell to recover 5,000,000 == 5*10**6 / 3.712
- base units = 3,100,000 == in the whole year, they sold so many units 
- base units TEST PERIOD = 1,085,000 == in the test period, they sold so many units

- The break even lift == the amount of lift necessary to make sense of the national investment in TV ads 
    - BREAK EVEN UNITS / BASE UNITS == 1.3M / 1.085M == 124%

- NET LIFT from TV ads = 39.4%

- conclusion - there is no sense in doing TV ads (39.4%). Do youtube ads instead (124%! 

# Analyzing an Experiment - Betty Spaghetty
- The practical reality - we have sales ONLY during the experiment, not only BEFORE an experiment (as was earlier)

- The ads feature 2 of the stock keeping units:
    - color crazy
    - gogo glam
- Test period: 17jun-17jul 2007

- Arizona was the test state
- Units sold:
    - color crazy: 1.8/week
    - gogoglam: 2.2/week

- California - control state
- Units sold:
    - Color crazy: 0.3/week
    - GoGoGlam: 1.2/week

- Lift: (1.8+2.2) / (0.3+1.2) = 2.(6) == 267%

- Betty test resluts:
    - ad budget: 3,000,000
    - retail selling price: 15
    - retail margin: 36%
    - manufacturer suggested price: 9.6
    - manufacturer contribution margin: 58% == 5.568

    - Break even units = 538,793 (3,000,000 / 5.568)

# Projecting Lift
- How much of lift can be expected in terms of units sold of BRAND_NAME from this TV ads

- Step 1: we are at Arizona and that's the 267% left (Arizona Test)
- Step 2: take from Arizona stores within the chain to the chain level of all the stores in Arizona. (Arizona Chain Sales)
- Step 3: go to the national level number of stores in the entire country in that chain (National Chain Sales)
- Step 4: all retail across all chains (All retail)
- Step 5: project the selling season

- Having all these steps at finished level, apply the numbers we have and see how much in unit sales can BRAND_NAME expect from a TV advertising campaign in December.

# Calculating projected lift: BettySpaghetty
- Control state: California
    - # of stores == control stores: 10% of chain stores
    - # sold: 1420
    - California % of national sales: 12%
    - national retailer sales - 11833 (1420/12)
    - retailer share: 25%
    - national units across all retailers sold: 47333 (11833/25%)
    - test% of annual sales: 5.50%
    - annual sales expected (without the TV ads - CONTROL STATE) - 860606 (47333/5.5%)

- So steps are:
    1. Calculate market share for test and control
    2. Calculate lift for test and control
    3. Calculate net lift
    4. Calculate manufacturer selling price
    5. Calculate contribution margin
    6. Calculate break even units
    7. Calculate break even lift as a % of the base
    8. Determine if it's worth it to run the campaign

- Test was run during June and July;
- The national campaign is going to run during the holiday season;
- What we have to do - take the annual sales # and look what % of annual sales is going to happen in holiday season.

- Holiday % of Annual sales: 45%
- Holiday units sold without ads: 387273 (860606 * 45%) == BASE SALES == SALES WITHOUT TV ADS

- Lift from ads: 267%
- Units after ads: 1034018 == 387273 * 267% == expected sales during the holiday season IF WE RUN A NATIONAL CAMPAIGN

- Compare expected sales with break even units:
    - to recap: break even units == 538793
    - expected units == 1034018
- Conclusion: should brand_name run campaign? 

# Pitfalls of Marketing Experiments
- Example: brand_name was supported with a $2M advertising campaign
- BUT
    - in holiday season - Hannah Montana started her career
- Result: sales not as expected so brand_name was upstaged by Hannah Montana brand
- Because experiments didn't know about Hannah Montanta => Advertising campaign failed.

# Amazon Goldbox Promotion: advantages
- change in realtime your campaign
- you could pull out of the campaign if you see it is not going well
- pays dividends in front term - because amazon uses specific algorithms
- multiple digital experiments can be run at the same time
- the cost of digital experiments is variable

# Takeaways
- Experiments are important because they assess cause and effect
- Pay attention to:
    - design
    - gap between test results and field implementation
    - difference bewteen test and campaign contexts
- Web experiments are cheaper and faster
    - costs of experiments can be variable rather than fixed
- Experiments provide forecasts of expected ROI
    - this can help with determining campaign budgets that are the best to maximize sales
