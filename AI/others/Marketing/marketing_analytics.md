# src: https://towardsdatascience.com/a-day-in-the-life-of-a-marketing-analytics-professional-83dd45f2e702
- digital analytics — tracking visits, clicks and conversions
- marketing analytics IS NOT Market Research
- Job can vary from light SQL coding through to full-blown machine-learning algorithms
- Tasks:
        - heavy SQL queries
        - Cluster analysis (ML technique to help me find ‘natural’ groupings of users)
                - I’m looking for below-average users of a product for an education-based
                marketing campaign. 
                - We talk about potential audiences, Key Performance Indicators (KPIs),
                strategies and budget. 
                - Reviewing Dashboards and Trends
                        - demographics
                        - marketing performance
                        - regional data segmentation
                        - marketing benchmarks
                - Get Campaign Results Analysis.
                        - pull up the media impression files and start the analysis.
                        - Almost all of our marketing is measured using a test and
                        control format — test audiences receive marketing while a similar
                        control audience receives no marketing.
                                -  goal is to see if there is a statistically
                                significance difference between the groups on product
                                behavior metrics
                - Predicting ROI (return on investment)
                        - The goal is to figure out if the campaign is worth 
                        the investment.
                        - I look at metrics such as potential reach, estimated
                        click-through rate and approximate CPAs (cost-per-acquisition).
                        -  use benchmarks and multivariate regression to run the
                        prediction.
                - Geographic Mapping of Activity
                        - Example: They ask for locations that have lot of lunchtime
                        foot traffic -- I build a heat map
                - write the measurement plan portion of a marketing campaign
                        - All campaign plans are required to include a measurement
                        section that outlines KPIs, secondary metrics, goals, target
                        audiences, geographies and measurement approach
- Product Analysts are responsible for
        - understanding deep trends and nuances in a specific product area
        - build and maintain key data tables for their product area.

# src: https://www.r-bloggers.com/marketing-analytics-and-data-science/
- Kaggle: Google Analytics Customer Revenue Prediction competition
- The problem outline:
        - The 80/20 rule has proven true for many businesses – only a small percentage
        of customers produce most of the revenue. As such, marketing teams are
        challenged to make appropriate investments in promotional strategies.
        - The Goal: Predict the revenue of a customer
- Step 1: understanding the Data (Exploratory data analysis (EDA))
        - This step will provide insight into a data set, uncover the underlying
        structure and extract important variables.
        - Data consists of approximately 1.7 million records (observations)
        - Each observation contains information, such as the number of times a user
        visited the store, the date/time of visit, the visitor’s country, etc. 
        - The data does not contain personal identifying information, with all persons
        identified by a visitor ID.
        - Missing data: 
                - There are over 900,000 observations in the training dataset,
                with each observation representing a store visit.
                - The Takeaway: There are 14 variables in the training dataset
                containing missing values. Most of these variables are related to Google
                AdWords campaigns, we surmise the missing transaction revenue data means
                there was no transaction.
        - Understanding Transaction Revenue
                - Only 11,515 web sessions resulted in a transaction, which is 1.27% of
                the total observations.
                - The Takeaway: Only a small percent of all the data resulted in a
                transaction, with almost all of the transactions resulting in less than
                $1000.
        - Sessions and Revenues by Date
                - There are some outliers with high daily revenues. Notice the revenue
                spikes marked in yellow.
                - Daily Revenues vary between $0 and approximately $27,000 and that
                there is no revenues trend visible.
                - The Takeaway: I should consider revenue anomalies to accurately
                predict customer revenue.
        - Sessions and Revenues by Workday
                - The Takeaway: The weekend website sessions and revenue are lower than
                the weekdays.
        - Sessions and Revenues by Month
                - The Takeaway: Looking at the month patterns, April has a high ratio of
                revenues/sessions, and November has a low ratio of revenues/sessions.
                This is something to keep track of.
        - Channel Grouping
                - Channels define how people come to your website. 
                - Google Analytics features were used here
                - The Takeaway: Organic Search and Social have a high number of
                sessions, with social producing very low revenue. Referral produces the
                most revenue, with a relatively small number of sessions.
        - Pageviews; All Sessions
                - A pageview indicates each time a visitor views a page on your website,
                regardless of how many hits are generated.
                - Most revenue relates to a low number of pageviews
                - The Takeaway: The distribution of pageviews is skewed. However, the
                second graph shows that most revenues came from a small number of
                sessions with a lot of pageviews! It may be that pageviews is a good
                factor for predicting revenue.
- Step 2: Data Preparation
        - Data preparation is the process of formatting the data for people
        (communication) and machines (analyzing/modeling), and perform correlation
        analysis to review and select the best features for modeling.
        - Correlation Analysis
                - The correlation analysis is important because it gives insight into
                which variables have a good chance of helping the model. Above,
                pageviews and hits are two highly correlated (good for predicting)
                variables.
        - Hypothesis: Important Variables
                - Hypothesis 1: The time of day is an indicator of how much a person
                spends. To test this hypothesis, I created a new variable, hour of day.
                I extracted the hour of day from the store visitor’s session time.
                - Hypothesis 2: Visitor behavior is an indicator of how much a person
                spends. The existing data contains each visitor’s number of hits,
                pageviews, and visits. Along with the individual variables, I created a
                sum of these and combined them as visitor behavior.
        - Perform an additional correlation analysis AFTER ADDING the new variables from
        Hypothesis 1 and Hypothesis 2
        - Additional hypotheses to be tested (not in the current model):
                - New site visits is an indicator of how much a person spends.
                - A time-series approach will produce a more accurate model.
                - Address holidays and big sales days (such as Black Friday) to increase
                revenue prediction. The data does contain spikes in revenue. It would be
                interesting to see if there is an annual trend in revenue spikes and if
                there is a geographic trend with revenue spikes.
- Step 3: Modeling
        - Deciding which variables that go into modeling will take into account the EDA
        takeaways, the correlation analysis, and my own hypotheses
        - The top-performing generated machine learning model was a distributed random
        forest.
- Step 4: Evaluation:
        - With the baseline model now complete, I will submit the model to the Kaggle
        competition for results.
- Step 5: Impact – Business Benefit
        - With the machine learning model complete, the Google Merchandise Store can
        identify customers that generate the most revenue.
        - Below are the business benefits to identifying high-value customers:
                - Reduce customer churn with retention marketing for high-value customers
                - Identify likely high-value customers by analyzing existing high-value
                customer purchasing behavior
                - Predict revenue based on high-value customer purchasing history
                - Focus product development on high-value customer products

# https://eng.lyft.com/lyft-marketing-automation-b43b7b7537cc
- Project: build a marketing automation platform to improve cost and volume efficiency while enabling our marketing team to run more complex, high-impact experiments.
- Requirements of the project:
	- Ability to predict the likelihood of a new user to engage with our product
	- Measurement mechanisms to allocate our marketing budgets across different
	internal and external channels
	- Levers to deploy these budgets across thousands of ad campaigns
- Examples of problems we needed to automate:
        - Updating bids across thousands of search keywords.
        - Turning off poor-performing display creative.
        - Changing referrals values by market.
        - Identifying high-value user segments.
        - Sharing learnings from different strategies across campaigns.
- Features examples:
        - Driver:
                - Region
                - Channel
                - Device
                - Cohort date
                - Funnel Velocity
                - Car Metadata
        - Rider Features:
                - Region
                - Channel
                - Device
                - Cohort date
                - user VALUE since application activation
                ...
- With an app, bidders (Google, Facebook, ...) have more time and energy to:
        - Understand our users and their interests
        - Ideate new ad formats, messaging & channels
        - Form hypotheses for big shots on goals
- We have many exciting ideas for the continued iterations to Symphony:
        - Incorporating seasonal effects like weather & time of day
        - Better marketplace context to inform our bidders
        - Intelligent segmentation & personalization
- Types of data to save:
        - Data Stores — tables, schemas, documents of structured data stores like Hive,
        Presto, MySQL, as well as unstructured data stores (like S3, Google Cloud
        Storage, etc.)
        - Dashboards/reports — saved queries, reports and dashboards in BI/reporting
        tools like Tableau, Looker, Apache Superset, etc.
        - Events/Schemas — Events and schemas stored in schema registries or tools like
        Segment.
        - Streams — Streams/topics in Apache Kafka, AWS Kinesis, etc.
        - Processing — ETL jobs, ML workflows, streaming jobs, etc.
        - People — I don’t mean a software stack, I mean good old people like you and me who carry data in our head and in our organizational structure, so information like name, team, title, data resources frequently used, data resources bookmarked are all important pieces of information in this category.
        
# src: https://www.data-mania.com/blog/data-science-in-marketing-what-it-is-how-you-can-get-started/
- Introduction:
        - the email really got me thinking about data science in marketing, and what it
        was about my LinkedIn profile that had piqued her interest
        - In fact, I’d say that about 30% of the work I do with my business is related
        tomarketing efforts. These efforts include social media marketing, content
        marketing, email marketing, social analytics, Google Analytics, online marketing
        strategy development, and Facebook advertising – just to name a few!
        - And since I use my skills in data science and analytics to bolster the
        marketing efforts of my brand, it makes sense why she would see me as a good fit
- Marketing Data Analyst:
        - Marketing Data Analysts work in a very similar capacity to Business Analysts,
        except that their efforts are focused solely on marketing initiatives
        - Marketing Data Analysts collect and analyze both internal and external
        datasets, and they use the information they surmise to help them plan and
        implement marketing initiatives for their organization
        - Marketing data analysts generally produce descriptive and diagnostic insights
        based on basic data monitoring and trend analysis
        - A big part of their role is based in market research and planning
        - As far as technical skill requirements, these professionals stick to using
        Excel, SQL, and maybe SAS to do their jobs. 
- Marketing Data Scientist:
        - It is a data scientist role that focuses exclusively on improving
        organizational marketing effectiveness
        - Marketing Scientists analyze both internal and external datasets, and they use
        the insights they derive to inform their organization about customer behavior,
        and to advise about modifications or additions to marketing tactics and analysis
        methodologies
        - Marketing data scientists are tasked with producing reliable predictive and
        prescriptive insights based on advanced statistical modeling and/or machine
        learning methodologies
        - Another big part of their role is soft skills – they must be able to
        communicate complex ideas in simple terms, so that management and support
        personnel can understand and benefit from the work they do
        - Job Roles:
                - Generate prescriptive insights – tactical and strategic insights to improve marketing effectiveness
                - Exploratory data analysis
                - Metric and method selection
                - A/B testing
                - Advising, training, and assisting management and other professionals
                in working with and understanding organizational data
        - Required Skills:
                - SQL
                - Data visualization (Tableau, D3.js, etc.)
                - Scripting in Python or R
                - Predictive modeling (statistical and/or machine learning methods)
                - Great “people skills” – to collaborate with data engineers, business
                management, and other support personnel

# src: https://smallbusiness.chron.com/statistics-applied-marketing-36531.html
- Introduction:
        - Statistics are applied in marketing to identify market trends, and to measure
        and evaluate the potential and success of marketing programs. The secret to
        successful marketing is to identify the target market accurately and to use
        effective marketing communications channels and tactics to reach it. Statistics
        can help the marketer achieve both of those goals as well as evaluate the
        success of the marketing effort and provide data on which to base changes to the
        marketing program.
- Data Source
        - The most basic use of statistics in marketing is as a source of data.
        - Statistics provide demographic information such as the number of potential
        customers in a geographical area, their ages, income levels and consumer
        preferences.
        - Used as part of competitor analysis, statistics can identify the major
        competitors, their market share and trends in the longevity of their products.
        - Industry sector data helps marketers understand the trends governing supply
        and demand of the product category and fluctuations in its popularity.
- Marketing Mix Modeling
        - The use of marketing mix modeling helps marketers identify the correct
        combination of marketing communications channels to use to reach the target
        market and provide the best return on the marketing investment. 
        - Modeling works by analyzing information and using the technique of statistical
        regression to determine the effectiveness of sales on the market. 
        - The formula for modeling includes creation of a model using sales volumes and
        value as a dependent variable, and then using various marketing channels to
        represent the independent variables.
- Market Tracking
        - Statistics are applied in market tracking to measure customer satisfaction,
        brand loyalty and support, and to assess the relationship of the marketer’s
        company with its customers. 
        - To implement a market-tracking program, the marketer needs access to company
        as well as industry statistics. The tracking program then analyzes the sales
        statistics across all brands in the market to ascertain which brand enjoys the
        highest levels of customer support and loyalty.
- Cross-sell Parameters
        - Marketers use statistics on household parameters to target buyers for
        customized promotions or to cross-sell secondary products. 
        - For example, Walmart’s loyalty card gives the store the ability to record all
        purchases of baby powder by its customers. By analyzing the statistics generated
        by these records, Walmart is able to identify those households that buy baby
        powder but do not buy other baby products.
        - This makes it possible to target those households as potential customers for a
        new brand of organic body powder, because they appear to use powder but do not
        have babies in the home.

# src: https://www.analyticsvidhya.com/blog/2018/12/guide-digital-marketing-analytics/
- As a Digital Analytics Professional, why should I even care about Digital Media?
        - Answer the questions:
                - What data is useful?
                - How to best capture this data for accuracy?
                - How to use the tool(s) you have created with data science
        - Work on Google Ads (Adwords) and Adsense
                - Contextual Targeting: Google Bot crawls through the page and uses
                Natural Language Processing (NLP) to categorize the page by analyzing
                the article topic. Only text is analyzed and images/videos are simply
                ignored
                - Placement Targeting: Bidding based simply on placement. Here,
                advertisers hand pick publishers where they wish to show their ads, and
                simply bid for placements
                - Interest-based Advertising: Finally, an advertiser can target ads
                based on inferred interest. This one is specifically targeted based on
                the visitor and not the content. For instance, if a user visits an
                airline website and then comes to Analytics Vidhya. Even though flight
                booking ads are not really relevant for AV content, they can still come
                up based on Interest-based Advertising
        - Key metrics (DoubleClick)
                - Impressions
                - Coverage
                - Monetized Pageviews
                - Viewable Impressions %
                - Clicks
                - CTR
                - Revenue / 1000 Sessions
                - eCPM
        - Working with GTM (Google Tag Manager) and Google Analytics platforms.
        - Remarketing
        - Checking average target audience
