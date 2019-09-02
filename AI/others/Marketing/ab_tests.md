# https://towardsdatascience.com/data-science-you-need-to-know-a-b-testing-f2f12aff619a
- A/B test, randomised controlled trial (RCT)

- The most accurate tool for estimating effect size & ROI.
- Provides us with **causality**
    - => we can prove that our new product actually works.

- INVALID APPROACH:
    - website, run both versions to selected customers and make a judgement based on these numbers alone.
    - Type I error — or falsely concluding that your intervention was successful (which here might be falsely concluding that layout B is better than Layout A). Also known as a false positive result.
    - Type II error — falsely concluding that your intervention was not successful. Also known as a false negative result.
    - These errors derive from a problem of drawing conclusions about a population (in this case, all website customers) from a sample (in this case, the visitors who participated in our layout trial).

- An A/B test will enable us to
    - accurately quantify our effect size and errors
    - calculate the probability that we have made a type I or type II error.

- **We should only estimate the ROI** (return on investment) **of a new product once we understand our effect size and errors**.
    - == only once we understand the true effect size and robustness of our results, can we proceed to making business-impact decisions

- **Key parts of an A/B test":
    1. Hypothesis
        - Should be testable and measurable
        - Has clearly defined evaluation criteria 
            - e.g. are we measuring the average or median? at the individual or group level?
    2. Randomization
        - randomize using R/Python
        - ensure our Treatmant and Control groupos are balanced 