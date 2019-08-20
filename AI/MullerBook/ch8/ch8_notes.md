# Approaching ML problem
1. think about what kind of question you want to answer
    - do you want expl analysis and just see if you find smt in your data?
    - do you already have a particular goal in your mind?
2. think about how to define and measure success
    - it is best if you can measure the performance of your
    algorithm directly using a business metric, like increased profit or decreased losses.
3. think about what the impact of a successful solution would be to your overall business or research goals

- Say goal is fraud detection.
    - Question 1: how to measure if fraud detection algo is actually working?
    - Q2: Do I have the right data to evaluate the algorithm?
    - Q3: If I am successful, what will be the business impact of my solution?

4. Acquiring the data
5. Building a working prototype
6. While trying out models, keep in mind that this is only a small part of a larger data science workflow, and model building is often part of a feedback CIRCLE of:
    - collectitng new data
    - cleaning data
    - building models
    - analyzing the models

- Analyzing mistakes:
    - what data could be collected
    - how the task could be reformulated to make machine learning more effective

- Collecting more or different data or changing the task formulation slightly might provide a much higher payoff than running endless grid searches to tune parameters.

# From prototype to production
- A relatively common solution is to **reimplement** the solution that was found by the analytics team inside the larger framework, using a high-performance language. This can be easier than embedding a whole library or programming language and converting from and to the different data formats.
    - This can be easier than embedding a whole library or programming language and converting from and to the different data formats.

- Aspects to keep in mind:
    1. reliability
    2. predictability
    3. runtime
    4. memory requirements
    ...
    N. Simplicity

- Critically inspect each part of your data processing and prediction pipeline and ask yourself how much complexity each step creates, how robust each component is to changes in the data or compute infrastructure, and if the benefit of each component warrants the complexity.

# Testing production systems
- Testing with test set - **offline evaluation**
- Next steps - online testing / live testing.

- To protect against these surprises, most user-facing services employ A/B testing, a form of blind user study.
- Using A/B testing enables us to evaluate the algorithms “in the wild,” which might help us to discover unexpected consequences when users are interacting with our model.
    - Often A is a new model, while B is the established system.

# Out-of-core learning and parallelization over a cluster
- Out-of-core learning describes learning from data that cannot be stored in main memory, but where the learning takes place on a single computer (or even a single processor within a computer).
    - The data is read from a source like the hard disk or the network either one sample at a time or in chunks of multiple samples, so that each chunk fits into RAM.
    - This subset of the data is then processed and the model is updated to reflect what was learned from the data
    - Then, this chunk of the data is discarded and the next bit of data is read.

- Parallelization over a cluster - distributing the data over multiple machines in a compute cluster, and letting each computer process part of the data.
    - distributing the data over multiple machines in a compute cluster, and letting each computer process part of the data.
    - Spark & Hadoop
