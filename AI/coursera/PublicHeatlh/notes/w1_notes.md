# Intro: about the course
- take a medical angle on statistics with examples coming from medical research and public health in particular
- how health research is reported in the news and other media
- whether we need to dig a bit deeper into the methods used in the researches from the different media sources
- take a peek up what medical research is and how & why you turn a vague notion into a scientifically testable hypothesis
- learn about key statistical concepts like sampling uncertainty variation missing values distributions etc
- get hands dirty with analysing a dataset exploring how what people eat affects their risk of cancer


# Uses of statistics in public heatlh
- Wiki: statistics is a branch of mathematics dealing with the collection, analysis, interpretation, presentation and organizing of data
- John Snow - published a study about relation between the districs in London and spreading of Cholera disease.
    - His theory was that the worst hit districts would be those nearest the River Thames into which a lot of sewage was pumped - not a bad theory!
- William Farr - published a contrary study about associations between the numbers of deaths and factors like elevation above sea level and the average number of people per house TO the fact of dying because of Cholera disease
- In 2004 the Journal of Public Health published an article re-examining the 1849 cholera outbreak data: https://www.ncbi.nlm.nih.gov/pubmed/15313591. The article presented the original data and the analysis performed by both Farr and Snow as well as a reanalysis. The authors used a method for the reanalysis that had not been developed in 1854, logistic regression
- What both men failed to recognise in their analysis is that correlation or associations between a potentially explanatory variable and outcome data does not imply causation
    - **correlation and associations do not imply causation!**
- Present day examples - UN has set goals for halving extreme poverty rates and halting the spread of HIV/AIDS by year of 2030.
    - The idea was to provide a framework of time-bound targets by which progress could be measured for each of the 8 goals
    - The UN's report made some bold claims about the action resulting from the goals
    - How do we know this and how have UN's statistics themselves help to drive progress?
    - The MDG monitoring experience … demonstrated that effective use of data can help to galvanize development efforts, implement successful targeted interventions, track performance and improve accountability
    - strengthening statistical capacity is the foundation for monitoring progress of the new development agenda
    - promoting open, easily accessible data and data literacy is key for effective use of data for development decision-making
- To get reliable data we need robust IT systems and enough capable analysts - a challenge for many developing countries

# Parkinson's disease treatment reports
- Article: https://www.bbc.co.uk/news/health-40814250
- Questions for the article (from the students):
    - What is the method by which the sample of 62 people was formed
    - How can we ensure the data was gathered and saved for further processing in the right way
    - Which type of epidemiological study is being used?
    - Is the number of 62 patients relevant enough?
    - Were the participants randomized? How was done the randomization of patients to both arms? 
    - Which inclusion and exclusion variables were considered (age group, sex, severity of disease, etc.)?
    - What is the hypothesis behind the success of the drug?
    - Could the researchers have done a bigger study?
    - Could the study have been done for a longer period?
    - What is the usual medication and is it the same for all the 62 patients?
    - Those the drug have side effects on parkison patient?
    - What was the affect of the blood glucose in the patients who were on the medications?
- Problems with the study: there are a lot of methodological issues with the study
- Medical research is hard to do in practice, and not every challenge can be overcome entirely
    - For instance, it's expensive to recruit large numbers of patients, particularly those that don't engage much with the health service
- The study will usually report the number of patients involved but might be short on detail on how they were recruited and what sort of patients didn't take part.
- Questions from the "partkinson's disease study issues" test:
    - Patient selection, i.e. how patients were chosen to take part in the study - You want to be able to assess things such as whether the study was conducted in a single centre or across multiple centres, what patients were eligible i.e. what was their inclusion/exclusion criteria, how many patients chose not to participate etc. These are important issues that are not covered in the article.
    - Treatment allocation, i.e. how patients were chosen to get which treatment - How did they decide which patients received exenatide and which received placebo? Did the clinicians decide or was a more objective, scientific process such as randomisation used? If there is prior knowledge about which treatment a patient will receive a form of selection bias is introduced into the study.
    - Small sample size (the size of the sample of the trial) - This was quite a small trial, so its small sample size was a major limitation
    - Blinded treatment group - Knowledge of which intervention a patient is receiving may introduce biases into the study and affect outcome(s). Therefore where possible it is important to blind patients, clinicians and all others involved in patient care, and any study staff involved in outcome assessment to treatment allocation.
    - Length of follow-up (the length of time that patients were followed up) - Patients were treated for 48 weeks, and then they were taken off the drug and monitored for another 3 months. That makes about a year in total, which is not that long (but still longer than many trials). If the follow-up time isn't very long, then you won't be able to see some side-effects or some improvements that take time to appear.
    - Outcome measure (the outcome of interest in the study) - It’s not clear what the outcome of interest is in this study. It does state that patients on “usual medication declined over 48 weeks of treatment. But those given exenatide were stable”. But we can’t say specifically what was declining or stable i.e. it’s unclear what they are measuring to demonstrate treatment effect and to assess the relevance of it.
    - Effect size – clinical vs statistical significance - The article states “This is the first clinical trial in actual patients with Parkinson's where there has been anything like this size of effect” but doesn’t state what this effect size is. We want to know effect sizes so we can assess likely clinical impact on patients.
    - Side-effects (side-effects of the drug) - They don’t mention side-effects in the article but with such a small sample size, rare important side-effects will not have been identified.
    - Patients withdrawing from the study (patients leaving the study) - Withdrawals result in missing data. If number of withdrawals differ between treatment groups this could mean that the reason why the patient withdraws is related to treatment. This would introduce a bias into the results, especially if patients withdraw because the treatment is causing adverse events or they think it's having no effect/benefit.
    - Replication of the study (repeating the study) - This is the first study to show such an effect and so it is important that the findings are replicated. This gives confidence that the original results are reliable and valid.

# Sampling
- 4 important statistical points to ask the authors of the Parkinson article:
    1. Where did they get these patients from and who were they? Single hospital, special clinic for Parkinson's patients
    2. How many patients were recruited? What is the sample size?
    3. How long patients were followed up for? Because Parkinson's disease progressses slowly, 3 months could be not enough for the valid conclusions
    4. What exactly was measured in the study? How the team measured the symptoms or the effects of the drug on the disease? 
- These 4 steps in short:
    1. where to recruit the patients
    2. sample size
    3. length of followup
    4. what to measure

# How to formulate a research question
- Writing out the research question in the form of a testable hypothesis is one of the first key steps in doing a decent research
- To answer any research question you first need to formulate it in such a way that it can be tested scientifically
- Sports analogy:
    - Informal question: whether Messi or Ronaldo is the best footballer of our generation?
    - Creating a scientific question:
        - what evidence can we give to support our keenly held opinions?
            - goals scored, goal/minute, speed of running
        - Hypothesis - Messi differs from Ronaldo on his numbers of goals and assists per minute played
- Main example for the course: **the effects of eating fruits and vegetables**
    - Hypothesis - "Eating fruit and vegetables reduces your risk of getting cancer"
- When you’re doing research question formulation first identify the following:
    - population
    - intervention
    - control
    - outcome
    - timeframe

# Formulating a research question for the Parkinson's disease and supplement studies
- Without knowing the exact research question it is impossible to understand the impact and the results of research
- Hint: identify the success assumptions
- How the study hypothesis is phrased determines how you analyse the data.
    - It also determines how you judge whether the new treatment is better than the comparison treatment
- Hint: what does "better" (worse, early, ...) or some other result-"adjective" actually means?
- Examples of correct research questions:
    - Does Exenatide plus usual care slow the rate of symptom worsening compared with placebo plus usual care in Parkinson's disease patients over 48 weeks of treatment?
    - Do vitamin and mineral supplements improve health? Do vitamin and mineral supplements cause harm?
    - Do vitamin and mineral supplements improve/prevent risk of cardiovascular disease and death?
- Examples of my research questions (probably incorrect):
    - Can the exenatide drug help stabilize patients with Parkinson that stay on their usual medication?
    - What are the positive and negative consequences to the human's health of taking most common supplements?
    - What is the effect of taking combinations of common vitamins and minerals on preventing and treating cardiovascular diseases?
