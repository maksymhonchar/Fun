# Central limit theorem
- CLT - theorem that states that the distribution of sample means approximates a normal distribution as the sample size gets larger (assuming that all samples are identical in size), regardless of population distribution shape.

- CLT (Expln 2) - theory that states that given a sufficiently large sample size from a population with a finite level of variance, the mean of all samples from the same population will be approximately equal to the mean of population.
- ALSO: all the samples will follow an approximate normal distribution pattern, with all variances being approximately equal to the variance of the population divided by each sample's size.

- Sample size >= 30 - sufficient for the CLT to hold (== distribution of sample means is fairly normally distributed).
- The more samples -> the more looking like that of a normal distribution.

- Explanation 3:
    - Let's say we have a distribution with it's mean and variance (and std).
    - Take N samples from the distribution and calculate it's mean (average value). Do that M times.
    - Plot each sample mean on the frequency distribution.
    - With m->inf, frequency distribution will be normal (will approximate to normal distribution).
    - With N->inf, std becomes slower, mean is the same, will even more approximate to normal distribution.
    - "If we add the mean of all of those actions in the distribution, we get a normal distribution" == frequency plot looks really like normal distribution

- Example:
    - If an investor is looking to analyze the overall return for a stock index made up of 1,000 stocks, he or she can take random samples of stocks from the index to get an estimate for the return of the total index.
    - The average returns from these samples approximates the return for the whole index and are approximately normally distributed.
        - The approximation holds even if the actual returns for the whole index are not normally distributed.

- Example: https://medium.com/greyatom/central-limit-theorem-a-real-life-case-study-da079b4ba2fc
    - Prereqs: A business client of FedEx wants to deliver urgently a large freight from Denver to Salt Lake City. When asked about the weight of the cargo they could not supply the exact weight, however they have specified that there are total of 36 boxes.
    - Task: tell the executives quickly whether or not they can do certain delivery.
    - Data we have: data about cargos:
        - mean = 33kg, std = 1.36kg;
        - max cargo weight for the plane: 1200kg
    - QUESTION: what is the probability that all of the cargo can be safely loaded onto the planes and transported?
    - ANSWER:
        1. Using CLT, find the mean and std deviation of the sample mean.
        mu_sample = mu = 33
        std_sample = std / sqrt(cnt_boxes) = 1.36 / sqrt(36) = ...
        plane_capacity = 1200kg
        2. calculate the critical mass (X crit) of each box
        x_crit = (plane_capacity) / (boxes_cnt) = 73.06 lb/box
        So, to safely takeoff the plane, the average weight of the each box should not exceed 73.06 lb/box.
        3. calculate the Z-score = (x_crit - mu_sample) / (std_sample) = ... = 2.12
        4. Refer to the z value from the table to find out the probability (standard normal probabilities table)
        P(x < x_crit) = 0.9830 = 98.3%
    - RESULT:
        - Now, you can go to the manager and tell him that I have done the calculations and the probability that the plan can safely takeoff is 98.3% and 1.7 % chance it cannot takeoff.
        - And now its up-to the manager to make a decision whether they are ready to take risk of 1.7% or not.

# Law of large numbers
- Theorem that describes the result of performing the same experiment a large number of times.
- According to the law, the AVERAGE of the results obtained from a large number of trials should be close to the expected value, and will tend to become closer as more trials are performed.

- RU: среднее арифметическое какой-либо достаточно большой выборки из фиксированного распределения близко к математическому ожиданию этого распределения.

- The LLN is important because it guarantees stable long-term results for the averages of some random events

- It is important to remember that the law only applies (as the name indicates) when a large number of observations is considered.

- Example: rolling a fair dice.
- Every roll as equal probability: (1+2+3+4+5+6)/6=3.5
- According to the LLN, if a large number of 6-sided dice are rolled, the average of their values (sample mean) is likely to be close to 3.5, with the precision increasing as more dice are rolled.

- Пример: предположим, что нам нужно оценить уровень заработка людей в каком-то регионе. Если мы будем рассматривать 10 наблюдений, где 9 человек получают 20 тыс. рублей, а 1 человек – 500 тыс. рублей, среднее арифметическое составит 68 тыс. рублей, что, естественно, маловероятно. Но если мы возьмем в расчет 100 наблюдений, где 99 человек получают 20 тыс. рублей, а 1 человек – 500 тыс. рублей, то при расчете среднего арифметического получим 24,8 тыс. рублей, что уже ближе к реальному положению дел. Увеличивая число наблюдений, мы будем заставлять среднее значение стремиться к истинному показателю.

- Conclusion 2: если опытов много, вне зависимости от того, как воздействуют факторы, всегда можно утверждать, что практическая вероятность близка к вероятности теоретической.

# Expected value (Математическое ожидание)
- Математическое ожидание - это мера среднего значения случайной величины в теории вероятности.
    - Математическое ожидание - это число, вокруг которого сосредоточены значения случайной величины.
    - Математическое ожидание - это средневзвешенная величина всех возможных значений, которые может принимать эта случайная величина.
    - Математическое ожидание – это сумма произведений всех возможных значений случайной величины на вероятности этих значений.
- Математическое ожидание характеризует распределение возможных параметров случайной величины.
- Применяется при проведении технического анализа, исследовании числовых рядов, изучении непрерывных и продолжительных процессов. Имеет важное значение при оценке рисков, прогнозировании ценовых показателей при торговле на финансовых рынках, используется при разработке стратегий и методов игровой тактики в теории азартных игр.
- Примеры:
    - МО - это средняя выгода от того или иного решения при условии, что подобное решение может быть рассмотрено в рамках теории больших чисел и длительной дистанции.
    - МО - это в теории азартных игр сумма выигрыша, которую может заработать или проиграть игрок, в среднем, по каждой ставке.
    - МО - это процент прибыли на выигрыш, умноженный на среднюю прибыль, минус вероятность убытка, умноженная на средний убыток.

- Wiki: МО - среднее значение случайной величины при стремлении количества выборок или количества её измерений (иногда говорят — количества испытаний) к бесконечности.

# Confidence interval (Доверительный интервал)
- Доверительный интервал - это интервал, построенный с помощью случайной выборки из распределения с неизвестным параметром, такой, что он содержит данный параметр с заданной вероятностью.
- Доверительный интервал - термин, используемый в математической статистике при интервальной оценке статистических параметров, более предпочтительной при небольшом объёме выборки, чем точечная. 
    - Доверительным называют интервал, который покрывает неизвестный параметр с заданной надёжностью.
- ... (todo)

# Когортный анализ
- Когортный анализ — метод анализа эффективности бизнеса. Суть состоит в том, чтобы анализировать поведение групп людей, объединенных по какому-либо признаку во времени.
- Когорты позволяют анализировать тренды внутри метрики и отличать продуктовые метрики от метрик роста проекта.
- Наиболее популярный фактор деления на когорты — первое посещение сайта/регистрация/установка приложения.
- Оценка продукта происходит не по итоговой метрике, а по каждой отдельной когорте этой метрики. Когорта — группа людей, которые сделали одно и то же действие в определенный период времени.
    - Пользователи разделяются на когорты, например, в момент первого посещения сайта/регистрации/установки приложения. И в дальнейшем анализ действий юзера проводится внутри каждой когорты.
- Кейсы:
    - почтовая рассылка. Результат email-рассылки на сайте X — конверсия отправленного письма в переходы составила 12%. Пользователи, которые зарегистрировались 3 недели назад (желтый график), переходят по ссылкам в письме в 2 раза чаще, чем пользователи, которые зарегистрировались 2 месяца назад (зеленый график).
    Результат email-рассылки на сайте X — конверсия отправленного письма в переходы составила 12%. Пользователи, которые зарегистрировались 3 недели назад (желтый график), переходят по ссылкам в письме в 2 раза чаще, чем пользователи, которые зарегистрировались 2 месяца назад (зеленый график).
    - рекламный баннер. Компания X запустила рекламную кампанию в Adwords. Если проводить оценку её эффективности только по доходности пользователя в день привлечения, то результаты не будут показательными.
    Пользователи в первый день жизни наиболее активны и приносят 30% от всей прибыли за день. На следующий день они приносят 10% прибыли, на следующий — еще 10%. Таким образом, накапливается эффект от рекламных переходов, и деньги продолжают поступать от юзеров, привлеченных какое-то время назад, в течение всего периода использования ими продукта.

# Variance (Дисперсия случайной величины)
- Technical definition:
    - V is The average of the squared differences from the mean
    - V is the expectation of the squared deviation of a random variable from its mean.
- It measures how far a set of (random) numbers are spread out from their average value.
- Example:
    - The data set 12, 12, 12, 12, 12 has a var. of zero (the numbers are identical).
    - The data set 12, 12, 12, 12, 13,013 has a var. of 28171000; a large change in the numbers equals a very large number.

# Covariance (ковариация)
- Covariance is a measure of the relationship between two random variables
- The metric evaluates how much – to what extent – the variables change together.
- Covariance is measured in units. The units are computed by multiplying the units of the two variables. The variance can take any positive or negative values.
    - Positive covariance: Indicates that two variables tend to move in the same direction.
    - Negative covariance: Reveals that two variables tend to move in inverse directions.

# Sampling bias
- Sampling bias - bias in which a sample is collected in such a way that some members of the intended population have a lower sampling probability than others.

- Biased sample == non-random sample of a population (or non-human factors) in which all individuals were NOT EQUALLY LIKELY to have been selected.

- Sampling bias occurs in practice as it is practically impossible to ensure perfect randomness in sampling. If the degree of misrepresentation is small, then the sample can be treated as a reasonable approximation to a random sample

- Types of sampling bias:
    - Selection from a specific real area
        - For example, a survey of high school students to measure teenage use of illegal drugs will be a biased sample because it does not include home-schooled students or dropouts
    - Self-selection bias
        - whenever the group of people being studied has any form of control over whether to participate
    - Pre-screening of trial participants, or advertising for volunteers within particular groups
        - For example, a study to "prove" that smoking does not affect fitness might recruit at the local fitness center, but advertise for smokers during the advanced aerobics class, and for non-smokers during the weight loss sessions
    - Exclusion bias
        - exclusion of particular groups from the sample, e.g. exclusion of subjects who have recently migrated into the study area (this may occur when newcomers are not available in a register used to identify the source population).
    - Healthy user bias
        - when the study population is likely healthier than the general population. For example, someone in poor health is unlikely to have a job as manual laborer.
    - Berkson's fallacy
        - when the study population is selected from a hospital and so is less healthy than the general population
        - This can result in a spurious negative correlation between diseases: a hospital patient without diabetes is more likely to have another given disease such as cholecystitis, since they must have had some reason to enter the hospital in the first place. 
    - Survivorship bias
        - only "surviving" subjects are selected, ignoring those that fell out of view.
        - For example, using the record of current companies as an indicator of business climate or economy ignores the businesses that failed and no longer exist.
    - Malmquist bias
        - an effect in observational astronomy which leads to the preferential detection of intrinsically bright objects.

- Problems due to sampling bias
    - Sampling bias is problematic because it is possible that a statistic computed of the sample is systematically erroneous
    - Sampling bias can lead to a systematic over- or under-estimation of the corresponding parameter in the population
    - Also, if the sample does not differ markedly in the quantity being measured, then a biased sample can still be a reasonable estimate.

- Historical examples: US elections
    - In the early days of opinion polling, the American Literary Digest magazine collected over two million postal surveys and predicted that the Republican candidate in the U.S. presidential election
    - The result was the exact opposite
    - The Literary Digest survey represented a sample collected from readers of the magazine, supplemented by records of registered automobile owners and telephone users
    - This sample included an over-representation of individuals who were rich, who, as a group, were more likely to vote for the Republican candidate

# Bias-variance tradeoff (дилемма смещения-дисперсии)
- Дилемма смещения–дисперсии — это свойство набора моделей предсказания, по которому модели с **меньшим смещением** в параметре оценки имеют **более высокую дисперсию** оценки параметра в выборках **и наоборот**.
- Дилемма или проблема смещения–дисперсии является конфликтом в попытке одновременно минимизировать эти два источника ошибки? которые мешают алгоритмам обучения с учителем делать обобщение за пределами тренировочного набора.
- **Смещение** — это ошибка, возникающая в результате ошибочного предположения в алгоритме обучения. В результате большого смещения алгоритм может пропустить связь между признаками и выводом (недообучение).
- **Дисперсия** — это ошибка чувствительности к малым отклонениям в тренировочном наборе. При высокой дисперсии алгоритм может как-то трактовать случайный шум в тренировочном наборе, а не желаемый результат (переобучение).

# Interaction (statistics) / Взаимодействие (статистика)
- Взаимодействие - ситуация, когда одновременное воздействие 2 и более переменных на другую переменную не является аддитивным.

- Взаимодействия исследуются в рамках регрессионного анализа (OLS)
- Взаимодействия также можно смотреть в ANOVA

- Если некие две переменные взаимодействуют, предсказать эффект от изменения одной из них на зависимую переменную становится сложнее. 

# Correlations
- Correlation is a bivariate analysis that measures the strength of association between two variables and the direction of the relationship.
- The value of the correlation coefficient varies between +1 and -1
    - +1 indicates a perfect degree of association between the two variables.
    - As the correlation coefficient value goes toward 0, the relationship between the two variables will be weaker.
- "+" means positive relationship, "-" indicates a negative relationship.
- Usually, in statistics, we measure 4 types of correlations:
    - Pearson correlation
    - Kendall rank correlation
    - Spearman correlation
    - Point-Biserial correlation

# Pearson correlation
- The most widely used correlation statistic to measure the degree of the relationship between linearly related variables.
- Example:
    - in stock market, if we want to measure the degree of the relationship between linearly related variables.
- Types of research questions a Pearson correlation can examine:
    - Is there a statistically significant relationship between age (years) and height (inches)?
    - Is there a relationship between temperature (Farenheit) and ice cream sales (income)?
    - Is there a relationship between job satisfaction (JSS) and income (dollars)?
- Assumptions:
    - For the Pearson r correlation, both variables should be normally distributed
    - linearity (straight line relationship between each of the two variables)
    - homoscedasticity (variance around the regression line is the same for all values of the predictor variable X) (Гомоскедастичность) (data is equally distributed about the regression line)
- Effect size (Cohen's standard)
    - 0.10 - 0.29 - small association
    - 0.30 - 0.49 - medium association
    - >0.50 - large association or relationship

# Kendall rank correlation
- non-parametric test that measures the strength of dependence between 2 variables.

# Spearman rank correlation (SRC)
- non-parametric test to measure the degree of association between two variables.
- SRC test does not carry any assumptions about the distribution of the data
- SRC test is the appropriate correlation analysis when the variables are measured on a scale that is at least ordinal.
- Types of research questions a Spearman Correlation can examine:
    - Is there a statistically significant relationship between participants' level of education (high school, bachelor's, graduate degree) and their starting salary?
    - Is there a statistically significant relationship between horse's finishing position a race and horse's age?
- Assumptions:
    - Data must be at least ordinal
    - Scores on one variable must be monotonically related to the other variable.

# Uniform distribution (Непрерывное равноменое распределение)
- A uniform distribution (rectangular distribution) - distribution that has constant probability.
- Плотность вероятности на этом интервале постоянна.
    - The area under the density curve is the probability.

# Z-distribution (== Standard normal distribution)
- Is a specific instance of the Normal Distribution
    - mean "0"
    - standard deviation of "1"

# T-distributioon (Student's t-distribution)

# Range, variance and std as measures of dispersion
- More disperse
    - more disperse == items are more further from the mean of the dataset.
    - more disperse == range is larger
    - more disperse == variance is larger
    - more disperse == std is larger
- Range == largest_value - smallest_value from a dataset
- Variance (of the population), sigma^2 == sum( (x_i - mean)^2 ) / dataset_size
    - variance of the sample: divide by DDOF
    - variance of the population == sum of "how far points were from the mean TO THE average of those"
- Standard deviation, sigma:
    - std is the square root of the variance
    - avoid the measure units to describe the data / the distribution
    - gives a good sense of how far away are the points from the mean.

# Power (statistics) - Статистическая мощность
- СМ - вероятность отклонения основной (или нулевой) гипотезы при проверке статистических гипотез в случае, когда конкурирующая (или альтернативная) гипотеза верна.
- Чем выше мощность статистического теста, тем меньше вероятность совершить ошибку второго рода.
- Величина мощности также используется для вычисления размера выборки, необходимой для подтверждения гипотезы с необходимой силой эффекта.

- Величина мощности при проверке статистической гипотезы зависит от следующих факторов:
    - величины уровня значимости, обозначаемого греческой буквой ALPHA, на основании которого принимается решение об отвержении или принятии альтернативной гипотезы;
    - величины эффекта (то есть разности между сравниваемыми средними);
    - размера выборки, необходимой для подтверждения статистической гипотезы.

- вероятность того, что мы на выборке примем гипотезу H1, если на самом деле она верна (= шанс обнаружить эффект, если он на самом деле есть).

- Статистическая мощность зависит от:
    - объёма выборки: чем он больше, тем она выше;
    - размера эффекта: чем он сильнее, тем она выше;
    - от используемого статистического критерия: для разных статистических критериев, проверяющих одну и ту же гипотезу, она будет разной.

- Статистическая мощность:
    - Является критерием для определения объёма выборки с учётом размера ожидаемого эффекта.
    - Важно! Только высокая мощность (0,95 и выше) даёт нам возможность делать достоверный вывод о том, что искомый эффект отсутствует (верна H0).
    - При недостаточной статистической мощности подобный вывод является необоснованным (правильный вывод: мы не обнаружили эффект, но не можем сказать, есть он или нет).

# p-value
- The P value, or calculated probability, is the probability of finding the observed, or more extreme, results when the null hypothesis (H0) of a study question is true.
(!) - P is also described in terms of rejecting H0 when it is actually true, however, it is not a direct probability of this state.

- The term significance level (alpha) is used to refer to a pre-chosen probability and the term "P value" is used to indicate a probability that you calculate after a given study.

- If your P value is less than the chosen significance level then you reject the null hypothesis i.e. accept that your sample gives reasonable evidence to support the alternative hypothesis.
(!) - It does NOT imply a "meaningful" or "important" difference; that is for you to decide when considering the real-world relevance of your result.

# Significance
- Significance is the percent of chance that a relationship may be found in sample data due to luck. 
- Researchers often use the 0.05% significance level.

# Sampling
- Sampling - selection a predetermined number of observations from a larger population.

- By getting such observations (samples) we can next make some statistical inferences about the population.
- Companies use sampling to identify the needs and wants of their target audience.

- RANDOM SAMPLING
    - In data collection, every individual observation has equal probability to be selected into a sample.
    - In random sampling, there should be no pattern when drawing a sample.
    - For example, a lottery system could be used to determine the average age of students in a university by sampling 10% of the student body.

- SYSTEMATIC SAMPLING
    - Systematic sampling uses a random starting point and a periodic interval to select items for a sample.
    - The sampling interval is calculated as the population size divided by the sample size
    - Example:
        - Assume that a CPA is auditing the internal controls related to the cash account and wants to test the company policy that stipulates that checks exceeding $10,000 must be signed by two people.
        - The accountant's population consists of every company check exceeding $10,000 during the fiscal year, which, in this example, was 300.
        - The CPA firm uses probability statistics and determines that the sample size should be 20% of the population or 60 checks.
        - The sampling interval is 5 (300 checks/60 sample checks);
            - therefore, the CPA selects every fifth check for testing.
        - Assuming no errors are found in the sampling test work, the statistical analysis gives the CPA a 95% confidence rate that the check procedure was performed correctly.
        - The CPA tests the sample of 60 checks and finds no errors;
        - the accountant concludes that the internal control over cash is working properly.

- Other types of random sampling
    - (src: https://www.statisticssolutions.com/sample-size-calculation-and-sample-size-justification/sampling/)

    - Simple random sampling: By using the random number generator technique, the researcher draws a sample from the population called simple random sampling. Simple random samplings are of two types. One is when samples are drawn with replacements, and the second is when samples are drawn without replacements.
    - Equal probability systematic sampling: In this type of sampling method, a researcher starts from a random point and selects every nth subject in the sampling frame. In this method, there is a danger of order bias.
    - Stratified simple random sampling: In stratified simple random sampling, a proportion from strata of the population is selected using simple random sampling. For example, a fixed proportion is taken from every class from a school.
    - Multistage stratified random sampling: In multistage stratified random sampling, a proportion of strata is selected from a homogeneous group using simple random sampling. For example, from the nth class and nth stream, a sample is drawn called the multistage stratified random sampling.
    - Cluster sampling: Cluster sampling occurs when a random sample is drawn from certain aggregational geographical groups.
    - Multistage cluster sampling: Multistage cluster sampling occurs when a researcher draws a random sample from the smaller unit of an aggregational group.

- Other typess of non-random sampling
    - NOTE: Non-random sampling is widely used in qualitative research.  Random sampling is too costly in qualitative research.

    - Availability sampling: Availability sampling occurs when the researcher selects the sample based on the availability of a sample. This method is also called haphazard sampling. E-mail surveys are an example of availability sampling.
    - Quota sampling: This method is similar to the availability sampling method, but with the constraint that the sample is drawn proportionally by strata.
    - Expert sampling: This method is also known as judgment sampling. In this method, a researcher collects the samples by taking interviews from a panel of individuals known to be experts in a field.

# Type I and Type II errors (Alpha, Beta)
- Type I error: False Positives
    - "False Hit", "False alarm"
    - Type I error == Alpha 
    - Alpha == Significance level - probability of rejecting the Null Hypothesis GIVEN that it is true
- Example: Alpha=0.05% ==> it is acceptable to have a 5% probability of incorrectly rejecting the null hypothesis.

- Type II error: False Negatives
    - "Missed"
    - Type II error == Beta
    - (1-Beta) == Statistic power == Статистическая мощность
        - вероятность того, что мы на выборке примем гипотезу H1, если на самом деле она верна (= шанс обнаружить эффект, если он на самом деле есть).

# R2 score, Coefficient of determination, Коэффициент детерминации
- R2 - proportion of the variance in the dependent variable that is predictable from the independent variable(s).

- It provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model

- R-squared is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.

- R-squared = Explained variation / Total variation

- R-squared is always between 0 and 100%:
    - 0% indicates that the model explains none of the variability of the response data around its mean.
    - 100% indicates that the model explains all the variability of the response data around its mean.

# Linear regression assumptions
- Linear regression is an analysis that assesses whether one or more predictor variables explain the dependent (criterion) variable.

- The regression has five key assumptions:
    1. Linear relationship
        - check it with scatter plots to review if data is linearly placed in 2d plots or not.
    2. Multivariate normality
        - check it with a histogram or a Q-Q plot.
        - Normality can be checked with a goodness of fit test, e.g., the Kolmogorov-Smirnov test
        - When the data is not normally distributed - apply a non-linear transformation (i.e. log-transform)
    3. No or little multicollinearity
        - Multicollinearity occurs when the independent variables are too highly correlated with each other.
        - Check multicollinearity using:
            - Correlating matrix - with Pearson's Bivariate Corr coeffs need to be smaller than 1
            - Tolerance - 1-R2. T<0.1 - there might be collinearity. T<0.01 - there certainly is.
            - VIF - variance inflation factor - 1/T. VIF>10 - there might be. VIF>100 - there certainly is multicollinearity between the variables.
        - Fix:
            - centering the data.
            - removing variables with high VIF values.        
    4. No auto-correlation
        - Autocorrelation occurs when the residuals are not independent from each other.
            - In other words when the value of y(x+1) is not independent from the value of y(x).
        - For instance, this typically occurs in stock prices, where the price is not independent from the previous price.
        - You can test the linear regression model for autocorrelation with the Durbin-Watson test
    5. Homoscedasticity
        - meaning the residuals are equal across the regression line

# ANOVA
- todo: http://www.dartistics.com/anova.html

- Analysis of Variance (ANOVA) is a parametric statistical technique used to compare datasets.
- The main purpose of an ANOVA is to test if two or more groups differ from each other significantly in one or more characteristics

- It is similar in application to techniques such as t-test and z-test, in that it is used to compare means and the relative variance between them.
- However, analysis of variance (ANOVA) is best applied where more than 2 populations or samples are meant to be compared.

- ANOVA assumptions:
    -  Independence of case. 
        - The case of the dependent variable should be independent or the sample should be selected randomly
        - There should not be any pattern in the selection of the sample.
    - Normality
        - Distribution of each group should be normal
        - The Kolmogorov-Smirnov or the Shapiro-Wilk test may be used to confirm normality of the group
    - Homogeneity
        - variance between the groups should be the same
        - Levene’s test is used to test the homogeneity between groups.

- Types of ANOVA:
    - One way analysis
        - comparing more than three groups based on one factor variable
        - For example, if we want to compare whether or not the mean output of three workers is the same based on the working hours of the three workers
    - Two way analysis
        - When factor variables are more than two
        - For example, based on working condition and working hours, we can compare whether or not the mean output of three workers is the same
    - K-way analysis
        -  When factor variables are k

- Key terms and concepts:
    - SUM OF SQUARE BETWEEN GROUPS
        1. we calculate the individual means of the group
        2. then we take the deviation from the individual mean for each group
        3. finally, we will take the sum of all groups after the square of the individual group. 
        - Sum of squares within group: In order to get the sum of squares within a group, we calculate the grand mean for all groups and then take the deviation from the individual group. The sum of all groups will be done after the square of the deviation.
    - F-RATIO
        - the sum of the squares between groups will be divided by the sum of the square within a group.
    - DEGREE OF FREEODM
        - subtract one from the number of groups
    - THE REJECTION REGION

- Example of using ANOVA
    - Step 0.1. Let’s consider three forms of channels, including:
        - Desktop/Laptop
        - Mobile
        - Tablet
    - Step 0.2. Also, let’s consider the number of goal completions during a five-day period
    - Step 1. Calculate the Mean for Each Channel.
        - Desktop/Laptop (mean 1) = 4.2
        - Mobile (mean 2) = 3.8
        - Table (mean 3) = 2.4
    - Step 2. Calculate the Mean for All Three Channels Combined
        - "grand mean"
        - The sum for the three channels is 52 and the mean is 3.47.
    - Determine the Sum of Squares Error (SSe)
        - We calculate SSe by squaring the difference between the observed value and the expected (mean) value:
            - Desktop/Laptop (group 1) = 2.8
            - Mobile (group 2) = 2.8
            - Tablet (group 3) = 5.2
    - Determine the Sum of Squares Between Groups (SSb)
    - Calculate the Mean Square Errors (MSe)
    - Determine the Mean Square Between Groups (MSb)
    - Determine the Actual F-value
    - Look Up the Critical F-value
    - Compare the Actual and Critical F-values
- Main Effect Interpretation
Device type and, separately, last touch channel appear to have a main effect on number of goals completed. We note that laptop/desktop users are more likely to complete more goals compared to phone users. Similarly, customers who rely on organic search to reach our website are more likely to complete more goals compared to other points of last contact.

# Биномиальное распределение вероятностей (биномиальный закон распределения вероятностей)
- wiki: Биномиа́льное распределе́ние в теории вероятностей — распределение количества «успехов» в последовательности из N независимых случайных экспериментов, таких, что вероятность «успеха» в каждом из них постоянна и равна P.

- Пусть проводится n независимых испытаний (не обязательно повторных), в каждом из которых случайное событие A может появиться с вероятностью P.
- Тогда случайная величина X – число появлений события A в данной серии испытаний, имеет биномиальное распределение.

- Вероятности p_i представляют собой члены бинома Ньютона, благодаря чему распределение и получило своё название.

- Пример:
    - Монета подбрасывается 5 раз
    - Тогда случайная величина X – количество появлений орла - распределена по биномиальному закону
    - Орёл обязательно выпадет или x0=0, x1=1, x2=2, x3=3, x4=4 или х5=5 раз.

# Exact test (точный тест)
- In statistics, an exact (significance) test is a test where if the Null hypotheses is true then all assumptions, upon which the derivation of the distribution of the test statistic is based, are met
- Using exact test you can actually get an exact p-value
- Одним из первых примеров пермутационного статистического критерия является точный тест Фишера, применяющийся в анализе таблиц сопряжённости для выборок маленьких размеров.