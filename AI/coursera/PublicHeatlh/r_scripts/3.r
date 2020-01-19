g <- read.csv('/home/max/Documents/github/Fun/AI/coursera/PublicHeatlh/w2_r_scripts/dataset.csv', sep=',', header=TRUE)

head(g)

g$fruit_veg = g$fruit + g$veg
g$five_a_day = ifelse(g$fruit_veg >= 5, 1, 0)

# Conduct chi2 test
chisq.test(x=g$fruit_veg, y=g$cancer)
chisq.test(x=g$five_a_day, y=g$cancer)

# Conduct independent-samples t-test
## x predicts y and where both x and y are continuous
t.test(g$age, g$bmi)
## same, but x is binary
t.test(g$bmi~g$cancer) 

t.test(g$age~g$cancer)
t.test(g$age~g$cancer, var.equal=TRUE)   # two-sided t-test

