# Import cancer data
dataset_filepath = 'Documents/github/Fun/AI/coursera/PublicHeatlh/w2_r_scripts/dataset.csv'
g <- read.csv(file = dataset_filepath, header=TRUE, sep=',')

# Display first/last rows
head(g)
tail(g)

# Display dimensions of dataset
dim(g)

# Display unique cnt
table(gender)  # for discrete variables
summary(bmi)  # for continuous variables

# Display specific rows/columns
g[4,]
g[,4]
g[1,2]
g[1:5,]
g[,'gender']
g$gender

# Cast to categorical
gender <- g['gender']
gender <- as.factor(g['gender'])

# Sum variables
fruit <- g[,'fruit']
veg <- g[,'veg']g
fruit_veg = fruit + veg
table(fruit_veg)
g$fruit_veg = g$fruit + g$veg

# Draw a histogram
hist(fruit_veg)

# Display aggregates
sum(g$fruit)
mean(g$fruit)
median(g$fruit)
quantile(g$fruit)

# Display data types
sapply(g, class)
unique(sapply(g, class))

# Display missing values
summary(is.na(g))
table(is.na(g))

table(g$smoking, exclude = NULL)  # NULL = “do not exclude anything”

# Dichotomize patients
g$five_a_day <- ifelse(g$fruit_veg >= 5, 1, 0)
summary(g$five_a_day)
table(g$five_a_day)

g$not_normal_bmi <- ifelse(g$bmi < 18.5 | g$bmi > 25, 1, 0)
table(g$not_normal_bmi)
