g <- read.csv(
  file = '/home/max/Documents/github/Fun/AI/coursera/PublicHeatlh/w2_r_scripts/dataset.csv',
  header = TRUE,
  sep = ','
)

g$fruit_veg = g$fruit + g$veg

# Display default histograms

hist(
  g$fruit_veg,
  main = "Daily consumption of combined fruit + vegetables",
  xlab = "Number of portions",
  axes = 'F'
)
axis(side = 1, at = seq(0, 15, 1))
axis(side = 2, at = seq(0, 20, 2))

# Display ggplot histograms
require(ggplot2)

ggplot() + 
  geom_histogram(data = g, aes(x = fruit_veg), binwidth = 0.5, fill = 'grey', col = 'black') +  # bins=20, 
  labs(x = "Number of portions", y = "Frequency") +
  scale_x_continuous(breaks = seq(from = 0, to = 15, by = 1))
