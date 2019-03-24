source: https://towardsdatascience.com/generative-adversarial-networks-gans-a-beginners-guide-5b38eceece24

# Generative Algorithms Classification.
- You can group generative algorithms into one of three buckets:
1. Given a label, they predict the associated features (Naive Bayes)
2. Given a hidden representation, they predict the associated features (Variational Autoencoders, Generative Adversarial Networks)
3. Given some of the features, they predict the rest (Inpainting, Imputation)
- GANs have incredible potential, because they can learn to imitate any distribution of data.

# GANs parts: Generative Part.
- Called the Generator.
- Given a certain label, tries to predict features.
- Example: given an email is marked as spam, predicts (generates) the tex tof the email.
- Generative models learn the distribution of individual classes.

# GANs parts: Adversarial Part.
- Called the Discriminator.
- Given the features, tries to predict the label.
- Example: given the text of an email, predicts (discriminates) whether spam or not-spam.
- Discriminative models learn the boundary between classes.

# How do GANs work?
- One neural network, called the Generator, generates new data instances, while the other, the Discriminator, evaluates them for authenticity.
- Example: game of cat and mouse between a counterfeiter (Generator) and a cop (Discriminator). The counterfeiter is learning to create fake money, and the cop is learning to detect the fake money. Both of them are learning and improving. The counterfeiter is constantly learning to create better fakes, and the cop is constantly getting better at detecting them. The end result being that the counterfeiter (Generator) is now trained to create ultra-realistic money!

# MNIST example
- Generator creates new images like those found in the MNIST dataset, which is taken from the real world.
- **The goal of the Discriminator**, when shown an instance from the true MNIST dataset, is to recognize them as authentic.
- Meanwhile, the Generator is creating new images that it passes to the Discriminator. It does so in the hopes that they, too, will be judged authentic, even though they are fake. 
- **The goal of the Generator** is to generate passable hand-written digits, to lie without being caught. **The goal of the Discriminator** is to classify images coming from the Generator as fake.
## GAN Steps for MNIST example.
2. The Generator takes in random numbers and returns an image.
2. This generated image is fed into the Discriminator alongside a stream of images taken from the actual dataset.
3. The Discriminator takes in both real and fake images and returns probabilities, a number between 0 and 1, with 1 representing a prediction of authenticity and 0 representing fake.
## Feedback loops.
- The Discriminator is in a feedback loop with the ground truth of the images (are they real or fake), which we know.
- The Generator is in a feedback loop with the Discriminator (did the Discriminator label it real or fake, regardless of the truth).

# Tips for training GANs
- Pre-training the Discriminator before you start training the Generator will establish a clearer gradient.
- When you train the Discriminator, hold the Generator values constant. When you train the Generator, hold the Discriminator values constant. This gives the networks a better read on the gradient it must learn by.
- GANs are formulated as a game between two networks and it is important (and difficult!) to keep them in balance. If either the Generator or Discriminator is too good, it can be hard for the GAN to learn.
- GANs take a long time to train. On a single GPU a GAN might take hours, on a single CPU a GAN might take days.

