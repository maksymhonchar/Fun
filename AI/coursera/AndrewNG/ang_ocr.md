# Photo OCR pipeline == sequence of different modules
- Machine learning pipeline == a system with many stages / components, several of which may use machine learning.
- It is really important that one egineer should work with only 1 module - ANG thought.

- Example of pipeline: face recognition from images
    1. Step 1: get camera image
    2. Step 2: preprocess (remove background)
        - face detection:
            - eyes segmentation
            - nose segmentation
            - mouth segmentation
    3. logistic regression
    4. Label

- Example of pipeline: OCR
    1. Input: image
    2. Text detection
        - rectangle around text region
    3. Character segmentation
        - character by character
    4. Character classification

- Text detection is an unusual problem in computer vision.
    - Because depending on the length of the text we are trying to find, these rectangles that we are trying to find can have completely different aspect.

# Sliding windows approach
- Simpler example: pedestrian detection
    - we want to find individual pedestrian that appear in the image.
    - why is it easier: aspect ration is kind of similar (because human's avr height/width ratio is similiar)
    - the exact height/width can be different - because people can be far/close to the camera. But aspect ratio on average should be the same.
    - Decide to standardize 82x36 pictures.
    - Collect large dataset of positive and negative examples (y=1 and y=0)
    - ALGORITHM:
        - Step 1
            - take a rectangular patch of the image. Maybe it could be in the left top corner.
            - slide rectangler a bit to right - train again
                - step size parameter / stride parameter - how far do we move the rectangle
                - stride=1 pixel works best
            - do that again and again, row by row, for the whole image.
        - Step 2
            - take larger patches and execute the same what we did in step1
        - Step 3-n:
            - take even larger patches and train algorithm with how we did that in step1
        - Result: hopefully, we get all pedestrians on the image.

- Return to text detection OCR:
    - get label set with positive (y=1) and negative (y=0) examples: where there is text (pos) and the regions where there is no text (neg).
        - so positive examples: patches with text
        - negative examples: patches without text
    - trian classifier
    - Apply classifier to an image:
        - running sliding windows classifier: NOT good, a lot of noise.
    - Apply expansion operator:
        - == take the image, take each of the white region and expand all of them
            - for every pixel, does it have white pixels in some distance nearby? 
            - if it does, color that pixel white in the rightmost image.
    - Finally: we can now look at contiguous white region
    - Ignore some blocks that shouldn't logically contain any text.

- You can apply sliding window approach to character segmentation:
    - Note: 1D sliding window for character segmentation:
        - only 1 row!
    - find out if there is split between 2 characters.
        - positive examples == middle of an image represents gap between 2 characters
        - negative examples == they don't represent mid point between 2 characters

# Getting lots of data and artificial data
- AndrewNG's thought: "I've seen over and over that one of the most reliable ways to get a high performance machine learning system is to take a low bias learning algorithm and train it on a massive training set".
- Artificial data synthesis - one way to get so much training data from for learning algorithm.
- Ideas:
    - we are essentially creating data fro m[xx] creating new data from scratch
    - we already have it's small label training set and we somehow have to amplify that training set or use a small training set to turn that into a larger training set.

- Let's use character recognition example
    - our current dataset: square patches with characters inside.
- One way to get more data: fonts
    - take fronts from your computer, segmentate the characters, use these new characters to retrain your algorithm.
        - don't forget to add a random background for the characters!
    - Result: synthetic data.
- Another approach: distortions
    - example: image of character "A"
    - apply artificial distrortions / warpings to the image to create N new examples

- Synthesis new data for speech recognition:
    - apply different audio distortions:
        - audio on bad cellphone connection
        - noisy background: crows
        - noisy background: machinery
        ...

- WARNING:
    - distortions introduced should be representation of the type of noise/distortions IN THE TEST SET!
        - example: audio - background noise, bad cellphone connection
    - Usually doesn't help to add purely random/meaningless noise to your data
        - example: unless we expect noisy data in test set, some random noise (ie Gaussian noise) wouldn't help
            - x = intensity (brightness) of pixel i
            - x_i = x_i + random_noise

# Discussion on getting more data using data synthesis
- Make sure you have a low bias classifier before expending the effort 
    - == plot learning curves
    - e.g. keep increasing the number of features / number of hidden units in neural network until you have a low bias classifier
    - So look at the algorithm - maybe all the additional data you gathered will be meaningles because of the algorithm's nature you use.
- Question to ask: "How much work would it be to get 10x as much data as we currently have?"
    - maybe it is really not that hard! So think before diving into data synthesis.
    - calculate how much hours it takes to get 1000, 10k, 1mln more data?

# Conclusion for ways to get more data
- Artificial data synthesis
- Collect and label it yourself
- Crowd sourcing
    - e.g. amazon mechanical turk

# Ceiling Analysis
- About character recognition task: what part of the pipeline to work next?
- What part of the pipeline should you spend the most time trying to improve it?

- Step 1
    - In order to make decisions, use some evaluation metric (like accuracy)
- Step 2
    - Go to first step of your pipeline.
    - Simulate the situation, when you have 100% of accuracy for that module on test set
    - Calculate accuracy for the entire system
- Step 3-n
    - Perform step2 for other modules of your pipeline
    - Example: mod1 100% accuracy, mod2 100% accuracy, mod3-modn NOT 100% accuracy
- Step n+1:
    - compare the accuracy values
    - what module has the most accuracy gain when it outputs 100% - work on that one!!!

- So in general Ceiling Analysis == "How much could you gain if some component is ideal?"
