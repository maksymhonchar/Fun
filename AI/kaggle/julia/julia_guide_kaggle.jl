
# Kaggle tutorial:
# https://www.kaggle.com/c/street-view-getting-started-with-julia/overview/julia-tutorial

# Load packages

using Images
using DataFrames
using CSV
using DecisionTree
using Statistics

;

"""
dataset_type:
    "train" or "test"
labels_info:
    IDs of each image to read
image_size:
    amount of pixels in image
dataset_dir_path:
    path to directory with "trainResized" and "testResized" directories.
"""
function read_image(dataset_type, labels_info, image_size, dataset_dir_path)
    # Initialize X matrix
    X = zeros(size(labels_info, 1), image_size)
    @show size(labels_info)
    for (idx, image_id) in enumerate(labels_info[:ID])
        # Load image
        image_full_path = "$(dataset_dir_path)/$(dataset_type)Resized/$(image_id).Bmp"
        image = load(image_full_path)
        image_greyscale = Gray.(image)
        # Transform image matrix to a vector and store it in data matrix
        X[idx, :] = reshape(image_greyscale, 1, image_size)
    end
    return X
end

;

# The images in the trainResized and testResized directories are 20x20 pixels,
# so image_size should be set to 400.

image_size = 20 * 20
dataset_dir_path = "/home/max/Documents/ai/julia/julia_kaggle/data"

# Load labels (ID and Class) for train and test 

# train_set
train_labels_info = CSV.read("$(dataset_dir_path)/trainLabels.csv")
@show size(train_labels_info)
@show names(train_labels_info)

# test_set
test_labels_info = CSV.read("$(dataset_dir_path)/sampleSubmission.csv")
@show size(test_labels_info)
@show names(test_labels_info)

# Show some info about these labels
@show first(train_labels_info, 5)
@show first(test_labels_info, 5)

;

# Read training and testing sets
@time X_train = read_image("train", train_labels_info, image_size, dataset_dir_path)
@time X_test = read_image("test", test_labels_info, image_size, dataset_dir_path)

@show size(X_train)
@show size(X_test)

;

# Create Y vector from each image.
# Target variable == first character of 'Class' column

y_train = map(col_value -> col_value[1], train_labels_info[:Class])
y_train = convert(Array{Int64, 1}, y_train)

@show size(y_train)
@show typeof(y_train)
@show y_train[1:10]

;

# Create a training model

# Learn patterns in the images that identify the character in the label.
# Use Julia version of Random Forest algorithm.
# Parameters to set:
#     number of features to choose at each split
#         sqrt(number_of_features)
#     number of trees
#         larger is better, but it takes more time to train
#     ratio of sampling
#         usually 1.0

num_of_features = sqrt(size(X_train, 2))
num_of_trees = 50
ratio_of_sampling = 1.0

# build_forest(labels::Array{T,1}, features::Array{S,2}, n_subfeatures, n_trees, partial_sampling)

model = build_forest(
    y_train, X_train,  # labels, features
    num_of_features,
    num_of_trees,
    ratio_of_sampling
)

@show model

;

# Use trained model: identify the characters in the test data

test_set_predictions = apply_forest(model, X_test)

;

# Convert predictions back to characters
test_labels_info[:Class] = convert(Array{Char, 1}, test_set_predictions)

;

# Print model accuracy

n_folds = 4

# nfoldCV_forest(
#     labels::Array{T,1}, features::Array{S,2},
#     n_folds::Integer, n_subfeatures::Integer, n_trees::Integer, partial_sampling::Float64,
# )

accuracy = nfoldCV_forest(
    y_train, X_train,  # labels, features
    n_folds,
    convert(Int, num_of_features),
    num_of_trees,
    ratio_of_sampling
)

@show mean(accuracy)

;

# Save predictions

CSV.write(
    "$(dataset_dir_path)/julia_submission.csv",
    test_labels_info,
    writeheader=true
)

;
