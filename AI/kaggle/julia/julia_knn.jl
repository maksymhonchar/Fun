
# guide src: https://www.kaggle.com/c/street-view-getting-started-with-julia/overview/knn-tutorial

# Main idea: implement k-Nearest Neighbor algorithm with Leave-One-Out-Fold
    # cross validation.

# Leave-One-Out-Fold-Cross-Validation (LOOF-CV) is similar to k-fold CV,
    # but k is set to be equal to the number of training points.
    # The model is tested on each individual data point after being
    # trained with the remaining points in the data.
    # Because LOOF-CV uses all but one of the data points for training,
    # it is not biased to any particular point or split, and could arguably give
    # a better estimation for the performance of the model. 

# todo: @time comparison:
#      transposed and NON transposed data
#      vectorized and for loop for euclidian distance

# Parallelization use
using Distributed

processes_cnt = 7
addprocs(processes_cnt)  # 8 cores

# Macro @everywhere - make function arguments, the definition of the functions
#    available to each process.
# Macro @distributed [reducer] - make a parallel for-loop

;

# Load packages
using Images
using DataFrames
using CSV
using DecisionTree(vcat)
using Statistics

# Helping function
showln(x) = (show(x); println())

showln("Libraries loaded")

;

# Load datasets (see julia_guide_kaggle.ipynb for details)
function read_image(dataset_type, labels_info, image_size, dataset_dir_path)
    # Initialize X matrix
    X = zeros(size(labels_info, 1), image_size)
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

image_size = 20 * 20
dataset_dir_path = "/home/max/Documents/ai/julia/julia_kaggle/data"

train_labels_info = CSV.read("$(dataset_dir_path)/trainLabels.csv")
test_labels_info = CSV.read("$(dataset_dir_path)/sampleSubmission.csv")
X_train = read_image("train", train_labels_info, image_size, dataset_dir_path)
X_test = read_image("test", test_labels_info, image_size, dataset_dir_path)
y_train = map(col_value -> col_value[1], train_labels_info[:Class])
y_train = convert(Array{Int64, 1}, y_train)

showln((size(X_train), size(X_test), size(y_train)))

showln("Loaded X_train, X_test, y_train")

;

# Prepare data for kNN algorithm

# Transpose X_train and X_test matrices
# In Julia, iteration over columns is faster than iteration over rows.
# Now, each column == one image file
X_train_tp = X_train'
X_test_tp = X_test'

showln((size(X_train_tp), size(X_test_tp)))

;

# Define distance function for kNN algorithm between 2 data points == 2 images

# NOTE: In Julia, 'for' loops can be faster than vectorized operations

@everywhere function euclidean_distance_vectorized(a, b)
    return dot(a-b, a-b)
end

@everywhere function euclidean_distance_forloop(a, b)
    distance = 0.0
    for i in 1:size(a, 1)
        distance += (a[i] - b[i]) * (a[i] - b[i])
    end
    return distance
end

showln("Created vectorized and forloop functions to calculate euclidean distance")

;

# Define kNN function

# NOTE: in Julia, creating an empty vector and filling it with each 
# element at a time is faster than copying the entire vector at once.

# NOTE: Since the code calculates the distance between the i data point
# and all the points in the training data, the closest point to
# the i point is itself with a distance of zero. Hence, we EXCLUDE it and
# select the next k points.

# Find k nearest neighbors of data point i
@everywhere function get_k_nearest_neighbors(X, i, k)
    n_rows, n_cols = size(X)
    ith_image = Array{Float32}(undef, n_rows)
    for i_idx in 1:n_rows
        ith_image[i_idx] = X[i_idx, i]
    end
    distances = Array{Float32}(undef, n_cols) # empty vector that will contain distances between i data point and each data point in the X matrix.
    jth_image = Array{Float32}(undef, n_rows)  # empty vector that will contain the j data point.
    for j_idx in 1:n_cols
        for i_idx in 1:n_rows
            jth_image[i_idx] = X[i_idx, j_idx]
        end
        # Calculate the distance and save the result
        distances[j_idx] = euclidean_distance_forloop(ith_image, jth_image)
    end
    sorted_neighbors = sortperm(distances)
    nearest_neighbors = sorted_neighbors[2:k+1]  # note: start with the 2nd one
    return nearest_neighbors
end

showln("Created function to get k nearest neighbors")

;

# Assign label to the ith point according to the labels of the k nearest neighbors

# training data: X matrix
# labels: y vector
@everywhere function assign_label(X, y, k, i)
    nearest_neighbors = get_k_nearest_neighbors(X, i, k)
    labels_counts = Dict{Int, Int}()
    highest_count = 0
    most_popular_label = 0
    for neighbor in nearest_neighbors
        neighbor_label = y[neighbor]
        if !haskey(labels_counts, neighbor_label)
            labels_counts[neighbor_label] = 0
        end
        labels_counts[neighbor_label] += 1
        if labels_counts[neighbor_label] > highest_count
            highest_count = labels_counts[neighbor_label]
            most_popular_label = neighbor_label
        end
    end
    return most_popular_label
end

showln("Created function to assign label using kNN")

;

# Assign label to each point in the training data
k = 3
y_predictions = @distributed (vcat) for i in 1:size(y_train, 1)
    assign_label(X_train_tp, y_train, k, i)
end

showln(size(y_predictions))

display(size(y_predictions))
display(size(y_train))

;

# Measure accuracy of the model by comparing our predictions with the true labels
loof_cv_accuracy = mean(y_predictions .== y_train)
showln("LOOF-CV accuracyy of 1-NN is $(loof_cv_accuracy)")  # k=1: 0.444; k=3: 0.445; k=5: 0.428

;
