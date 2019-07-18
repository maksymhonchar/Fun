
# https://nbviewer.jupyter.org/github/bensadeghi/julia-datascience-talk/blob/master/datascience-talk.ipynb#Background

# Hierarchical Built-In Types

@show subtypes(Number)

@show subtypes(Real)

@show subtypes(Integer)

@show subtypes(String)

;

@show 5/3  # Floating Point

@show pi  # mathematical constant

@show 2//3 + 1  # Rational

@show big(2) ^ 1000  # BigInt

;

s = "My own string"

@show typeof(s)
@show s[4]
@show s[:]

你好 = "(｡◕_◕｡)ﾉ  "  # Unicode Names and Values

@show typeof(你好)
@show 你好 ^ 3 ;

@show "String" * "concatenation"

;

# Custom user-defined types

struct NewType
    a::Float16
    b::Float32
    c::Float64
end

@show typeof(NewType)

newtype_obj = NewType(1.0, 2.0, 3.0)

@show typeof(newtype_obj)

@show typeof(newtype_obj.a), typeof(newtype_obj.b), typeof(newtype_obj.c)

;

using LinearAlgebra

# Vectors

v = [1, 1]

@show v + [2, 0]
@show v .+ 1
@show 5 * v

@show dot(v, v)  # using LinearAlgebra

@show norm(v)  # using LinearAlgebra

;

# Matrices

M = [1 1; 0 1]

@show M

@show M .+ 1

@show M + [0 0; 3 3]

@show 2M

@show M * M
@show M ^ 2

@show M * v

;

# Gaussian elimination (метод гаусса)

b = M * v

@show M \ b  # solve back for v

;

# Functions

f(x) = 10x

function g(x)
    return x * 10
end

@show f(5), g(5)

# Anonymous functions assigned to variables

h = x -> x * 10

i = function(x)
    x * 10
end

@show h(5), i(5)

;

# Operators are functions

@show +(4, 5)

plus = +
@show plus(4, 5)

# Multiple Dispatch

bar(x::String) = println("String is $x")
bar(x::Integer) = println("Integer*10 is $(x * 10)")
bar(x::NewType) = println("NewType: \"a\" is $(x.a)")

@show methods(bar)

bar("hello!")
bar("100")
bar(newtype_obj)

# Object-oriented programming

# Method overloading

struct SimpleStruct
    data::Union{Int64, String}
    set::Function
    
    function SimpleStruct()
        this = new()
        this.data = Unio # ?!
        
        function setter(x::Integer)
            this.data = x
        end
        
        function setter(x::String)
            this.data = x
        end
        
        this.set = setter
    end
    
    function ShowData()
        println("data is $(this.data)")
    end
end

so_1 = SimpleStruct()

;

# Functional programming

values_rng = 1:5

custom_mapper = x -> x + 1
even_filter = x -> x%2 == 0
sum_reducer = (x, y) -> x + y

@show map(custom_mapper, values_rng)

@show filter(even_filter, values_rng)

@show reduce(sum_reducer, values_rng)

;

# Metaprogramming

# Code generation
# Functions for exponentiating to the powers of 1 to 5

for n in 1:5
    s = "power$(n)(x) = x ^ $(n)"
    expression = Meta.parse(s)
    eval(expression)
end

@show power5(2)

;

# Macros

# Example: timer for timeit expression

macro custom_timeit(expression)
    quote
        t = time()
        result = $expression  # do the evaluation
        elapsed = time() - t
        println("Elapsed time:", elapsed)
        return result
    end
end

@custom_timeit cos(2pi)
@custom_timeit sin(2pi)

macro showln(value)
    quote
        show($value)
        println("")
    end
end

@showln("First value")
@showln("second value")

;

# Basic statistics

using Statistics

x = rand(100)  # uniform distribution [0, 1]

@show mean(x), var(x), skewness(x), kurtosis(x)

println("")

@show describe(x)

;

# Distributions

using Distributions

normal_distr = Normal(0, 2)

@show normal_distr

@show pdf(normal_distr, 0.0)

@show cdf(normal_distr, 0.0)

# fit a distribution to data using Maximum likelihood estimation
x_rand_uniform = rand(normal_distr, 1000)

@show fit_mle(Normal, x_rand_uniform)

;

# Tabular Data - DataFrames

using DataFrames

df = DataFrame(
    A = [1, 2, 3],
    B = ['a', 'b', 'c'],
    C = [1//2, 2//3, 3//4],
    D = [true, false, true]
)

@show df

df[1, :D] = false

@show df[!, [:C, :D]]

;

# Grouping by

using RDatasets

iris = dataset("datasets", "iris")

display(first(iris, 3))

display(
    by(iris, :Species, df -> mean(df[!, :PetalLength]))
)

# Data Visualization: ASCIIPlots

using UnicodePlots

iris_petal_len = iris[!, :PetalLength]
iris_petal_wid = iris[!, :PetalWidth]

display(
    scatterplot(iris_petal_len, iris_petal_wid)
)

# Data Visualization: Winston

using Winston

display(
    scatter(iris_petal_len, iris_petal_wid)

)

display(
    plot(iris_petal_len, iris_petal_wid)
)

# xlabel("PetalLength") - doesn't work
# ylabel("PetalWidth") - doesn't work

# Data Visualization: Gadfly

using Gadfly

set_default_plot_size(20cm, 12cm)

# plot(
#     iris,
#     x="PetalLength", y="PetalWidth",
#     color="Species",
#     Geom.point
# )

@doc plot

p = Gadfly.plot(
    iris,
    x=:SepalLength, y=:SepalWidth,
    color="Species"
)

p

# ML Algorithms: clustering

using Clustering

features = convert(Matrix, iris[:, 1:4])

n_clusters = 3
disp_lvl = Symbol(2)
kmeans_result = kmeans(features, n_clusters)

p = Gadfly.plot(
    iris,
    x=:PetalLength, y=:PetalWidth,
    color=kmeans_result.assignments,
    Geom.point
)

p

# ML Algorithms: PCA (Principal Component Analysis)

using MultivariateStats

pca_obj = fit(PCA, features; maxoutdim=2)

@show pca_obj

reduced = transform(pca_obj, features)

@show size(features), size(reduced)

p = Gadfly.plot(
    iris,
    x=reduced[1, :], y=[2, :],
    color="Species",
    Geom.point
)

# display(p)  # doesn't work

# ML Algorithms: Regression

using MultivariateStats

# Generate a noisy linear system
features = rand(1000, 3)  # feature matrix
coeffs = rand(3)  # ground truth of weights
targets = features * coeffs + 0.1 * randn(1000)  # generate response

# Linear least square regression
coeffs_llsq = llsq(features, targets; bias=false)

# Ridge regression
reg_coef = 0.1
coeffs_ridge = ridge(features, targets, reg_coef; bias=false)

@show coeffs

@show coeffs_llsq

@show coeffs_ridge

;

# Cross validation: k-fold

using MLBase

n = length(targets)
@show n

# Define training set indices function
function training(inds)
    coeffs = ridge(
        features[inds, :], targets[inds], 0.1; bias=false
    )
    return coeffs
end

# Define test error evaluation function
function eval_err(coeffs, inds)
    y = features[inds, :] * coeffs
    rms_error = sqrt(mean(abs2(targets[inds] .- y)))
end

# Cross validate
scores = cross_validate(
    inds -> training(inds),
    (coeffs, inds) -> eval_err(coeffs, inds),
    n,
    Kfold(n, 10)
)

@show scores

@show mean_and_std(scores)

;

n_test = convert(Int32, length(targets) * 2)
train_rows = shuffle([1:length(targets)] .> n_test)
features_train, features_test = features[train_rows, :], features[!train_rows, :]
targets_train, targets_test = targets[train_rows], targets[!train_rows]
