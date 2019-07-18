
# src: http://ajkl.github.io/2015/08/10/Titanic-with-Julia/

# DataFrame introduction
# https://en.wikibooks.org/wiki/Introducing_Julia/DataFrames

# Julia documentation index.html
# https://docs.julialang.org/en/v1/manual/documentation/index.html

# Plots.jl
# https://github.com/JuliaPlots/Plots.jl
    # backends: http://docs.juliaplots.org/latest/backends/
        # PyPlot https://github.com/JuliaPy/PyPlot.jl
            # examples: https://docs.juliaplots.org/latest/examples/pyplot/

# Load libraries
using DataArrays
using DataFrames
using CSV
using FreqTables
using StatsBase

using Gadfly

# Snippet to use show() and println() in one line
showln(x) = (show(x); println())
showln("Libraries loaded")

# Load dataset

train_set = CSV.read("titanic/train.csv")
test_set = CSV.read("titanic/test.csv")

showln("Train and Test set loaded")

# Describe loaded train set
println("train_set head, 3 elements:")
showln(first(train_set, 3))
println("train_set tail, 3 elements:")
showln(last(train_set, 3))
println("train_set column names:")
showln(names(train_set))
println("train_set column types:")
showln(eltypes(train_set))
println("train_set shape:")
showln(size(train_set))
println("Describe train_set:")
show(describe(train_set), allcols=true)

# Describe loaded test set
println("test_set shape:")
showln(size(test_set))
println("Describe test_set:")
show(describe(test_set), allcols=true)

# Exploratory data analysis

# Explore unique values in every column

# Way 0: custom
function show_unique_values(dataset::DataFrame, max_unique_vals::Int64=10)
    for column_symbol in names(dataset)
        unique_col_values = unique(dataset[column_symbol])
        if size(unique_col_values, 1) < max_unique_vals
            println(column_symbol, ", unique values are ", unique_col_values)
        else
            println(column_symbol, " has too many unique values to display")
        end
    end
end

println("Showing unique values for train_set:")
show_unique_values(train_set)

println()

println("Showing unique values for test_set:")
show_unique_values(test_set)

# Way 1
survive_zero = count(i -> (i == 0), train_set[:Survived])
survive_one = count(i -> (i == 1), train_set[:Survived])
display(survive_zero)
display(survive_one)

# Way 2: "using FreqTables"
survive_freq_table = freqtable(train_set[:Survived])
display(survive_freq_table)

# Way 3: "using StatsBase"
display(countmap(train_set[:Survived]))

# Way 4: "using StatsBase"
display(counts(train_set[:Survived]))

# View proportions of people survived / not survived
display(proportions(train_set[:Survived]))
display(proportionmap(train_set[:Survived]))

# "counts" does not work for categorical variables. 
# Use "countmap()" instead
display(countmap(train_set[:Sex]))

# Create dimension: indicate if person was a child or not
train_set[:Child] = 1
train_set[isna(train_set[:Age]), :Child] = 1

train_set[train_set[:Age] .< 18, :Child] = 1
train_set[train_set[:Age] .> 18, :Child] = 0

display(head(train_set))

# NOTE: this notebook isn't finished
