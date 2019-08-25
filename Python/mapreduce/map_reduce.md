# src
- https://courses.cs.washington.edu/courses/cse344/11au/sections/section8/section8-mapreduce-solution.pdf

# mapreduce
- map -> shuffle & sort -> reduce

# mapreduce
1. Input: set of (k,v) pairs
2. Run map function on each (k,v) pair, producing a bag of intermediate (k,v) pairs:
~~~~
map (in_key, in_value):
    // do some prorcessing on (in_key, in_value)

    emit_intermediate(hkey1, hvalue1)
    emit_intermediate(hkey2, hvalue2)
    // ...
~~~~
3. The MapReduce implementation groups the intermediate (key, value) pairs by the intermediate key
    - MapReduce grouping also outputs a bag containing all the values associated with each value of the grouping key
    - In addition, grouping is separated from aggregation computation, which goes in the reduce function
4. Run reduce function on each distinct intermediate key, along with a bag of all the values associated with that key. Result:
~~~~
reduce (hkey, hvalues):
    // do some preprocessing on hkey, each element of hvalues[]

    emit(fvalue1)
    emit(fvalue2)
    // ...
~~~~

# Example: selecting tuples from R where a<10
~~~~
map(inkey, invalue):
    if inkey<10:
        emit_intermediate(inkey, invalue)

reduce(hkey, hvalues[]):
    for each t in hvalues:
        emit(t)
~~~~

# Example: eliminate duplicates from R:
- Use the fact that duplicate elimination in the bag relational algebra is equivalent to grouping on all attributes of the relation.
- MapReduce does grouping for us, so all we need is to make the entire tuple the intermediate key to group on
~~~~
map(inkey, invalue):
    // We won't use the intermediate value, so we just put in a dummy value.
    emit_intermediate(invalue, 'abc')

reduce(hkey, hvalues[]):
    emit(hkey)
~~~~

# Example: how many of each rating type exist? stars range: [1-5], 5 options
- Algo:
    - MAP each input line to (rating, 1)
    - REDUCE each rating with the sum of all the 1's

~~~~
def mapper_get_ratings(self, _, line):
    (uid, mid, rating, timestamp) = line.split('\t')
    yield (rating, 1)

def reducer(self, hkey, hvalues):
    yield (hkey, sum(hvalues))
~~~~