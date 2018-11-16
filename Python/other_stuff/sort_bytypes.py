dishes = {
    "dish3": {
        "type": "type1",
        "weight": "200/50",
        "price": "92.00"
    },
    "dish2": {
        "type": "type2",
        "weight": 123,
        "price": 123.0
    },
    "dish1": {
        "type": "type3",
        "weight": "200/50",
        "price": "70.00"
    },
    "dish4": {
        "type": "type4",
        "weight": "200/50",
        "price": "70.00"
    },
    "dish5": {
        "type": "type1",
        "weight": "200/50",
        "price": "70.00"
    }
}

# Empty dictionary to hold dishes, separated by types
types = {}

# 1. Separate types into dictionary {type1:{}, type2:{}, type3:{}, ...}
for dish in dishes:
    # [dish] here is a key (string).
    specific_type = dishes[dish]['type']
    types[specific_type] = {}

# 2. Add dishes to the types dicitionary.
for dish in dishes:
    # [dish] here is a key (string).
    specific_type = dishes[dish]['type']
    specific_dish = {
        dish: dishes[dish]
    }
    types[specific_type].update(specific_dish)

"""
types = {
    type1: {
        dish1: { dish_info___as_a_dictionary }
        dish2: { dish_info___as_a_dictionary }
        ...
    },
    type2: {
        ...
    },
    ...
}
"""
# Print the result!
print(types)
