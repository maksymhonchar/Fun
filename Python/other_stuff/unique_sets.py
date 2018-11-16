import random

min_question_id = 1
max_question_id = 60
variants_to_create = 30
questions_in_variant = 3

variants = map(
    lambda _: random.sample(range(min_question_id, max_question_id), questions_in_variant),
    range(variants_to_create)
)

print(list(variants))
