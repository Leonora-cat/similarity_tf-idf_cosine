# Similarity_TF-IDF_CosineSimilarity
Using tf-tdf and cosine-similarity to calculate the similarity between 2 texts

## **Files and Purpose of this project**
### input files
11 files of different types of work orders (A)
2 files of work orders (B)
### purpose
for each work order in B, figure out which type in A this work order belongs to

## **Method**
For each work order in B, for each type in A, calculate their tf-idf weights and use these 2 weights as vectors, calculate the cosine similarity between these 2 vectors.

Each work order in B will calculate the cosine similarity with the 11 types in A, 11 cosine similarities in total. The type with the highest similarity is the type that this work order belongs to.

