---
layout: post
title: Using skrub
date: 2023-05-11 15:09:00
description: JupyterCon 2023 tutorial
tags: code
categories: sample-posts
featured: true
---
JupyterCon 2023 presentation tutorial: 
Machine learning with dirty tables: encoding, joining and deduplicating

````markdown
```python
# %% [markdown]
# # Installing *dirty_cat* from source

# %%
# From source:
# !git clone https://github.com/dirty-cat/dirty_cat
# !pip install ./dirty_cat

# From PyPi:
# !pip install dirty_cat

# %%
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# # Using the library with an example: employee salaries
# 
# We will load a dataset which contains information on more than 9000 employees from Montgomery County, Maryland:

# %%
from dirty_cat.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
# Aliases
X = dataset.X
y = dataset.y

# Pre-processing steps
X.drop(["underfilled_job_title", "department", "division"], axis="columns", inplace=True)

X.head()

# %% [markdown]
# Our goal will be to predict the annual salary using this information.

# %% [markdown]
# ## **1. Encoding dirty categorical variables**

# %% [markdown]
# ![Encoding](photos/encoding.png)

# %% [markdown]
# ## A problem of similarity

# %%
# Pick a sample with similar employee position titles
sample = X[X["employee_position_title"].str.contains("Fire|Social")].sample(n=10, random_state=50).head(10)
sample["employee_position_title"]

# %% [markdown]
# Let's see how `OneHotEncoder` behaves with those

# %%
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
s_enc_ohe = ohe.fit_transform(sample[["employee_position_title"]])

# Make it look nice in Jupyter by wrapping it in a DataFrame
pd.DataFrame(s_enc_ohe, columns=ohe.categories_[0], index=sample["employee_position_title"])

# %% [markdown]
# - OneHot gives equidistant encodings!
# - Dimensionality explodes
# - Rare categories (maybe unknown to test set)

# %% [markdown]
# # Similarity encoding: easy to understand

# %%
from dirty_cat import SimilarityEncoder

sim = SimilarityEncoder()
s_enc_sim = sim.fit_transform(sample[["employee_position_title"]])

pd.DataFrame(s_enc_sim, columns=sim.categories_[0], index=sample["employee_position_title"])

# %% [markdown]
# The similarity encoding on the other hand encodes the similarities between each category.

# %% [markdown]
# How? Using the **n-gram similarity**:
# 
# ![Encoding](photos/ngram.png)
# 
# - $Similarity=\frac{\text{# n-grams in common}}{\text{# n-grams in total}}$
# - Based on substring comparison.
# - Faster than Levenshtein or Jaro-Winkler with better results.

# %% [markdown]
# In conclusion:
# 
# The `SimilarityEncoder` is extending the OHE logic based on the n-gram morphological similarity.

# %% [markdown]
# # Gamma-Poisson encoding: by topics and interpretable

# %%
from dirty_cat import GapEncoder

gap = GapEncoder(n_components=10, random_state=0)

pos_enc = gap.fit_transform(X[["employee_position_title"]])
print(f"Shape of encoded vectors: {pos_enc.shape}")

# %%
# We can print the labels that were infered for each topic
topic_labels = gap.get_feature_names_out(n_labels=3)
for k, labels in enumerate(topic_labels):
    print(f"Topic n°{k}: {labels}")

# %%
import matplotlib.pyplot as plt

encoded_labels = gap.transform(sample[["employee_position_title"]])
plt.figure(figsize=(8, 6))
plt.imshow(encoded_labels)
plt.xlabel("Latent topics", size=12)
plt.xticks(range(0, 10), labels=topic_labels, rotation=50, ha="right")
plt.ylabel("Data entries", size=12)
plt.yticks(range(0, 10), labels=sample[["employee_position_title"]].to_numpy().flatten())
plt.colorbar().set_label(label="Topic activations", size=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# In conclusion:
# 
# **The `GapEncoder` extracts latent topics from categories and uses them to evaluate their similarity.**

# %% [markdown]
# # Min-hash encoding: very scalable

# %%
from dirty_cat import MinHashEncoder

minhash = MinHashEncoder()
minhash_enc = minhash.fit_transform(sample[["employee_position_title"]])

print(minhash_enc)

# %% [markdown]
# The resulting encoded category will be the intersection of its components. 
# 
# ![MinHashEncoder](photos/minhash2.png)
# 
# Source: *P.Cerda, G.Varoquaux. Encoding high-cardinality string categorical variables (2019)*

# %% [markdown]
# **CCL: The `MinHashEncoder` is an extremely efficient encoding method based on the minhash function.**

# %% [markdown]
# # Comparing encoding methods
# 
# We'll run a pipeline with each encoding method we just saw, and a learner, here a `HistGradientBoostingRegressor`.

# %%
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit

all_scores = dict()
all_times = dict()

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

for method in [
    ohe,
    SimilarityEncoder(),
    GapEncoder(n_components=50),
    MinHashEncoder(n_components=100),
]:
    name = method.__class__.__name__
    encoder = make_column_transformer(
        (ohe, ["gender", "department_name", "assignment_category"]),
        ("passthrough", ["year_first_hired"]),
        (method, ["employee_position_title"]),
        remainder="drop",
    )

    pipeline = make_pipeline(encoder, HistGradientBoostingRegressor())
    results = cross_validate(pipeline, X, y, cv=ShuffleSplit(n_splits=3, random_state=0))
    scores = results["test_score"]
    times = results["fit_time"]
    all_times[name] = times
    all_scores[name] = scores

# %%
from seaborn import boxplot

_, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6))

ax = boxplot(data=pd.DataFrame(all_scores), orient="h", ax=ax1)
ax1.set_xlabel("Prediction score (R2)", size=20)
[t.set(size=20) for t in ax1.get_yticklabels()]


boxplot(data=pd.DataFrame(all_times), orient="h", ax=ax2)
ax2.set_xlabel("Computation time (s)", size=20)
[t.set(size=20) for t in ax2.get_yticklabels()]

plt.tight_layout()

# %% [markdown]
# # Automating the boring stuff with the `TableVectorizer`
# 
# Typically, when we want to assemble different encoders for our dataset, we'll use the `ColumnTransformer`:

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
from dirty_cat import TableVectorizer

pipeline_tv = make_pipeline(
    TableVectorizer(),
    HistGradientBoostingRegressor(),
)

# %%
pipeline_tv.fit(X_train, y_train)
print("TableVectorizer pipeline score:", pipeline_tv.score(X_test, y_test))

# %% [markdown]
# What does it do? Let's see:

# %%
from pprint import pprint  # pretty print
pprint(pipeline_tv[0].transformers_)

# %% [markdown]
# It recognized and used the datetimes (`date_first_hired`) automatically, and encoded it with `dirty_cat.DatetimeEncoder()` 😉

# %% [markdown]
# The `TableVectorizer` is simpler to use than `ColumnTransformer`, and works out of the box for most dirty tables.
# It can be easily customized and supports numerical, categorical and datetime features.

# %% [markdown]
# Conclusion:
# - The `TableVectorizer` is best to use as it's automatically taking care of the encoding choices.

# %% [markdown]
# # **2. Fuzzy joining tables with dirty data**

# %% [markdown]
# ![fj](photos/fj.png)

# %% [markdown]
# ## Better `pandas.merge`: `fuzzy_join`

# %%
baltimore = pd.read_csv("https://data.baltimorecity.gov/datasets/baltimore::baltimore-city-employee-salaries.csv")[["agencyName", "jobClass", "annualSalary"]]
baltimore = baltimore.groupby(by=["agencyName", "jobClass"]).mean().reset_index()
baltimore.tail()

# %%
pd.merge(X, baltimore, left_on='employee_position_title', right_on='jobClass')

# %%
from dirty_cat import fuzzy_join

X2 = fuzzy_join(X, baltimore, left_on='employee_position_title', right_on='jobClass', return_score=True)
X2.head()

# %%
X2[["employee_position_title", "jobClass", "matching_score"]].sort_values("matching_score").head(10)

# %%
X2_bis = fuzzy_join(X, baltimore, left_on='employee_position_title', right_on='jobClass', match_score=0.6)
X2_bis.head()

# %% [markdown]
# # Automating the boring stuff: multiple `fuzzy_join`'s with `FeatureAugmenter`

# %% [markdown]
# Case of a datalake: often the case in real production settings (big companies or public institutions).
# 
# You need to join multiple tables on the initial one to add information (feature augmentation).

# %%
population = pd.read_csv("https://opendata.maryland.gov/api/views/sk8g-4e43/rows.csv?accessType=DOWNLOAD")
population.tail()

# %%
minimum_wage = pd.read_csv("https://raw.githubusercontent.com/Lislejoem/Minimum-Wage-by-State-1968-to-2020/master/Minimum%20Wage%20Data.csv", encoding='latin')
minimum_wage = minimum_wage[minimum_wage["State"] == 'Maryland'][["State.Minimum.Wage.2020.Dollars", "Year", "State"]]
minimum_wage.head()

# %% [markdown]
# Repeating `fuzzy_join`'s over and over for each new table tables is painful:

# %% [markdown]
# We now have a class we can introduce in our ML pipeline!

# %%
from dirty_cat import FeatureAugmenter

faugmenter = FeatureAugmenter(tables=[
        (population, "Year"),
        (minimum_wage, "Year"),
    ],
    main_key="year_first_hired",
)

# %%
pipeline_fj = make_pipeline(
    faugmenter,
    TableVectorizer(),
    HistGradientBoostingRegressor(),
)

pipeline_fj.fit(X2_train, y2_train)

print("TableVectorizer with fuzzy_join pipeline score:", pipeline_fj.score(X2_test, y2_test))

# %% [markdown]
# Conclusion:
# - The `fuzzy_join` is a function that allows you to join two tables on imprecise correspondences. It is based on the n-gram morphological similarity of categories.
# - The `FeatureAugmenter` can do this on multiple tables on a common join key. scikit-learn compatible, can be used in a pipeline.

# %% [markdown]
# # **3. Deduplicating dirty categorical variables**

# %% [markdown]
# ![deduplicating](photos/deduplicated.png)

# %% [markdown]
# ## Clean typos from your data with deduplication

# %%
from dirty_cat import deduplicate
deduplicate(sample["employee_position_title"])

# %% [markdown]
# Good for getting back into the OHE use case. Beware of potential losses of information.
```
````
