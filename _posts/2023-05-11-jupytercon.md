---
layout: post
title: JupyterCon 2023 talk on dirty-cat
date: 2023-07-20 16:11:00-0400
description: JupyterCon 2023 tutorial
tags: code software
categories: sample-posts
featured: true
---

If you weren't able to attend the JupyterCon in May and you're curious to learn more about machine learning with dirty tables, I'm happy to share the recording of my presentation!

<iframe width="560" height="315" src="https://www.youtube.com/embed/lvDN0wgTpeI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

***

Abstract:
Data scientists and analysts working with Jupyter are too often forced to deal with dirty data (with typos, abbreviations, duplicates, missing values...) that comes from various sources.

Let us step in the shoes of a data scientist, and with a Jupyter Notebook try to perform a classification or regression task on data coming from a collection of raw tables.

In this tutorial, we will demonstrate how dirty_cat, an open source Python package developed in our team, can help with table preparation for machine learning tasks and improve results of prediction tasks in the presence of dirty data.

#### Common problems we will be tackling:
<ul>
    <li>joining groups of tables on inexact matches;</li>
    <li>de-duplicating values;</li>
    <li>encoding dirty categories with interpretable results.</li>
</ul>

And all of this on dirty categorical columns that will be transformed into numerical arrays ready for machine learning.
