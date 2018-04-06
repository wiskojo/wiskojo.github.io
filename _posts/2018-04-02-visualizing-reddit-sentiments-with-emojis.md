---
title: "Visualizing Reddit Sentiments with Emojis"
description: "An attempt at building a map of Reddit sentiments"
author: "Wis Kojohnjaratkul"
comments: true
---

## Introduction and Background
***

<p align="center"> 
<img src="https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/320px-Reddit_logo_and_wordmark.svg.png">
</p>

### Overview
Reddit is a social news aggregation site and discussion board that houses many interactive communities. These communities, termed *subreddits*, each have a particular thematic focus that collectively spans many aspects of pop culture and beyond. In this study, we attempt to explore the sentimental nature of these various interconnected communities by means of analyzing the user comments posted within them. Our end goal is to realize a reasonably intuitive map of Reddit that encapsulates the sentimental ties among some of its most popular subreddit communities.

### Background
In our endeavor to map the emotional contents of various subreddit communities, we will have to choose some form of sentiment analysis to conduct on each comment observation from our dataset. Sentiment analysis refers to the process of computationally extracting and quantifying the emotions and attitudes expressed through textual data and is a specific subfield of the much broader discipline of natural language processing (NLP). Recent studies within this field, particularly into social media sentiments, have successfully utilized a variety of noisy labels ranging from emoticons to hashtags as effective forms of distant supervision. Following this paradigmn, we will be incorporating a Deep Neural Network for sentiment classification developed by Felbo et. al which can be found on their [Github page](https://github.com/bfelbo/DeepMoji). This work introduces an interesting approach to classifying textual data into sentiment categories of (64) relevant emojis. More specifically, the so-called *DeepMoji* model applies the already well established LSTM architecture to an extensive corpus of tweet-emoji labeled data to learn text-sentiment associations while also incorporating an original approach for transfer learning (referred to by the authors as the "chain-thaw" method) which allows it to generalize to more conventional sentiment classification tasks. Taking inspiration from this work, we will apply the pretrained *DeepMoji* model towards the Reddit comments dataset in the hopes of being able to map some of the top subreddit communities into a low-dimensional embedding of sentiments. This visualization objective is reminiscent of previous works such as [this](https://peerj.com/articles/cs-4/) and [this](http://rhiever.github.io/redditviz/#) in that we are trying to bring to light interesting relationships between various subreddit communities. In this case, however, we are less so interested about similarities in concrete subject matter, and more so in the emotional connections/disparities between subreddits and of how these diverse communities encourage discussions that involve differential sentimental expressions.

### DeepMoji
Model Description (from their Github page)
<br>
> DeepMoji is a model trained on 1.2 billion tweets with emojis to understand how language is used to express emotions. Through transfer learning the model can obtain state-of-the-art performance on many emotion-related text modeling tasks.

If you want more details on the technical aspects of the model, feel free to check out the corresponding [white paper](https://arxiv.org/pdf/1708.00524.pdf). Omitting much of the complexity of the architecture, the model basically takes in tokenized string inputs, each for which it will analyze up to some predefined max length and then output a prediction as to what emoji(s) would best associate with the corresponding input text (given as a normalized probability distribution over 64 emojis produced by the final softmax layer, the exact emojis of which can be found below).

![Emoji Overview](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/emoji_overview.png)

### Our Data
For our dataset we will be using a small portion of the enormous [Reddit comments dataset](https://www.reddit.com/r/datasets/comments/65o7py/updated_reddit_comment_dataset_as_torrents/), specifically for the month of March 2017. From within this data subset, we are only interested in comments that come from the top 125 or so subreddits (give and take a handful of extra subreddits that I threw in the mix just for the sake of fun and personal curiosity). Furthermore, it is important to note also that we are only sampling from comments that fullfill certain criterias. For example, comments that have been removed or comments of deleted users are not considered. Likewise, comments with links and/or subreddit references will not be well understood by our sentiment model due to its lack of contextual information and are thus exempt from the study as well. Comments with special formatting and such are "sanitized" before they are analyzed and finally, due to architectural and computational constraints, we also choose to remove any comments with character count greater than 300 or less than 10. Note that there is certainly still a *lot* more that we can do with this in terms of preprocessing but we will choose to go with this simple solution for now as to not complicate our study. With this set of comments data ready (totalling around 3-4 million comments), we let *DeepMoji* do its thing and then we collect its sentiment predictions, in the form of emojis, for each comment observation. With that all set and done, we are now ready to move on to bigger and better things. 

### Data Visualization
***
At this point, we can already do some preliminary data visualization to get a sense of how redditors (a nickname for Reddit users) express themselves on Reddit. For example, we may be interested in which emojis are most commonly predicted by our DeepMoji model for comments from across all the relevant subreddits. From the chart below we can see that many people on Reddit, judging from the emotional contents of their posts, feel 😕 👍🏼 😳 and 😡.

<iframe width="700" height="410" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/1.embed?link=false"></iframe>

Perhaps we are also interested in what the emoji distribution of each individual subreddit looks like. Let's take a look then, for example, at what emojis are predicted for the very popular subreddit, r/AskReddit.

<iframe width="700" height="410" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/3.embed?link=false"></iframe>

For the sake of experimentation, we can also choose to consider weighted emoji distributions. The weighted distribution below, unlike the unweighted one above, scales the frequency of each comment-emoji prediction by the model's associated predictive confidence for each comment. Note that the distribution below is also for r/Askreddit but it definitely looks quite a bit different.

<iframe width="700" height="410" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/5.embed?link=false"></iframe>

### Explorative Analysis
***
After fooling around with our data for a little bit, we can now move on and actually begin to work towards our original goal of building a Reddit sentiment map of sorts. To this end, we will work step by step to compile a repertoire of useful analytical components that could help us with developing our map; this is done largely through experimentation with various techniques for modelling, visualization, dimensionality reduction, clustering, etc. Let's explore then, shall we?

#### PCA
When it comes to dimensionality reduction, [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) really is the first approach that comes to mind; it's simple and oftentimes very effective. As you can see below, we interpret each subreddit as a point in high (64) dimensional space based on its emoji distribution and allow PCA to project those points onto an embedding along the two principal components. Quite expectedly however, due perhaps to the sparsity of the emoji distributions across subreddits, PCA seems to exagerate the variance along the first few principal components leaving behind an uninspiring embedding. We can definitely do better.

![png](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/output_37_0.png)

#### t-SNE
We will instead opt to use the very famous [t-SNE algorithm](https://lvdmaaten.github.io/tsne/) to do dimensionality reduction on our sentiment data. Because the t-SNE approach operates in a manner that engages in preserving local neighborhoods, it proves robust against outliers and is quite appropriate for this sort of task. It is important to note, however, that with this more visually satisfying embedding, the euclidean distances between different points are no longer good approximations for sentimental distances between the associated subreddits; i.e. unlike PCA, t-SNE will naturally attempt to preserve high dimensional neighbors in lower dimensions but it makes no guarantees that the distances between clusters within the lower dimensional embedding holds much meaning (the global geometry also seems to be quite substantially sensitive to the perplexity parameter). This makes t-SNE great for visualization but not so great for learning latent vector space representations (it's not made for that anyways).

![png](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/output_39_0.png)

#### Highlights of Interesting t-SNE Clusters
Regardless, t-SNE in this case produces great results towards our end goal. Here, we manually highlight a few of the interesting clusters that show up within our embedding. Notably, subreddits from the following categories are featured:

- gaming
- sports
- fitness
- politics
- music
- nsfw
- misc
- food
- stem

An interesting thing to note here is how sports subreddits like r/nfl, r/nba, r/baseball, and r/sports are clustered together, however, r/soccer is not; this is perhaps an example of how interrelated subreddits with comparable subject matters, sports in this case, can still house discussions with divergent sentimental atmospheres.

![png](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/output_41_0.png)

#### Spectral Clustering with RBF Kernel
Now that we have a workable embedding, we will begin attempting to label subreddit clusters. Our first attempt will be with the [spectral clustering algorithm](https://en.wikipedia.org/wiki/Spectral_clustering), specifically using the default RBF kernel to generate the underlying affinity matrix for our data. This gives us respectable results and is able to formulate sensible clusters to the eye. However, we will later come to see that the spectral clustering technique, somewhat from rough empirical insight, is not so well suited for this particular application (especially in comparison to something like K-means).

![png](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/output_43_0.png)

<iframe width="700" height="700" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/17.embed?link=false"></iframe>

#### Subreddit Sentimental Discrepancy
Taking a peak at the underlying affinitiy matrix generated by the spectral clustering algorithm, we may start to become curious as to what the sentimental distances between the various subreddits may look like; perhaps we can generate our own pre-computed affinity matrix with some metric of distance that is more appropriate than the one implicitly given by the RBF kernel. Here is where [JS divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) comes in. If we interpret each point representing a subreddit in 64-dimensional space as a probability distribution, we can find the *distance* between them by calculating the pairwise KL divergence between each subreddit; we will follow through with this line of reasoning but use JS distance instead of KL for purposes of symmetry. We can see that the generated JS matrix below is very similar to the affinity matrix above. In fact, we can convert our JS measure of *distance* into *affinity* by subtracting each element in the matrix from the max element of the JS matrix. This will produce an affinity matrix derived from JS distance that is very similar to the affinity matrix generated by spectral clustering with the RBF kernel.

<iframe width="700" height="700" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/19.embed?link=false"></iframe>

We can feed our pre-computed affinity matrix to the spectral clustering algorithm and see that, expectedly, the clustering is very similar to our previous attempt.

![png](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/output_50_0.png)

#### K-Means Clustering
Falling back on more conventional methods, we attempt to generate a clustering with the [K-means algorithm](https://en.wikipedia.org/wiki/K-means_clustering) which actually proves to be rather effective. What isn't readily obvious is that spectral clustering's focus on graph connectivity makes it less suitable for this particular task and that K-means, which excels at finding globular clusters grouped by geometrical proximity, seems to be the more appropriate technique here.

![png](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/output_52_0.png)

#### Visualizing Networks of Subreddits with Strong Sentimental Connections
Perhaps clusters don't tell the whole story; this thought urges us to pursue a different kind of visualization for our sentiment data that would perhaps help elucidate the difference in results and effectiveness between applying K-means and spectral clustering. With this in mind, we generate a network graph for our data with subreddits as nodes and edges connecting between subreddits whose JS distance is lower than some predefined threshold. Interestingly enough, we see that even within some very obvious, strongly coupled, subreddit clusters, that the subreddits within those clusters don't have sufficient affinity to be connected. We can then perhaps deduce that t-SNE succesfully localizes these clusters as neighbors in the embedding due to their relative similarity; i.e. subreddits that are highly sentimentally idiosyncratic and are clustered together are not necessarily very similar to one another, but they are just so far removed from every other subreddit that they are forced into being neighbors. This could explain why spectral clustering, which focuses prominently on connectivity strength, fails to appropriately find these clusters and why on the other hand K-means, which focuses on relative geometric proximity, excels.

<iframe width="700" height="410" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/11.embed?link=false"></iframe>

#### A Simple Interpretation of Embedding Regions
We are now interested in segmenting our embedding into distinct regions and assigning each an emoji which should represent that region's (of subreddits) collective sentiment. This is quite a difficult task with many open ended solutions. For the sake of simplicity, we devise a highly basic attempt at interpreting the various regions of our generated t-SNE embedding. The strategy is this, first create local clusters within the lower dimensional embedding and aggregate the emoji distributions of each subreddit within that cluster into one collective, per cluster, emoji distribution. Then simply take the mode of the distribution and assign that emoji as the *collective sentiment* of the embedding region that is populated by the subreddits that constitute that cluster. Carrying out this process yields the following embedding interpretation.

<iframe width="700" height="410" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/13.embed?link=false"></iframe>

### Putting It All Together
***
We've now examined various interesting aspects of our sentiment data, investigated a handful of neat techniques for data analysis, and successfully gathered up some useful results from our explorative endeavor. Assembling these diverse components together, we can now establish a somewhat intuitive *Reddit Sentiment Map* of sorts that is the culmination of our lighthearted scientific expedition.

<iframe width="700" height="410" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/15.embed?link=false"></iframe>

### Conclusions and Discussion
***
By exploration of various analytical methods, we've been able to construct a map of Reddit that visually illustrates the internal diversity of its communal sentiments; we've shown that the very intriguing *DeepMoji* model can be utilized to assist us with this construction, and that in combination with the help of other powerful data analytic techniques, we were indeed able to successfully carry out a playful, yet in many ways imperfect, study that combines two amusing and somewhat quirky aspects of modern culture, *Reddit and Emojis*. Thanks for reading my first blog post. You can check the project out on [Github](https://github.com/wiskojo/Reddit-and-Emojis) if you're interested and any kind of feedback is valuable and very much welcomed.
