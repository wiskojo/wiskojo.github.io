---
title: "Visualizing Reddit Sentiments with Emojis"
---

## Introduction and Background
***

<p align="center"> 
<img src="https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/320px-Reddit_logo_and_wordmark.svg.png">
</p>

### Overview
Reddit is a social news aggregation site and discussion board that houses many interactive communities. These communities, termed "subreddits", each have a particular thematic focus that collectively spans many aspects of pop culture and beyond. In this study, we attempt to illuminate the sentimental nature of the discourses held within these various interconnected communities by means of visualizing their underlying communal sentiments. By the end of our study, we wish to realize a somewhat comprehensive map of Reddit that encapsulates the sentiments of some of its most popular subreddit communities.

### Background
In our endeavor to map the emotional contents of various subreddit communities, we will have to choose some form of sentiment analysis to conduct on each comment observation from our dataset. Sentiment analysis refers to the process of computationally extracting and quantifying the emotions, attitudes, and opinions expressed through textual data and is a specific subfield of the much broader discipline of natural language processing (NLP). Recent studies within this field into social media sentiments have successfully utilized a variety of noisy labels ranging from a handful of emoticons to assortments of hashtags as a form of distant supervision. Following this paradigmn, we will be incorporating a Deep Neural Network for sentiment classification that is detailed in a relatively recent paper published by Felbo et. al [1]. This work introduces an interesting approach to classifying textual data into sentimental categories of (64) relevant emojis. More specifically, the so-called "DeepMoji" model applies already established LSTM architectures to an extensive corpus of tweet-emoji labeled data to learn text-sentiment associations while also incorporating an original approach for transfer learning that the authors refer to as the "chain-thaw" method (which has been empirically shown by the authors to allow the model to generalize to more traditional sentiment analysis benchmarks). Taking inspiration from this work, we will apply the pretrained DeepMoji model towards the Reddit comments dataset in the hopes of being able to map subreddits into a low-dimensional embedding of sentiments. This clustering and visualization objective is similar to that of [2,3] in that we are trying to bring to light interesting relationships between various subreddit communities. In this case, however, we are less so interested about similarities in concrete subject matter and more so in the emotional connections/disparities between subreddits and of how these diverse communities encourage discussions that involve differential sentimental expressions.

*References*
- 1) https://arxiv.org/pdf/1708.00524.pdf and https://github.com/bfelbo/DeepMoji
- 2) https://peerj.com/articles/cs-4/
- 3) http://rhiever.github.io/redditviz/#

### Dataset
- Dataset Name: Reddit Comments Dataset
- Link to the dataset: https://www.reddit.com/r/datasets/comments/65o7py/updated_reddit_comment_dataset_as_torrents/
- Number of observations: Millions of comments

This is a dataset containing all Reddit comments posted from 2005-2017. We will be focusing on the latest available (2017) comment data, specifically for the month of March. The format of the file is a bz2 compressed JSON structure with the following column attributes (we will only be using the body, controversiality, score, and subreddit fields for this particular project):

- author
- author_cakeday
- author_flair_css_class
- author_flair_text
- body
- controversiality
- created_utc
- distinguished
- edited
- gilded
- id
- link_id
- parent_id
- retrieved_on
- score
- stickied
- subreddit
- subreddit_id

### Data Cleaning/Preprocessing
***
The Reddit comments dataset is very clean and well-structured as it is. We will not have much to do in terms of data cleaning and wrangling. That being said, the dataset is quite large and must be loaded in selectively by chunks. For simplicity of this particularl study, we define the top *n* subreddits by subscriber count that we want to be included in the analysis and load in only comments from those specific subreddits (in addition to any custom defined subreddits in the notebook paramaters). After all relevant comments are loaded into our Pandas dataframe, we will preprocess the data focusing on removing obvious observations that will not work well with our study and ensuring that the input comment bodies are formatted correctly for the DeepMoji model to analyze.

### Preliminary Data Preprocessing
In the context of sentiment analysis, we can imagine that certain types of comments are not conducive to analysis. Obviously, comments that have been removed or comments of deleted users should not be included in the study. Additionally, comments with links and subreddit references will not be well understood by our sentiment model due to its lack of contextual information and should be removed as well. Comments with CSS formatting and such should be sanitized of any formatting strings and finally, due to architectural and computational constraints, we will choose to remove any comments with character count greater than 300 or less than 10. Note that there is still a *lot* more that we can do with this in terms of preprocessing but we will go with this simple solution for now as to not complicate our study.

*Aside about the integrity of DeepMoji's outputs*
<br>
The reason for which we choose to remove all comments with character count > 300 is because DeepMoji was trained on Twitter data which has a similar character count limit. Any lengthy comment will not be representative of DeepMoji's training sample and thus may or may not end up just "confusing" the model. Regardless, I'm pretty sure DeepMoji's attention layer will focus in on particular keywords pretty early on in an input sentence anyways and thus very long comments will probably just drain precious computation time with very little benefits.

### Set up the DeepMoji Neural Network

*Model Description (from their Github page)*
<br>
"DeepMoji is a model trained on 1.2 billion tweets with emojis to understand how language is used to express emotions. Through transfer learning the model can obtain state-of-the-art performance on many emotion-related text modeling tasks."

If you want more details on the technical aspects of the model feel free to check out the corresponding white paper referenced at the beginning of this notebook. Omitting much of the complexity of the architecture, the model basically takes in a tokenized string input which it will analyze up to some predefined max length, and it will make a prediction as to what emoji(s) would best associate with the input text (as a normalized probability distribution over 64 emojis outputted by the final softmax layer--the exact emojis of which can be found below).

However, before the model can be used for predictive tasks, it needs to first be initialized with a vocabulary (and pretrained weights in this case). The vocabulary for the pretrained model is included in the model's Github page which we've copied over into the project directory and the pretrained weights can be downloaded from a dropbox link that is also provided on the Github page. Optimally, we would want to expand the vocabulary to include words that are found in our dataset but not in the original pretrained dataset, however, this would be a pain so we'll chose to overlook this complication also. Here we also configure the max (character) length that the model will analyze. For this study, we choose to use a max length of only 30 characters because of computational limitations. If one had access to better hardware, they can freely change the param_deepmoji_maxlen defined at the top of the notebook to analyze more characters from each comment.

![Emoji Overview](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/emoji_overview.png)

### Process Comments Through the DeepMoji Model
We now go through the various steps needed to batch analyze our comment data; this includes tokenization of comments pre-analysis and then afterwards populating the appropriate column attributes of our dataframe with the predictions and confidence outputted by DeepMoji post-analysis (in this case we choose to consider only the top 2 emojis predicted by the model along with their associated probabilities).

### Data Visualization
***
We can do some preliminary data visualization to get a sense of some of the communal sentiments that pervade various subreddits. That being said, this project is principally a data visualization endeavor and so we will have a lot more data visualization forthcoming in the more substantial *Data Analysis and Results* section that follows.

#### Visualize Top Predicted Emojis
We may be interested in which emojis are most commonly predicted by our DeepMoji model for comments from across all the relevant subreddits.

<iframe width="700" height="410" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/1.embed?link=false"></iframe>

#### Visualize Per-Subreddit Emoji Distribution
We may also be interested in the weighted and unweighted emoji distribution of each subreddit (the unweighted distribution is precisely the frequency of each emoji predicted by our DeepMoji model for all comments from that particular subreddit whereas the weighted distribution scales the frequency of the predicted emoji with the model's associated predictive confidence--the default configuration of this notebook will use the weighted distribution when conducting further analysis).

<iframe width="700" height="410" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/3.embed?link=false"></iframe>

<iframe width="700" height="410" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/5.embed?link=false"></iframe>

### Data Analysis and Results
***
We now aim to build up an arsenal of various useful components needed to construct a comprehensive sentimental map of Reddit; this is done largely by experimenting with various techniques for modelling, visualization, dimensionality reduction, clustering, and etc.

#### Load Each Subreddit's Emoji Distribution into a Dataframe
First, we summarize our emoji distribution data into a singular dataframe indexed by subreddit name for convenience of use; for each subreddit, the corresponding columns will embed the normalized weighted or unweighted emoji distribution, interpreted as a discrete probability mass function over all 64 emojis, of that particular subreddit.

#### Define Function to Generate Preliminary Mappings
Throughout the course of this study we will make a diverse display of sentiment mappings of Reddit that explores different clustering mechanisms and dimensionality reduction techniques; in an effort to reduce wholesale code copying, we will localize that functionality as a subroutine here in advance.

#### Construct Low Dimensional Embedding with PCA
When it comes to dimensionality reduction, PCA is really the first approach that comes to mind; it's simple and oftentimes effective. As you can see, we interpret each subreddit as a point in high dimensional (64-dimensional) space based on its emoji distribution and allow PCA to project those points onto an embedding along the two principal components. Quite expectedly however, due perhaps to the sparsity of the emoji distributions across subreddits, PCA seems to exagerate the variance along the first few principal components leaving behind an uninspiring embedding. We can definitely do better.

![png](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/output_37_0.png)

#### Construct Low Dimensional Embedding with t-SNE
At the suggestion of Shuai, we will instead use the t-SNE algorithm to do dimensionality reduction on our emoji distributions. Because the t-SNE approach operates in a manner that engages in preserving local neighborhoods, it proves robust against outliers and is quite appropriate for this sort of task. Note, however, that with this embedding the euclidean distance between points is no longer a good approximation of sentimental distance between those associated subreddits; i.e. unlike PCA, t-SNE is non-convex and while it will naturally attempt to preserve high dimensional cluster locality in lower dimensions, it makes no guarantees that distances between clusters within the lower dimensional embedding holds much meaning (the global geometry also seems to be quite substantially sensitive to the perplexity parameter). This makes t-SNE great for visualization but not so great for learning interpolatable, latent space representations (it's not made for that anyways I don't think).

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

An interesting thing to note here is how sports subreddits like r/nfl, r/nba, r/baseball, and r/sports are clustered together, however, r/soccer is not; this is perhaps an example of how interrelated subreddits with analogous subject matters, sports in this case, can still house discussions with diveregent sentimental atmospheres.

![png](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/output_41_0.png)

#### Spectral Clustering with RBF Kernel
Now that we have a workable embedding, we will attempt to label subreddit clusters. Our first attempt will be with the spectral clustering algorithm, specifically using the default RBF kernel to generate the underlying affinity matrix. This gives us respectable results and is able to formulate sensible clusters to the eye. However, we will later come to see that the spectral clustering technique, somewhat from rough empirical insight, is not so well suited for this particular application (especially in comparison to something like K-means).

![png](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/output_43_0.png)

<iframe width="700" height="700" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/17.embed?link=false"></iframe>

#### Calculate Pairwise Discrepancy of Emoji Distributions across Subreddits
Taking a peak at the underlying affinitiy matrix generated by the spectral clustering algorithm we may start to become curious as to what the sentimental distance between the various subreddits may look like; perhaps we can generate our own pre-computed affinity matrix with some metric that is more appropriate. Here is where JS divergence comes in. If we interpret each point representing a subreddit in 64-dimensional space as a probability distribution, we can find the *distance* between them by calculating the pairwise KL divergence between each subreddit; we will follow through with this line of reasoning but instead use JS distance instead of KL for purposes of symmetry. We can see that the generated JS matrix below is very similar to the affinity matrix above. In fact, we can convert our JS measure of *distance* into *affinity* by subtracting each element in the matrix from the max element of the JS matrix. This will generate an affinity matrix derived from JS distance that is very similar to the affinity matrix produced by spectral clustering with the RBF kernel. Perhaps this demonstrates some mathematical connection between the two methodologies but I'm not so sure at this point in time.

<iframe width="700" height="700" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/19.embed?link=false"></iframe>

#### Spectral Clustering with Affinity Matrix derived from J-S Divergence
We can feed our pre-computed affinity matrix to the spectral clustering algorithm and see that, expectedly, the clustering is very similar to our previous attempt.

![png](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/output_50_0.png)

#### K-Means Clustering
Falling back on more conventional methods, we attempt to generate a clustering with the K-means algorithm which actually proves to be rather effective. What isn't readily obvious is that spectral clustering's focus on graph connectivity makes it less suitable for this particular task and that K-means, which excels at finding globular clusters grouped by geometrical proximity, seems to be the more appropriate technique here.

![png](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/output_52_0.png)

#### Visualizing Strong Subreddit Sentimental Connections
Perhaps clusters don't tell the whole story; this thought urges us to pursue a different kind of visualization for our emoji distributions that would perhaps help elucidate the difference in results and effectiveness between applying K-means and spectral clustering. With this in mind, we generate a network graph for our data with subreddits as nodes and edges connecting between subreddits whose JS distance is lower than a predefined threshold (or rather in this case conversely that the pairwise affinity is higher than some predefined threshold--default 0.45). Interestingly enough, we see that even within some very obvious, strongly coupled, subreddit clusters, that the subreddits within those clusters don't have sufficient affinity to be connected. We can then deduce that t-SNE succesfully localizes these clusters as neighbors in the embedding due to their relative similarity; i.e. subreddits that are highly sentimentally idiosyncratic and are clustered together are not necessarily very similar to one another, but they are just so far removed from every other subreddit that they are forced into being neighbors. This could explain why spectral clustering, which focuses prominently on connectivity strength, fails to appropriately find these clusters and why on the other hand K-means, which focuses on relative geometric proximity, excels.

<iframe width="700" height="410" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/11.embed?link=false"></iframe>

#### Interpreting Embedding Regions with Modes of Low-Dimensional Cluster Sentiments
We are now interested in segmenting our embedding into distinct regions and assigning each an emoji which should represent that region's (of subreddits) collective sentiment. This is quite a difficult task with many open ended solutions. I imagine that one can possibly conceive an intricate autoencoder scheme that couples the task of learning a latent representation with that of predicting sentiments of collections of subreddits by constraining the cost function in some way. This is far too advanced and currently out of the scope of my skills and thus I will instead make a highly simplified attempt at interpreting the various regions of our generated t-SNE embedding. The strategy is this, first create local clusters within the lower dimensional embedding and aggregate the emoji distributions of each subreddit within that cluster into one collective, per cluster, emoji distribution. Then, simply take the mode (emoji) of the distribution and assign that emoji as the *collective sentiment* of that embedding region populated by the subreddits that constitute that cluster.

![png](https://raw.githubusercontent.com/wiskojo/wiskojo.github.io/master/resources/2018-04-02-visualizing-reddit-sentiments-with-emojis/output_56_0.png)

<iframe width="700" height="410" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/13.embed?link=false"></iframe>

### Putting It All Together
***
We've now examined various interesting aspects of our sentiment data, investigated a handful of neat techniques for data analysis, and successfully gathered up some useful results from our explorative endeavor. Assembling these diverse components together, we can now establish a somewhat polished *Reddit Sentiment Map* of sorts that is the culmination of our lighthearted scientific expedition.

<iframe width="700" height="410" frameborder="0" scrolling="no" src="//plot.ly/~wiskojo/15.embed?link=false"></iframe>

### Privacy/Ethics Consideration
***
Besides the obvious chronological bias in the data (that this chunk of Reddit comment data from March 2017 may not be wholly representative of all comments made over time), there is not much cause for concern in terms of privacy and ethical considerations--due to the comprehensive nature of the dataset, in that it equitably includes all comments made within a particular timeframe, there is no selective prejudice in terms of what comments are accessible to be analyzed. Additionally, the analysis focuses primarily on comment bodies and strips away any notion of authorship from each comment before it is processed and analyzed for sentimental value. Regardless, because these comments are public posts made online and users are anonymous by nature, we can safely assume that their privacy is not breached in this particular study. As for ethical considerations, we can imagine that the greatest extent of harm could come from mislabelling certain communities as promoting certain types of discourses and sentiments that they might not actually be. If such is the case, this is a fault of our combined methodologies, regardless, really no one should feel offended by the results of this rather playful analysis--what we offer is merely a *possible* representation of Reddit communal sentiments and any ridiculous assertion that this study aims to deface certain subreddit communities is wholly unfounded.

### Conclusions and Discussion
***
By exploration of various analytical methods, we've been able to construct a map of Reddit that visually illustrates the internal diversity of its communal sentiments; we've shown that the very intriguing DeepMoji model can be utilized to assist us with this construction, and that in combination with other powerful data analytic techniques, we are able to successfully carry out a lighthearted study that combines two amusing and somewhat outlandish aspects of modern culture, *Reddit and Emojis*.

This is a neat little project that could serve as a foray into more serious and rigorous studies but, alas, it really shouldn't be taken all too seriously in its current form. The principal foundation upon which excellent data analysis is carried out lies in the preprocessing, cleaning, and sampling of data. With regards to this, we've done an acceptable job of hacking everything to cooperate nicely together and produce somewhat meaningful results, yet at the same time we've really done a lackluster job of upholding the integrity of our study. To really tackle this idea with any degree of seriousness, which would require significantly more time than currently available, much more meticulous steps must be taken to ensure that our sampling strategy is robust, that the very critical DeepMoji model is able to incorporate analysis of comments that are much longer and more diverse, and that we have sufficient datapoints and computational power to extend and scale our study as necessary.

Regardless, there really is a lot of complexity to human sentiments and this study simply does not do it justice. Future works should especially look to focus on building a sentiment embedding that is interprettable; one that maintains significance in distances between points and clusters and one whose manifold regions are interpolatable and can be interpreted as gradients of shifting sentiments. As ridiculous as it may sound, we should also seek to develop a better understanding of emojis themselves--consider, for example, building visualizations of emoji cooccurrences to qualify the interactions and relationships between the underlying human sentiments embedded in these frivolous digital figures. For now though, this is probably as far as I'll be able to go with this silly little idea of *Reddit and Emojis*.
