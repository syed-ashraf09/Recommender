import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

column_names=['user_id','item_id','rating','timestamp']
df=pd.read_csv("C:\\Users\\Ashraf\\Documents\\python-data-science-and-machine-learning-bootcamp-jose-portilla-master\\19-Recommender-Systems\\u.data",sep='\t',names=column_names)
print(df.head())
movie_titles=pd.read_csv("C:\\Users\\Ashraf\\Documents\\python-data-science-and-machine-learning-bootcamp-jose-portilla-master\\19-Recommender-Systems\\Movie_Id_Titles")
print(movie_titles.head())
df=pd.merge(df,movie_titles,on='item_id')#joining 2 datasets on item_id which is common on both datasets
print(df.head())

sns.set_style('white')
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
# to get avg of ratings for every title by grouping everything by title and ascending false to see the titles with best ratings, probably only 2 ppl hv rated a movie 5 stars and that doesnt help
df.groupby('title')['rating'].count().sort_values(ascending=False).head()#top 5 titles with most ratings
ratings=pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings']=pd.DataFrame(df.groupby('title')['rating'].count())
print(ratings.head())
ratings['num of ratings'].hist(bins=70)
plt.show()
ratings['rating'].hist(bins=70)
plt.show()
#dist relation btw avg rating and number of ratings
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
plt.show()

moviemat=df.pivot_table(index='user_id',columns='title',values='rating')
print(moviemat.head())
print(ratings.sort_values('num of ratings',ascending=False).head(10))
starwars_user_ratings=moviemat['Star Wars (1977)']
liarliar_user_ratings=moviemat['Liar Liar (1997)']
print(starwars_user_ratings.head())
similar_to_starwars=moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar=moviemat.corrwith(liarliar_user_ratings)
corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlation'])
corr_starwars.dropna(inplace=True)
print(corr_starwars.head())
#if we sort the dataframe by correlation we should get the movies that is most similar to starwars
print(corr_starwars.sort_values('correlation',ascending=False).head())
'''Here some random movies are shown that they are perfectly correlated to starwars, coz most likely these movies heppen ti be seen by only one person who also 
   happens to rate starwars. We can fix this by filtering out movies that has less than certain number of ratings. So we take a threshold 
   From the hist gram of number of ratings line(23), the series decline after 100, we wil take the threshold as 100'''
corr_starwars=corr_starwars.join(ratings['num of ratings'])
print(corr_starwars)
print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlation',ascending=False).head())

corr_liarliar=pd.DataFrame(similar_to_liarliar,columns=['correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar=corr_liarliar.join(ratings['num of ratings'])
print(corr_liarliar)
print(corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('correlation',ascending=False).head())