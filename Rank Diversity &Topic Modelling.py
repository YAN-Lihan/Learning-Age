#!/usr/bin/env python
# coding: utf-8

# ## Rank Diversity

# In[19]:


import pandas as pd
import os


# In[20]:


#文件路径
file_dir = r'C:\Users\11730\OneDrive - City University of Hong Kong\Desktop\datav3\datav3\爬虫\分小时数据'
#构建新的表格名称
new_filename = file_dir + '\\new_file.csv'
#找到文件路径下的所有表格名称，返回列表
file_list = os.listdir(file_dir)
new_list = []


# In[21]:


for file in file_list:
    #重构文件路径
    file_path = os.path.join(file_dir,file)
    #将excel转换成DataFrame
    dataframe = pd.read_excel(file_path)
    #保存到新列表中
    new_list.append(dataframe)


# In[22]:


df.head(5)


# In[23]:


#多个DataFrame合并为一个
df = pd.concat(new_list)


# In[24]:


df1=df.drop_duplicates(subset='热搜词',keep='first',inplace=False)


# In[25]:


df1.head(5)


# In[26]:


df2=df1[['排名','热搜词']]


# In[27]:


#写入到一个新csv表中
df2.to_csv(new_filename,index=False,encoding='gb18030')


# In[52]:


df3=df2.groupby(['排名']).size()


# In[54]:


print(df2['排名'].value_counts())


# In[58]:


df3=pd.DataFrame(df2['排名'].value_counts())


# In[61]:


df3


# In[63]:


df3=df3.reset_index()


# In[64]:


df3


# In[65]:


df4=df3.rename(columns={'index':'Rank','排名':'Rank_Diversity'})


# In[66]:


df4


# In[67]:


df4.sort_values(by='Rank',inplace=True,ascending=True)


# In[68]:


df4


# In[70]:


import matplotlib.pyplot as plt
import numpy as np


# In[71]:


df4.plot('Rank','Rank_Diversity',kind='scatter')


# ## Topic modelling

# In[103]:


df6=df1[['热搜词']]


# In[104]:


new_filename2 = file_dir + '\\new_file2.txt'


# In[105]:


df6.to_csv(new_filename2,sep='\t',index=False,encoding='gb18030')


# In[106]:


infile = open(new_filename2, 'r',encoding='gb18030')


# In[94]:


lines = infile.readlines()

# remove \n at the end of each line 
lines = [l.strip() for l in lines]

# remove empty lines
lines = [l for l in lines if l != ""]

# display the number of lines in the text file
# note: each line is treated as a sole document
len(lines)


# In[95]:


from sklearn.feature_extraction.text import CountVectorizer


# In[97]:


tf_vectorizer = CountVectorizer(stop_words=frozenset(["的", "在","我"]))
dtm_epl = tf_vectorizer.fit_transform(lines)


# In[99]:


dtm_epl


# In[118]:


from sklearn.decomposition import LatentDirichletAllocation

# train a latent dirichlet allocation model with number of topics = 3
lda = LatentDirichletAllocation(n_components=10, random_state=0) #random=0没神魔关系 只是一个stand，just give it a number

# fit the dtm into the lda object
lda.fit(dtm_epl)


# In[119]:


# get the topic-word(term) association for the LDA object
topic_word_matrix = lda.components_

# retrieve top n_top_words words for each topic
no_top_words = 5
feature_names = tf_vectorizer.get_feature_names()

# create a dataframe for displaying the results
rows = []

for topic_id, topic in enumerate(topic_word_matrix):
    row = ["Topic #" + str(topic_id) + ":"]
    row += [
        feature_names[i] + "*" + str(np.round(topic[i] / np.sum(topic), 4))
        for i in topic.argsort()[:-no_top_words - 1:-1]
    ]
    rows.append(row)

topic_word_df = pd.DataFrame(rows, columns=['Topic', 'Top 1 Word*Prob', 'Top 2 Word*Prob',                                             'Top 3 Word*Prob', 'Top 4 Word*Prob', 'Top 5 Word*Prob'])

topic_word_df


# In[ ]:




