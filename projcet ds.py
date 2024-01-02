#!/usr/bin/env python
# coding: utf-8

# # Introduction 
# 
# #In this project, we will talk about the effects of body performance: “12 columns and 13,394 rows.”
# #Among the factors we will talk about are age, gender, height, weight, fat percentage, diastolic, systolic, gribforce, sit and bend forward, sit-ups counts, broad jump, and class.
# #We will study the effect of each of these factors on the rest of the factors in some detail using (pandas, numpy, matplotlib)
# #Before the analysis of the dataset, data wrangling phase has been conducted to clean the data from unimportant columns, noisy data, and other problems. 
# #Before data wrangling phase, general properities about the dataset has been addressed.

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# # DATA WRANLING

# In[10]:


# Load dataset
df=pd.read_excel("Downloads/bodyPerformance.xlsx") 


# In[11]:


df


# In[12]:


# Display all information about columns on the dataset
print("\n Information of Df")
df.info()


# In[13]:


# First fifth rows of DF
print("\nHead of DF: \n", df.head(5))


# # DATA CLEANING

# In[14]:


# number of null data for each column
df.isna().sum()


# In[9]:


# fill gender column by using fillna function
clean_gender_column=df["gender"].fillna("no gender",inplace=True)


# In[10]:


# fill age column by using fillna function

clean_age_column=df["age"].mean()
df["age"].fillna(clean_age_column,inplace=True)


# In[11]:


# fill height_cm column by using fillna function
clean_height_cm_column=df["height_cm"].mean()
df["height_cm"].fillna(clean_height_cm_column,inplace=True)


# In[11]:


# fill weight_kg column by using fillna function
clean_weight_kg_column=df["weight_kg"].mean()
df["weight_kg"].fillna(clean_weight_kg_column,inplace=True)


# In[12]:


# fill body_fat% column by using fillna function
clean_bodyfat_column=df["body_fat"].mean()
df["body_fat"].fillna(clean_bodyfat_column,inplace=True)


# In[13]:


# fill diastolic column by using fillna function
clean_diastolic_column=df["diastolic"].mean()
df["diastolic"].fillna(clean_diastolic_column,inplace=True)


# In[14]:


# fill systolic column by using fillna function
clean_systolic_column=df["systolic"].mean()
df["systolic"].fillna(clean_systolic_column,inplace=True)


# In[15]:


# fill gripForce column by using fillna function
clean_gripForce_column=df["gripForce"].mean()
df["gripForce"].fillna(clean_gripForce_column,inplace=True)


# In[16]:


# fill sit_and_bend_forward_cm column by using fillna function
clean_sitandbendforward_cm_column=df["sit_and_bend_forward_cm"].mean()
df["sit_and_bend_forward_cm"].fillna(clean_sitandbendforward_cm_column,inplace=True)


# In[17]:


# fill sit_ups_counts column by using fillna function
clean_sit_upscounts_column=df["sit_ups_counts"].mean()
df["sit_ups_counts"].fillna(clean_sit_upscounts_column,inplace=True)


# In[18]:


# fill broad_jump_cm column by using fillna function
clean_broadjump_cm_column=df["broad_jump_cm"].mean()
df["broad_jump_cm"].fillna(clean_broadjump_cm_column,inplace=True)


# In[19]:


# fill class column by using fillna function
clean_class_column=df["class"].fillna("no class",inplace=True)


# In[20]:


df.isna().sum()


# In[ ]:


# remove noisy data 

# df.drop(df.index[df[""]])


#  

# In[21]:


# number of rows
f"number of row before remove duplicated : {df.shape[0]}"


# In[22]:


# check Rows Duplication
f"number of duplicated rows : {sum(df.duplicated())}"


# In[23]:


# remove duplicated rows
df.drop_duplicates(keep="first" , inplace=True)


# In[24]:


f"number of duplicated rows : {sum(df.duplicated())}"


# In[25]:


f"number of row after remove duplicated : {df.shape[0]}"


# In[26]:


# Display all information about columns on the dataset
df.info()


# # DATA ANALYSIS

# In[27]:


Fitness = (df['height_cm']-100) / df['weight_kg']

df.insert(4,'Fitness',Fitness)
df.head(13)


# In[18]:


print("""Descriptive Statistics
about df""")
print(df.describe())
print("""

From the above result 
we get some important
insights.

1-minimum age =21,

maxmum =64,

average=36.775594.

2-minimum height_cm =125,

maxmum = 193,

average=168.5567450.

3-minimum weight_kg =26.3,

maxmum = 138,

average=67.448310.

4-minimum weight_kg =0.532009,

maxmum = 2.194301,

average=1.031915.""")


# # DATA VISUALIZATION

# In[29]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2);
ax1.violinplot(df["body_fat"], showmedians=True)
ax1.set_title('body_fat')
ax1.set_xticks([1])
ax2.violinplot(df.age, showmedians=True)
ax2.set_title('age')
ax2.set_xticks([1])
plt.show()
#From the above graph, we get some important insights from the distribution of body_fat and distribution of age.body_fat is right skew .
#it seems to have no outlier. age seems to have outliers (as shown from maximum value of the violin plot) from the distribution. 
#age is left skew .the body_fat column in the dataset is more expressive about the movies more than age.


# In[30]:


sn.kdeplot(df["body_fat"], color='blue', shade='True');
plt.xlabel('body_fat', fontsize = 13)
plt.ylabel('performance', fontsize=13)
plt.title('fat Distribution', fontsize=13)
plt.show()
#The minimum body_fat value is 3, Average body_fat is about '23', Maximum body_fat is '53'. The plot is righ skew. Most movies exceeds the average body_fat.


# In[31]:


gender_df = df.assign(gender=df['gender'].str.split('|')).explode('gender')
gender_df['gender'].value_counts().plot(kind='bar', color='blue', title='Number of person for each gender'
                                         , xlabel='Gender Type', ylabel='Num of person');
#From the above graph: Male has the highest gender type.


# In[32]:


gb = gender_df.groupby('gender')['gripForce'].mean()

gb.plot(kind='barh', color='blue', title='Mean gripForce for each gender'
        , xlabel='gripForce', ylabel='Gender Type',figsize=(15,6));
#From the above, Male achieve the better gripforce.


# In[33]:


gender_df = df.assign(gender=df['class'].str.split('|')).explode('class')
gender_df['class'].value_counts().plot(kind='bar', color='blue', title='Number of person for each class'
                                         , xlabel='Gender Type', ylabel='Num of person');
#From the above graph: the classes has the equel gender type.


# In[34]:


gb = gender_df.groupby('class')['body_fat'].mean()

gb.plot(kind='barh', color='blue', title='Mean body_fat for each class'
        , xlabel='body_fat', ylabel='class Type',figsize=(15,6));
#From the above, class D achieve the better body_fat.


# In[35]:


df.plot(x= 'body_fat', y= 'gripForce' , kind='scatter', color = 'blue');
#From the above bar graph: there is a strong negative relationship between gripForce and body_fat. if the body_fat of a movie increases the gripForce decreases.


# In[36]:


df.plot(x= 'weight_kg', y= 'gripForce' , kind='scatter', color = 'blue');
#From the above bar graph: there is a strong positive relationship between weight_kg and gripForce. if the weight_kg of a movie increases the gripForce increases.


# In[37]:


df_ = df[['age','weight_kg','gripForce']]
corr_mat=df_.corr(method='pearson')
sn.heatmap(corr_mat, cmap='viridis' , annot = True)
plt.title('age_weight_kg_gripForce')
plt.show()
#From the above Heatmap: There is correlation between three variables: age, weight_kg, and gripForce. 
#correlation is computed for each two variable: There is strong positive correlation between gripForce and weight_kg.
#in addition there is a weak negative correlation between age and gripForce.


# In[38]:


df_ = df[['sit_and_bend_forward_cm','sit_ups_counts','broad_jump_cm']]
corr_mat=df_.corr(method='pearson')
sn.heatmap(corr_mat, cmap='viridis' , annot = True)
plt.title('sit_and_bend_forward_cm sit_ups_counts broad_jump_cm')
plt.show()

#From the above Heatmap: There is a positive correlation between three variables: sit_and_bend_forward_cm, sit_ups_counts, and broad_jump_cm. 
#correlation is computed for each two variable: There is strong positive correlation between sit_ups_counts and broad_jump_cm. 
#besides there is a string positive correlation between broad_jump_cm and sit_ups_counts (0.75). in addition there is a weak positive correlation between sit_and_bend_forward_cm and sit_ups_counts.


# In[48]:


df.plot.scatter(x="diastolic", y="systolic", c='gripForce', s=100);


# In[15]:


df.plot(subplots=True, figsize=(200, 200));


# In[16]:


df.plot.bar();


# In[ ]:


#END


# In[ ]:





# In[ ]:





# In[ ]:




