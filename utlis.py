import numpy as np
import pandas as pd
# spliting data
from sklearn.model_selection import train_test_split
# preprocessing and feature transformation 
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import PowerTransformer,OrdinalEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn_features.transformers import DataFrameSelector
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv("zomato.csv")
df.drop(columns=["url" , "name" , "reviews_list","phone","votes"],axis=1,inplace=True)
def handle_rate(rate):
    try:
        return rate.split("/")[0]
    except:
        return np.nan
df["rate"]=df["rate"].apply(handle_rate)
df.columns=df.columns.str.lower()
df["rate"]=pd.to_numeric(df["rate"],errors="coerce")
def create_traget(rate):
        if rate >=3.75:
         return "Yes"
        elif rate <3.75:
          return "No" 
        else:
            np.nan
df["target"]=df["rate"].apply(create_traget)
# drop rate 
df.drop(columns=["rate"],axis=1,inplace=True)
df.drop(columns=["dish_liked"],axis=1,inplace=True)
df.drop(index=df[df["target"].isna()].index.to_list(),axis=0,inplace=True)
df.drop_duplicates(inplace=True)
df.rename(columns={"listed_in(type)":"resturant_catogray"},inplace=True)
df.rename(columns={"listed_in(city)":"city"},inplace=True)
df["approx_cost"]=pd.to_numeric(df["approx_cost(for two people)"],errors="coerce")
df.drop(columns="approx_cost(for two people)",axis=1,inplace=True)
df.rename(columns={"address":"Delivery or rest"},inplace=True)
def handle_cuisines(value):
    try:
        return len(value.split(","))
    except:
        return np.nan

df["cuisines"]=df["cuisines"].apply(handle_cuisines)
def handle_menu_item(value):
    try:
        return len(value.split(","))
    except:
        return np.nan

df["menu_item"]=df["menu_item"].apply(handle_menu_item)
def handle_address(value):
    try:
        if value=="Delivery Only":
            return value
        else:
            return "Delivery and rest"
    except:
        return np.nan

df["Delivery or rest"]=df["Delivery or rest"].apply(handle_address)
def handle_rest_type(value):
    try:
        return len(value.split(","))
    except:
        return np.nan

df["rest_type"]=df["rest_type"].apply(handle_rest_type)
df.rename(columns={"rest_type":"#rest_types"},inplace=True)
df.rename(columns={"cuisines":"#cuisines"},inplace=True)
df.rename(columns={"menu_item":"#menu_items"},inplace=True)
X=df.drop("target",axis=1)
y=df["target"]
dict_target={
    "Yes":1,
    "No":0
}
y=y.map(dict_target)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.01,stratify=y,shuffle=True,random_state=42)
num_continous_cols=["approx_cost","#menu_items"]
num_discrete_cols=['#rest_types', '#cuisines']
cat_cols=X.select_dtypes(exclude="number").columns.to_list()


cat_pipe=Pipeline(steps=[
    ("selector",DataFrameSelector(cat_cols)),
    ("impute",SimpleImputer(strategy="most_frequent")),
    ("encoder",OrdinalEncoder())
])
num_continous_pipe=Pipeline(steps=[
    ("selector",DataFrameSelector(num_continous_cols)),
    ("impute",SimpleImputer(strategy="median")),
    ("tranform",StandardScaler()) 
])
num_discrete_pipe=Pipeline(steps=[
    ("selector",DataFrameSelector(num_discrete_cols)),
    ("impute",SimpleImputer(strategy="most_frequent")),
    ("tranform",StandardScaler()) 
])
all_pipe=FeatureUnion(transformer_list=[
    ("caterogical_pipline",cat_pipe),
    ("num_continous_pipe",num_continous_pipe),
    ("num_discrete_pipe",num_discrete_pipe),
])
_ =all_pipe.fit(X_train)




def process_new(X_new):
    ''' This Function is to apply the pipeline to user data. Taking a list.
    
    Args:
    *****
        (X_new: List) --> The users input as a list.

    Returns:
    *******
        (X_processed: 2D numpy array) --> The processed numpy array of userf input.
    '''
    
    ## To DataFrame
    df_new = pd.DataFrame([X_new])
    df_new.columns = X_train.columns

    ## Adjust the Datatypes
    df_new['Delivery or rest'] = df_new['Delivery or rest'].astype('str')
    df_new['online_order'] = df_new['online_order'].astype('str')
    df_new['book_table'] = df_new['book_table'].astype('str')
    df_new['location'] = df_new['location'].astype('str')
    df_new['#rest_types'] = df_new['#rest_types'].astype('float')
    df_new['#cuisines'] = df_new['#cuisines'].astype('float')
    df_new['#menu_items'] = df_new['#menu_items'].astype('float')
    df_new['resturant_catogray'] = df_new['resturant_catogray'].astype('str')
    df_new['city'] = df_new['city'].astype('str')
    df_new['approx_cost'] = df_new['approx_cost'].astype('float')
 



    ## Apply the pipeline
    X_processed = all_pipe.transform(df_new)


    return X_processed






