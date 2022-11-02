import pickle

import app as app
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
import os
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port, debug=True)


# create flask app
app = Flask(__name__)

# Load the pickle model
classification = pickle.load(open("classification.pkl", "rb"))
regression = pickle.load(open("regression.pkl", "rb"))
cluster = pickle.load(open("cluster.pkl", "rb"))


@app.route("/")
def Home():
    return render_template("home1.html")


@app.route('/discount')
def discount():
    return render_template("classification.html")


@app.route('/sales_target')
def sales_target():
    return render_template("regression.html")


@app.route("/product")
def product():
    return render_template("association.html")


@app.route("/analyze")
def analyze():
    return render_template("your_report.html")


@app.route('/loyal')
def loyal():
    # Load the csv file
    df = pd.read_csv("cluster-df.csv")
    df["cluster"] = cluster.labels_
    rslt_df_1 = df.loc[df['cluster'] == 0]
    rslt_df_2 = df.loc[df['cluster'] == 1]
    rslt_df_1 = rslt_df_1['CustomerID'].values.tolist()
    rslt_df_2 = rslt_df_2['CustomerID'].values.tolist()
    return render_template("cluster.html", cluster1=rslt_df_1, cluster2=rslt_df_2)


@app.route("/predict_discount", methods=["POST"])
def predict_discount():
    val_list = []
    for y in request.form.values():
        y_list = [y]
        le = LabelEncoder()
        y_new = le.fit_transform(y_list)
        val_list.append(float(y_new))
    features = [np.array(val_list)]
    prediction = classification.predict(features)
    return render_template("classification.html", prediction_text="Discount Status {}".format(prediction))


@app.route("/predict_sales_target", methods=["POST"])
def predict_sales_target():
    val_list = []
    for x in request.form.values():
        if x == 'Consumer':
            val_list.append(float(0))
        elif x == 'Corporate':
            val_list.append(float(1))
        elif x == 'Home Office':
            val_list.append(float(2))
        else:
            val_list.append(float(x))

    features = [np.array(val_list)]
    prediction = regression.predict(features)
    return render_template("regression.html", prediction_text="Target Quantity is {}".format(abs(prediction)))


# Import & Filter Data:
def data_filter(dataframe, state=False, State=""):
    if state:
        dataframe = dataframe[dataframe["State"] == State]
    return dataframe


# Invoice Product Matrix:
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['OrderID', 'ProductID'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['OrderID', 'ProductName'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


# Invoice Product Matrix:
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['OrderID', 'ProductID'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['OrderID', 'ProductName'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


# Find Product name with Product ID:
def check_id(dataframe, productid):
    product_name = dataframe[dataframe["ProductID"] == productid]["ProductName"].unique()[0]
    return productid, product_name


# Apriori Algorithm & Association Rules:
def apriori_algo(dataframe, support_val=0.001):
    from mlxtend.frequent_patterns import apriori, association_rules

    inv_pro_df = create_invoice_product_df(dataframe, id=True)
    frequent_itemsets = apriori(inv_pro_df, min_support=support_val, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=support_val)
    sorted_rules = rules.sort_values("support", ascending=False)
    return sorted_rules


def recommend_product(dataframe, product_id, support_val=0.001, num_of_products=5):
    sorted_rules = apriori_algo(dataframe, support_val)
    recommendation_list = []
    for idx, product_n in enumerate(sorted_rules["antecedents"]):
        for j in list(product_n):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[idx]["consequents"])[0])
                recommendation_list = list(dict.fromkeys(recommendation_list))
    return recommendation_list[0:num_of_products]


def recommendation_func(dataframe, product_id, support_val=0.001, num_of_products=5):
    if product_id in list(dataframe["ProductID"].astype("str").unique()):
        product_list = recommend_product(dataframe, product_id, support_val, num_of_products)
        if len(product_list) == 0:
            print("There is no product can be recommended!")
        else:
            for i in range(0, len(product_list[0:num_of_products])):
                print(check_id(dataframe, product_list[i]))

    else:
        print("Invalid Product Id, try again!")


@app.route("/product_recommend", methods=["POST"])
def product_recommend():
    list_output = []
    df = pd.read_csv("association.csv")
    x_list = []
    for x in request.form.values():
        x_list.append(x)
    print(x_list[0])
    list_output.append(recommendation_func(df, x_list[0]))
    print(list_output)
    return render_template("cluster.html", association=list_output)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
