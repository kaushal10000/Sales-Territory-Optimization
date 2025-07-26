import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def load_data(filepath="synthetic_sales_data.csv"):
    data = pd.read_csv(filepath, parse_dates=['CreatedDate', 'ClosedDate'])
    return data

def preprocess_data(data):
    # One-hot encode categorical variables needed for modeling
    features = ['CustomerSegment', 'Industry', 'Region', 'ProductCategory', 'DealSize', 'PreviousInteractions', 'LeadScore']
    data_model = pd.get_dummies(data[features], drop_first=True)
    return data_model

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

def add_propensity_scores(data, model, features):
    data_model = pd.get_dummies(data[features], drop_first=True)
    # Align columns if any mismatch between training and scoring data
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    if model_features is not None:
        missing_cols = set(model_features) - set(data_model.columns)
        for c in missing_cols:
            data_model[c] = 0
        data_model = data_model[model_features]
    data['PropensityScore'] = model.predict_proba(data_model)[:, 1]
    # Calculate optimization score = Propensity * Profit
    data['OptimizationScore'] = data['PropensityScore'] * data['Profit']
    return data

def profit_cube_analysis(data):
    # Multi-dimensional profit aggregation
    profit_by_region = data.groupby('Region')['Profit'].sum().reset_index()
    profit_by_product = data.groupby('ProductCategory')['Profit'].sum().reset_index()
    profit_by_segment = data.groupby('CustomerSegment')['Profit'].sum().reset_index()
    return profit_by_region, profit_by_product, profit_by_segment

def plot_dashboard(data, profit_by_region, profit_by_product, profit_by_segment):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Profit by Region (bar chart)
    sns.barplot(data=profit_by_region, x='Region', y='Profit', hue='Region', legend=False, ax=axes[0,0], palette='Blues_d')
    axes[0,0].set_title('Total Profit by Region')
    axes[0,0].set_ylabel('Profit')
    axes[0,0].tick_params(axis='x', rotation=0)
    
    # 2. Profit by Product Category (bar chart)
    sns.barplot(data=profit_by_product, x='ProductCategory', y='Profit', hue='ProductCategory', legend=False, ax=axes[0,1], palette='Greens_d')
    axes[0,1].set_title('Total Profit by Product Category')
    axes[0,1].set_ylabel('Profit')
    axes[0,1].tick_params(axis='x', rotation=0)
    
    # 3. Profit by Customer Segment (bar chart)
    sns.barplot(data=profit_by_segment, x='CustomerSegment', y='Profit', hue='CustomerSegment', legend=False, ax=axes[1,0], palette='Purples_d')
    axes[1,0].set_title('Total Profit by Customer Segment')
    axes[1,0].set_ylabel('Profit')
    axes[1,0].tick_params(axis='x', rotation=0)
    
    # 4. Top 10 Leads by Optimization Score (horizontal bar)
    top_leads = data.sort_values('OptimizationScore', ascending=False).head(10)
    top_leads = top_leads.sort_values('OptimizationScore')  # sort low to high for clean horizontal bars

    sns.barplot(
        x='OptimizationScore', 
        y='CustomerID', 
        data=top_leads, 
        ax=axes[1,1], 
        palette='Reds_r',
        hue='CustomerID', 
        legend=False
    )

    axes[1,1].set_title('Leads to Prioritize by Customer ID')
    axes[1,1].set_xlabel('Optimization Score')
    axes[1,1].set_ylabel('Customer ID')
    axes[1,1].tick_params(axis='y', labelsize=10)
    
    plt.tight_layout(pad=4.0)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig("sales_optimization.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Loading data...")
    data = load_data()
    
    print("Preprocessing data...")
    features = ['CustomerSegment', 'Industry', 'Region', 'ProductCategory', 'DealSize', 'PreviousInteractions', 'LeadScore']
    X = preprocess_data(data)
    y = data['Win']
    
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    model = train_model(X_train, y_train)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    print("Adding propensity scores and calculating optimization score...")
    data = add_propensity_scores(data, model, features)
    
    print("Performing Profit Cube analysis...")
    profit_by_region, profit_by_product, profit_by_segment = profit_cube_analysis(data)
    
    print("Generating dashboard...")
    plot_dashboard(data, profit_by_region, profit_by_product, profit_by_segment)
    
    print("\nTop 10 leads to prioritize:")
    top_10 = data.sort_values('OptimizationScore', ascending=False).head(10)[
        ['CustomerID', 'CustomerSegment', 'Region', 'ProductCategory', 'SalesRep', 'Profit', 'PropensityScore', 'OptimizationScore']]
    print(top_10.to_string(index=False))

if __name__ == "__main__":
    main()
