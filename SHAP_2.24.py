import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap
from catboost import CatBoostRegressor
import os
from datetime import datetime
matplotlib.use('Agg')

# Reading data
file_path = "simulation_data.xlsx"
data = pd.read_excel(file_path)

# Create a new folder for storing the results
data_file_name = os.path.splitext(os.path.basename(file_path))[0]
current_time = datetime.now().strftime("%m%d_%H%M%S")
folder_path = f"SHAP_{data_file_name}_{current_time}"
os.makedirs(folder_path, exist_ok=True)


def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    LF = Q1 - 1.5 * IQR
    UF = Q3 + 1.5 * IQR
    data = data[~((data < LF) | (data > UF)).any(axis=1)]
    return data

data = remove_outliers(data)

X = data[['Pressure', 'Power', 'Spacing', 'Depo_time', 'SiH4', 'NH3', 'N2', 'H2']].values
y = data['rate'].values

feature_mapping = {
    'Pressure': 'Chamber Pressure',
    'Power': 'RF Power',
    'Spacing': 'Electrode Spacing',
    'Depo_time': 'Deposition Time',
    'SiH4': 'SiH4 Flux',
    'NH3': 'NH3 Flux',
    'N2': 'N2 Flux',
    'H2': 'H2 Flux'
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

fit1 = CatBoostRegressor(
    iterations=269,
    depth=3,
    learning_rate=0.3226,
    random_state=0,
    silent=True)

fit1.fit(X_train, y_train)

pred1_train = fit1.predict(X_train)
pred1_test = fit1.predict(X_test)

# ------------------------
# SHAP Analysis
# ------------------------

# Calculation of SHAP value
explainer = shap.Explainer(fit1)
shap_values = explainer(X_test)

full_feature_names = [feature_mapping[feat] for feat in ['Pressure', 'Power', 'Spacing', 'Depo_time',
                                                         'SiH4', 'NH3', 'N2', 'H2']]
shap_values.feature_names = full_feature_names

# Feature Importance Bar Plot
shap.plots.bar(shap_values, max_display=10)
plt.savefig(os.path.join(folder_path, "SHAP_Bar_Plot.png"))
bar_data = pd.DataFrame({
    'Feature': shap_values.feature_names,
    'Mean |SHAP| Value': np.abs(shap_values.values).mean(axis=0)
})
bar_data.to_csv(os.path.join(folder_path, "SHAP_Bar_Plot_data.csv"), index=False)
plt.clf()

# Summary Plot
shap.summary_plot(shap_values, X_test)
plt.savefig(os.path.join(folder_path, "SHAP_Summary_Plot.png"))
summary_data = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
summary_data['Feature Values'] = [str(x) for x in X_test.tolist()]
summary_data.to_csv(os.path.join(folder_path, "SHAP_Summary_Plot_data.csv"), index=False)
plt.clf()

# Force Plots for Individual Predictions
sample_indices = np.random.choice(len(X_test), 5, replace=False)
for i, sample_index in enumerate(sample_indices):
    shap.force_plot(
        explainer.expected_value,
        shap_values.values[sample_index],
        X_test[sample_index],
        feature_names=shap_values.feature_names,
        matplotlib=True
    )
    plt.savefig(os.path.join(folder_path, f"SHAP_Force_Plot_sample_{i}.png"))
    plt.clf()

    fx = explainer.expected_value + shap_values.values[sample_index].sum()

    force_data = pd.DataFrame({
        'Feature': shap_values.feature_names,
        'Feature value': X_test[sample_index],
        'SHAP Value': shap_values.values[sample_index]
    })

    force_data['base'] = explainer.expected_value
    force_data['fx'] = fx

    force_data.to_csv(os.path.join(folder_path, f"SHAP_Force_Plot_sample_{i}_data.csv"), index=False)

# shap.dependence_plot
features = ['Pressure', 'Power', 'Spacing', 'Depo_time', 'SiH4', 'NH3', 'N2', 'H2']
num_features = len(features)
num_cols = num_features - 1
num_rows = num_features

fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
all_dependence_data = []

for i, main_feature in enumerate(features):
    main_full_name = feature_mapping[main_feature]
    feature_index = shap_values.feature_names.index(main_full_name)
    sub_axes = axes[i]

    interacting_features = [f for f in features if f != main_feature]

    for j, interacting_feature in enumerate(interacting_features):
        interacting_full_name = feature_mapping[interacting_feature]
        interaction_index = shap_values.feature_names.index(interacting_full_name)

        shap.dependence_plot(
            feature_index,
            shap_values.values,
            X_test,
            feature_names=shap_values.feature_names,
            interaction_index=interaction_index,
            ax=sub_axes[j]
        )
        sub_axes[j].set_title(f'{main_full_name} vs {interacting_full_name}')

        dependence_data = pd.DataFrame({
            main_full_name: X_test[:, feature_index],
            interacting_full_name: X_test[:, interaction_index],
            f'SHAP Value of {main_full_name}': shap_values.values[:, feature_index]
        })
        dependence_data.to_csv(os.path.join(folder_path, f"SHAP_Dependence_Plot_{main_full_name}_vs_{interacting_full_name}_data.csv"), index=False)
        all_dependence_data.append(dependence_data)

plt.tight_layout()

plt.savefig(os.path.join(folder_path, "SHAP_Dependence_Plots_Interactions.png"))
plt.clf()

# Multi-sample Decision Plot
sample_indices = np.random.choice(len(X_test), 10, replace=False)
selected_shap_values = shap_values.values[sample_indices]
plt.figure(figsize=(12, 6))
shap.decision_plot(explainer.expected_value, selected_shap_values, feature_names=shap_values.feature_names)
plt.savefig(os.path.join(folder_path, "SHAP_Multisample_Decision_Plot.png"), bbox_inches='tight')
decision_data = pd.DataFrame(selected_shap_values, columns=shap_values.feature_names)
decision_data.to_csv(os.path.join(folder_path, "SHAP_Multisample_Decision_Plot_data.csv"), index=False)
plt.clf()

# Waterfall Plot for the last sample
plt.figure(figsize=(12, 6))
shap.waterfall_plot(shap.Explanation(values=shap_values.values[sample_index], base_values=explainer.expected_value,
                                     data=X_test[sample_index], feature_names=shap_values.feature_names))
plt.savefig(os.path.join(folder_path, "SHAP_Waterfall_Plot.png"), bbox_inches='tight')
waterfall_data = pd.DataFrame({'Feature': shap_values.feature_names, 'SHAP Value': shap_values.values[sample_index]})
waterfall_data.to_csv(os.path.join(folder_path, "SHAP_Waterfall_Plot_data.csv"), index=False)
plt.clf()
