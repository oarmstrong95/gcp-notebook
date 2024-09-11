import matplotlib.pyplot as plt

def create_oneway_plot(data, feature, target):
    """
    Create a oneway plot to visualize the relationship between a feature and the mean target value.
    Parameters:
    data (pandas.DataFrame): The input data.
    feature (str): The name of the feature to plot.
    target (str): The name of the target variable.
    Returns:
    None
    """
    # Group the data by the feature and calculate the mean target value for each group
    grouped_data = data.groupby(feature)[target].mean()
    
    # Plot the mean target values
    plt.figure(figsize=(10, 6))
    plt.plot(grouped_data.index, grouped_data.values, marker='o')
    plt.xlabel(feature)
    plt.ylabel('Mean Target Value')
    plt.title(f'Oneway Plot: {feature} vs. {target}')
    plt.savefig(f'diagnostics/{feature}.png')
