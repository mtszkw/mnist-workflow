import matplotlib.pyplot as plt
from xgboost import plot_importance

def plot_rmse_bar_chart(rmse_train: float, rmse_valid: float, rmse_test: float, ax):
    bar_chart_x = ["Test", "Validation", "Training"]
    bar_chart_x_pos = [i for i, _ in enumerate(bar_chart_x)]
    bar_chart_y = [rmse_test, rmse_valid, rmse_train]
    ax.barh(bar_chart_x_pos, bar_chart_y)
    ax.set_yticks(bar_chart_x_pos)
    ax.set_yticklabels(bar_chart_x)
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Data set")
    ax.set_title("Root Mean Square Error")


def plot_feature_importance(importance, ax):
    for key in importance.keys():
        importance[key] = round(importance[key], 2)
    plot_importance(importance, importance_type='gain', show_values=True, ax=ax)
    