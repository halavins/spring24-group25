from rich.jupyter import display
import ipywidgets as widgets
from ipywidgets import interact
from ipywidgets import HBox, VBox
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy.optimize import minimize
import numpy as np
from scipy.stats import rankdata

def parameterized_model_performance(Z, steepness):
    ind = min(int(100 * Z), 99)
    alert_rates, recalls, precisions = pr_curves(steepness=steepness)
    return recalls[ind], precisions[ind]


def pr_curves(steepness):
    def recall_function(alert_rate, max_recall=1, steepness=10):
        return max_recall * (1 - np.exp(-steepness * alert_rate))

    def precision_function(alert_rate, high_precision=0.9, decay_rate=5):
        return high_precision * np.exp(-decay_rate * alert_rate)

    def generate_pr_curve_data(alert_rate_points=100, max_recall=1, steepness=10, high_precision=0.9, decay_rate=5):
        alert_rates = np.linspace(0, 1, alert_rate_points)
        recalls = [recall_function(ar, max_recall, steepness) for ar in alert_rates]
        precisions = [precision_function(ar, high_precision, decay_rate) for ar in alert_rates]
        return alert_rates, recalls, precisions

    alert_rates, recalls, precisions = generate_pr_curve_data(alert_rate_points=100, max_recall=1, steepness=steepness,
                                                              high_precision=1, decay_rate=5)
    return alert_rates, recalls, precisions


def calculate_cost(filter_threshold,
                   gpt_review_threshold,
                   gpt_ban_threshold,
                   conversation_history_length,
                   filter_steepness,
                   gpt_steepness,
                   new_connection_messages,
                   potential_scam_messages,
                   economic_damage_per_scam,
                   ban_good_user_cost,
                   review_cost):
    recall_filter, precision_filter = parameterized_model_performance(filter_threshold, filter_steepness)

    if gpt_review_threshold <= gpt_ban_threshold:
        return float('inf'), recall_filter, precision_filter, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    filtered_messages = (filter_threshold * new_connection_messages)

    unfiltered_scam_messages_cost = (1 - recall_filter) * potential_scam_messages * economic_damage_per_scam

    avg_tokens_per_message = 50
    total_tokens_for_analysis = filtered_messages * conversation_history_length * avg_tokens_per_message
    effective_tokens_for_analysis = total_tokens_for_analysis / 0.8
    cost_per_million_tokens = 3
    cost_of_ai_compute = (effective_tokens_for_analysis / 1_000_000) * cost_per_million_tokens

    recall_gpt_review, precision_gpt_review = parameterized_model_performance(gpt_review_threshold, filter_steepness)
    recall_gpt_ban, precision_gpt_ban = parameterized_model_performance(gpt_ban_threshold, gpt_steepness)

    messages_to_review = filtered_messages * (gpt_review_threshold - gpt_ban_threshold)
    messages_to_ban = filtered_messages * gpt_ban_threshold
    messages_dismissed = filtered_messages * (1 - gpt_review_threshold)

    banned_good_users = messages_to_ban * (1 - precision_gpt_ban)
    banned_not_scam_cost = banned_good_users * ban_good_user_cost
    cost_of_human_labor = messages_to_review * review_cost
    dismissed_scam_messages = messages_dismissed * (1 - recall_gpt_review)
    dismissed_scam_messages_cost = dismissed_scam_messages * economic_damage_per_scam

    compute_and_labor_cost = cost_of_ai_compute + cost_of_human_labor
    missed_scam_cost = unfiltered_scam_messages_cost + dismissed_scam_messages_cost
    adverse_action_cost = banned_not_scam_cost

    # Total daily cost
    total_daily_cost = compute_and_labor_cost + missed_scam_cost + adverse_action_cost
    return total_daily_cost, recall_filter, precision_filter, recall_gpt_review, precision_gpt_review, recall_gpt_ban, precision_gpt_ban, cost_of_ai_compute, cost_of_human_labor, unfiltered_scam_messages_cost, dismissed_scam_messages_cost, banned_not_scam_cost, messages_to_review, messages_to_ban, messages_dismissed, banned_good_users, dismissed_scam_messages




def scam_detection_analysis(new_dm_proportion,
                            scam_rate,
                            economic_damage_per_scam,
                            review_cost,
                            ban_good_user_cost,
                            conversation_history_length,
                            filter_steepness,
                            gpt_steepness,
                            filter_threshold,
                            gpt_review_threshold,
                            gpt_ban_threshold):
    # Total messages per day
    total_messages = 500_000_000

    # Calculate the number of messages between new connections
    new_connection_messages = (new_dm_proportion / 100) * total_messages

    # Estimate potential scam messages
    potential_scam_messages = scam_rate * new_connection_messages

    # Optimization function
    def optimization_function(params):
        filter_threshold_opt, gpt_review_threshold_opt, gpt_ban_threshold_opt = params
        total_daily_cost, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = calculate_cost(
            filter_threshold_opt,
            gpt_review_threshold_opt,
            gpt_ban_threshold_opt,
            conversation_history_length,
            filter_steepness,
            gpt_steepness,
            new_connection_messages,
            potential_scam_messages,
            economic_damage_per_scam,
            ban_good_user_cost,
            review_cost
        )
        return total_daily_cost

    # Generate data for plotting PR curves
    recalls_filter = []
    precisions_filter = []
    recalls_gpt = []
    precisions_gpt = []

    alert_rates_filter, recalls_filter, precisions_filter = pr_curves(steepness=filter_steepness)
    alert_rates_gpt, recalls_gpt, precisions_gpt = pr_curves(steepness=gpt_steepness)

    # Display calculated values for the current filtering percentage
    total_daily_cost, recall_filter, precision_filter, recall_gpt_review, precision_gpt_review, recall_gpt_ban, precision_gpt_ban, selected_cost_of_ai_compute, selected_cost_of_human_labor, selected_unfiltered_scam_messages_cost, selected_dismissed_scam_messages_cost, selected_banned_not_scam_cost, selected_messages_to_review, selected_messages_to_ban, selected_messages_dismissed, selected_banned_good_users, selected_dismissed_scam_messages = calculate_cost(
        filter_threshold,
        gpt_review_threshold,
        gpt_ban_threshold,
        conversation_history_length,
        filter_steepness,
        gpt_steepness,
        new_connection_messages,
        potential_scam_messages,
        economic_damage_per_scam,
        ban_good_user_cost,
        review_cost)

    # Plot the PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 3))

    # Plot the Filter Model PR curve
    ax1.plot(recalls_filter, precisions_filter, marker='o', linestyle='--')
    ax1.set_title('Filter Model Precision-Recall Curve')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.2)
    ax1.grid(True)

    filter_alert_index = min(int(100 * filter_threshold), 99)
    recall, precision = recalls_filter[filter_alert_index], precisions_filter[filter_alert_index]
    ax1.axvline(x=recall, color='blue', linestyle='--', label='Filter Threshold')
    ax1.annotate(f'Alert Rate {filter_threshold:.2%}', xy=(recall, precision), xytext=(recall + 0.05, precision - 0.1),
                 arrowprops=dict(facecolor='blue', shrink=0.05), color='blue')
    ax1.legend()

    # Plot the GPT Model PR curve
    ax2.plot(recalls_gpt, precisions_gpt, marker='o', linestyle='--')
    ax2.set_title('GPT Model Precision-Recall Curve')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.2)
    ax2.grid(True)

    # Annotate GPT model thresholds
    for threshold, label, color in [(gpt_review_threshold, 'Review', 'green'), (gpt_ban_threshold, 'Ban', 'red')]:
        gpt_alert_index = min(int(100 * threshold), 99)
        recall, precision = recalls_filter[gpt_alert_index], precisions_filter[gpt_alert_index]
        ax2.axvline(x=recall, color=color, linestyle='--', label=f'{label} Threshold')
        ax2.annotate(f'{label} Rate: {threshold:.2%}', xy=(recall, precision), xytext=(recall + 0.05, precision - 0.1),
                     arrowprops=dict(facecolor=color, shrink=0.05), color=color)
    ax2.legend()

    plt.show()

    ##################
    # 3D PLOT
    ##################

    result = minimize(optimization_function,
                      [filter_threshold, gpt_review_threshold, gpt_ban_threshold],
                      bounds=[(0, 1), (0, 1), (0, 1)],
                      method='Nelder-Mead',
                      tol=1e-6)
    optimum_filter_threshold, optimum_gpt_review_threshold, optimum_gpt_ban_threshold = result.x
    optimum_cost, optimum_filter_recall, optimum_filter_precision, optimum_gpt_recall, optimum_gpt_precision, _, _, optimum_cost_of_ai_compute, optimum_cost_of_human_labor, optimum_unfiltered_scam_messages_cost, optimum_dismissed_scam_messages_cost, optimum_banned_not_scam_cost, optimum_messages_to_review, optimum_messages_to_ban, optimum_messages_dismissed, optimum_banned_good_users, optimum_dismissed_scam_messages = calculate_cost(
        optimum_filter_threshold,
        optimum_gpt_review_threshold,
        optimum_gpt_ban_threshold,
        conversation_history_length,
        filter_steepness,
        gpt_steepness,
        new_connection_messages,
        potential_scam_messages,
        economic_damage_per_scam,
        ban_good_user_cost,
        review_cost)

    filter_thresholds = np.linspace(0, 1, 100)
    gpt_ban_thresholds = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(filter_thresholds, gpt_ban_thresholds)
    Z = np.array([calculate_cost(ft,
                                 optimum_gpt_review_threshold,
                                 bt,
                                 conversation_history_length,
                                 filter_steepness,
                                 gpt_steepness,
                                 new_connection_messages,
                                 potential_scam_messages,
                                 economic_damage_per_scam,
                                 ban_good_user_cost,
                                 review_cost)[0] for ft, bt in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

    # Plot the interactive 3D contour plot using plotly
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(title='Daily Cost as a Function of Filter and GPT Ban Thresholds',
                      scene=dict(
                          xaxis_title='Filter Threshold',
                          yaxis_title='GPT Ban Threshold',
                          zaxis_title='Daily Cost ($)'),
                      autosize=True, width=1000, height=800)

    # Mark the optimum point on the contour plot
    fig.add_trace(go.Scatter3d(x=[filter_threshold], y=[gpt_ban_threshold], z=[total_daily_cost],
                               mode='markers', marker=dict(size=10, color='blue'), name='Current Point'))

    fig.add_trace(go.Scatter3d(x=[optimum_filter_threshold], y=[optimum_gpt_ban_threshold], z=[optimum_cost],
                               mode='markers', marker=dict(size=10, color='red'), name='Current Point'))

    fig.show()

    #############
    # 4D Plot
    #############

    # Define the thresholds range
    filter_thresholds = np.linspace(0, 1, 30)
    gpt_review_thresholds = np.linspace(0, 1, 30)
    gpt_ban_thresholds = np.linspace(0, 1, 30)

    # Create a meshgrid
    X, Y, Z = np.meshgrid(filter_thresholds, gpt_review_thresholds, gpt_ban_thresholds)

    # Flatten the meshgrid arrays for easier iteration
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Calculate the function values for each combination
    cost_values = np.array([
        calculate_cost(ft, grt, bt,
                       conversation_history_length,
                       filter_steepness, gpt_steepness,
                       new_connection_messages, potential_scam_messages,
                       economic_damage_per_scam, ban_good_user_cost, review_cost)[0]
        for ft, grt, bt in zip(X_flat, Y_flat, Z_flat)
    ])

    finite_cost_values = cost_values[np.isfinite(cost_values)]

    # Apply logarithmic transformation and round the values
    epsilon = 1e-9  # Small constant to avoid log(0)
    log_transformed_costs = np.log1p(finite_cost_values - finite_cost_values.min() + epsilon)
    rounded_log_costs = np.round(log_transformed_costs, 1)

    # Rank the rounded, transformed cost values
    ranks = rankdata(rounded_log_costs)

    # Create a new array for the ranks, inserting NaN for infinite values
    ranked_costs = np.full(cost_values.shape, np.nan)
    ranked_costs[np.isfinite(cost_values)] = ranks

    # Create the 3D scatter plot using Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=X_flat, y=Y_flat, z=Z_flat,
        mode='markers',
        marker=dict(
            size=5,
            color=ranked_costs.flatten(),
            colorscale='Rainbow',  # Reverse the color scale
            opacity=0.1,
            colorbar=dict(title='Ranked Cost'),
            showscale=True
        )
    )])

    # Add the optimum point to the plot
    fig.add_trace(go.Scatter3d(
        x=[optimum_filter_threshold], y=[optimum_gpt_review_threshold], z=[optimum_gpt_ban_threshold],
        mode='markers', marker=dict(size=10, color='red'), name='Optimum Point'
    ))

    # Update the layout
    fig.update_layout(
        title='Cost as a Function of Filter, GPT Review, and GPT Ban Thresholds',
        scene=dict(
            xaxis_title='Filter Threshold',
            yaxis_title='GPT Review Threshold',
            zaxis_title='GPT Ban Threshold'
        ),
        autosize=True, width=1000, height=800
    )

    fig.show()

    print(f"potential_scam_messages: {potential_scam_messages:,.2f}")
    print(f"selected_messages_to_review: {selected_messages_to_review:,.2f}")
    print(f"selected_messages_to_ban: {selected_messages_to_ban:,.2f}")
    print(f"selected_messages_dismissed: {selected_messages_dismissed:,.2f}")
    print(f"selected_banned_good_users: {selected_banned_good_users:,.2f}")
    print(f"selected_dismissed_scam_messages: {selected_dismissed_scam_messages:,.2f}")

    print(f"selected_total_daily_cost: ${total_daily_cost}")
    print(f"selected_cost_of_ai_compute: ${selected_cost_of_ai_compute}")
    print(f"selected_cost_of_human_labor: ${selected_cost_of_human_labor}")
    print(f"selected_unfiltered_scam_messages_cost: ${selected_unfiltered_scam_messages_cost}")
    print(f"selected_dismissed_scam_messages_cost: ${selected_dismissed_scam_messages_cost}")
    print(f"selected_banned_not_scam_cost: ${selected_banned_not_scam_cost}")

    print(f"Optimum Total Daily optimum_messages_to_review: {optimum_messages_to_review}")
    print(f"Optimum Total Daily optimum_messages_to_ban: {optimum_messages_to_ban}")
    print(f"Optimum Total Daily optimum_messages_dismissed: {optimum_messages_dismissed}")
    print(f"Optimum Total Daily optimum_banned_good_users: {optimum_banned_good_users}")
    print(f"Optimum Total Daily optimum_dismissed_scam_messages: {optimum_dismissed_scam_messages}")
    print(f"Optimum Total Daily optimum_cost_of_ai_compute: ${optimum_cost_of_ai_compute:,.2f}")
    print(f"Optimum Total Daily optimum_cost_of_human_labor: ${optimum_cost_of_human_labor:,.2f}")
    print(f"Optimum Total Daily optimum_unfiltered_scam_messages_cost: ${optimum_unfiltered_scam_messages_cost:,.2f}")
    print(f"Optimum Total Daily optimum_dismissed_scam_messages_cost: ${optimum_dismissed_scam_messages_cost:,.2f}")
    print(f"Optimum Total Daily optimum_banned_not_scam_cost: ${optimum_banned_not_scam_cost:,.2f}")

    print(f"Filter Threshold: {filter_threshold}")
    print(f"GPT Review Threshold: {gpt_review_threshold}")
    print(f"GPT Ban Threshold: {gpt_ban_threshold}")
    print(f"Total Daily Cost: ${total_daily_cost:,.2f}")
    print(f"Filter Model Recall: {recall_filter:.2%}")
    print(f"Filter Model Precision: {precision_filter:.2%}")
    print(f"GPT Model Review Recall: {recall_gpt_review:.2%}")
    print(f"GPT Model Review Precision: {precision_gpt_review:.2%}")
    print(f"GPT Model Ban Recall: {recall_gpt_ban:.2%}")
    print(f"GPT Model Ban Precision: {precision_gpt_ban:.2%}")
    print("\nOptimum Values:")
    print(f"Optimum Filter Threshold: {optimum_filter_threshold}")
    print(f"Optimum GPT Review Threshold: {optimum_gpt_review_threshold}")
    print(f"Optimum GPT Ban Threshold: {optimum_gpt_ban_threshold}")
    print(f"Optimum Total Daily Cost: ${optimum_cost:,.2f}")


# Create layout for widgets
style = {'description_width': 'initial'}

# Create interactive widgets
new_dm_proportion_slider = widgets.FloatSlider(value=1, min=1, max=20, step=1,
                                               description='New DM % (New Connections %):', style=style,
                                               layout=widgets.Layout(width='50%'))
scam_rate_slider = widgets.FloatSlider(value=0.01, min=0.01, max=1, step=0.01, description='Scam Rate (Scam %):',
                                       style=style, layout=widgets.Layout(width='50%'))
economic_damage_per_scam_slider = widgets.FloatSlider(value=10, min=0, max=100, step=10,
                                                      description='Economic Damage per Scam ($):', style=style,
                                                      layout=widgets.Layout(width='50%'))
review_cost_slider = widgets.FloatSlider(value=0.05, min=0, max=10, step=0.01, description='Review Cost ($):',
                                         style=style, layout=widgets.Layout(width='50%'))
ban_good_user_cost_slider = widgets.FloatSlider(value=5, min=0, max=10, step=1, description='Ban Good User Cost ($):',
                                                style=style, layout=widgets.Layout(width='50%'))
conversation_history_length_slider = widgets.IntSlider(value=1, min=1, max=50, step=1,
                                                       description='Conversation History Length:', style=style,
                                                       layout=widgets.Layout(width='50%'))
filter_steepness_slider = widgets.IntSlider(value=20, min=20, max=200, step=10, description='Filter Model Performance:',
                                            style=style, layout=widgets.Layout(width='50%'))
gpt_steepness_slider = widgets.IntSlider(value=100, min=20, max=200, step=10, description='GPT Model Performance:',
                                         style=style, layout=widgets.Layout(width='50%'))
filter_threshold_slider = widgets.FloatSlider(value=0.2, min=0, max=1, step=0.01, description='Filter Threshold:',
                                              style=style, layout=widgets.Layout(width='50%'))
gpt_review_threshold_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.01,
                                                  description='GPT Review Threshold:', style=style,
                                                  layout=widgets.Layout(width='50%'))
gpt_ban_threshold_slider = widgets.FloatSlider(value=0.05, min=0, max=1, step=0.01, description='GPT Ban Threshold:',
                                               style=style, layout=widgets.Layout(width='50%'))

# Arrange widgets in a box
controls = VBox([
    new_dm_proportion_slider,
    scam_rate_slider,
    economic_damage_per_scam_slider,
    review_cost_slider, ban_good_user_cost_slider,
    conversation_history_length_slider,
    filter_steepness_slider,
    gpt_steepness_slider,
    filter_threshold_slider,
    gpt_review_threshold_slider, gpt_ban_threshold_slider
], layout=widgets.Layout(width='100%'))


# Function to update the plots
def update_plots(new_dm_proportion, scam_rate, economic_damage_per_scam, review_cost, ban_good_user_cost,
                 conversation_history_length,
                 filter_steepness,
                 gpt_steepness, filter_threshold, gpt_review_threshold, gpt_ban_threshold):
    scam_detection_analysis(new_dm_proportion, scam_rate, economic_damage_per_scam, review_cost, ban_good_user_cost,
                            conversation_history_length,
                            filter_steepness,
                            gpt_steepness, filter_threshold, gpt_review_threshold, gpt_ban_threshold)


# Create the interactive output
output = widgets.interactive_output(update_plots, {
    'new_dm_proportion': new_dm_proportion_slider,
    'scam_rate': scam_rate_slider,
    'economic_damage_per_scam': economic_damage_per_scam_slider,
    'review_cost': review_cost_slider,
    'ban_good_user_cost': ban_good_user_cost_slider,
    'conversation_history_length': conversation_history_length_slider,
    'filter_steepness': filter_steepness_slider,
    'gpt_steepness': gpt_steepness_slider,
    'filter_threshold': filter_threshold_slider,
    'gpt_review_threshold': gpt_review_threshold_slider,
    'gpt_ban_threshold': gpt_ban_threshold_slider
})

# Display widgets on top and plots below
display(VBox([controls, output]))
