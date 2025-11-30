import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import json
import os
from scipy.stats import kendalltau

# ----------------- Generic function: read last weights -----------------
def get_last_trained_weights(json_path='another_final_trained_weights.json'):
    try:
        with open(json_path, 'r') as f:
            all_data = json.load(f)
        if isinstance(all_data, list) and len(all_data) > 0:
            last_entry = all_data[-1]
            return {attr: last_entry[attr] for attr in ranking_attributes if attr in last_entry}
        else:
            return None
    except:
        return None

# ----------------- Generic function: Normalize weights -----------------
def normalize_weights(weights_dict):
    total = sum(weights_dict.values())
    if total > 0:
        for k in weights_dict:
            weights_dict[k] = max(0.0, weights_dict[k]) / total
    else:
        avg = 1.0 / len(weights_dict)
        for k in weights_dict:
            weights_dict[k] = avg
    return weights_dict

# ===================== STEP 1: Load Data =====================
df = pd.read_csv('makeup_data_updated.csv')

df['log_review_counts'] = np.log1p(df['review_counts'])  # log(x + 1)
ranking_attributes = ['current_price', 'unit_price', 'star_rating', 'log_review_counts']
lower_is_better_attributes = ['current_price', 'unit_price']

print("--- Preview of Raw Data ---")
print(df.head())
print("\nData Column Info:")
print(df.info())
print("-" * 30)

# ===================== STEP 2: User Filter Input =====================
print("\n--- Initial User Filtering ---")

while True:
    try:
        user_on_sale_input = int(input("Do you want to see discounted products? (Enter 1 for Yes, 0 for No): ")) 
        if user_on_sale_input in [0, 1]:
            user_on_sale_choice = user_on_sale_input
            break
    except ValueError:
        pass

available_categories = df['product_category'].unique().tolist()
print("\nAvailable product categories:")
for i, category in enumerate(available_categories):
    print(f"{i+1}. {category}")

while True:
    try:
        category_index = int(input(f"Select a product category (Enter a number 1-{len(available_categories)}): ")) 
        if 1 <= category_index <= len(available_categories):
            user_category_choice = available_categories[category_index - 1]
            break
    except ValueError:
        pass

while True:
    fs_choice = input("Do you want to see products with free shipping? (Enter 1 for Yes, 0 for No): ")  
    if fs_choice in ['0', '1']:
        fs_choice = int(fs_choice)
        break

while True:
    fast_choice = input("Do you want to see products deliverable within 3 days? (Enter 1 for Yes, 0 for No): ") 
    if fast_choice in ['0', '1']:
        fast_choice = int(fast_choice)
        break

filtered_df = df[
    (df['is_on_sale'] == user_on_sale_choice) &
    (df['product_category'] == user_category_choice) &
    (df['is_free_shipping'] == fs_choice) &
    (df['is_3day_delivery'] == fast_choice)
].copy()

filtered_df['recommend_count'] = 0

if filtered_df.empty:
    print("No products remain after filtering. Exiting.")
    exit()

print(f"\nNumber of products after filtering: {len(filtered_df)}")

# ===================== STEP 3: User Attribute Importance Ranking =====================
print("\n--- User Attribute Importance Ranking ---")
print("Please rank the following attributes by importance (1 = most important):", ranking_attributes)

user_attribute_ranking = {}
to_rank = ranking_attributes[:]
rank = 1
while to_rank:
    print("\nAttributes available for ranking:")
    for i, attr in enumerate(to_rank):
        print(f"{i+1}. {attr}")
    try:
        choice = int(input(f"Enter the number for attribute ranked #{rank}: ")) 
        if 1 <= choice <= len(to_rank):
            user_attribute_ranking[to_rank.pop(choice - 1)] = rank
            rank += 1
    except ValueError:
        continue

print("\nRanking result:")
for attr, r in sorted(user_attribute_ranking.items(), key=lambda x: x[1]):
    print(f"{attr}: Rank {r}")

# ===================== STEP 4: Use Previous Training Results =====================
previous_weights = get_last_trained_weights()
if previous_weights:
    print("✅ Using weights from the last training as initial weights.")
    initial_weights = previous_weights
else:
    print("⚠️ No previous result found, initializing weights based on current ranking.")
    max_rank = len(ranking_attributes)
    initial_weights = {
        attr: (max_rank - rank + 1)
        for attr, rank in user_attribute_ranking.items()
    }
    initial_weights = normalize_weights(initial_weights)

updated_weights = initial_weights.copy()

print("\n--- Initial Attribute Weights ---")
print(updated_weights)
print("-" * 30)

# ===================== STEP 5-6: Initial Recommendation Function =====================

# ---- Normalize filtered_df for scoring ----
def get_normalized_scaled_df(df, ranking_attributes, lower_is_better_attributes):
    df_scaled = df.copy()
    scaler = MinMaxScaler()
    df_scaled[ranking_attributes] = scaler.fit_transform(df_scaled[ranking_attributes])
    for attr in lower_is_better_attributes:
        if attr in df_scaled.columns:
            df_scaled[attr] = 1.0 - df_scaled[attr]
    return df_scaled

df_scaled = get_normalized_scaled_df(filtered_df, ranking_attributes, lower_is_better_attributes)

# ---- Compute scores with normalized features + initial weights + UCB bonus ----
def calculate_scores_with_ucb(df_processed, weights, t, alpha=0.05):
    scores = np.zeros(len(df_processed))
    for attr, weight in weights.items():
        if attr in df_processed.columns:
            scores += df_processed[attr].values * weight
    bonus = alpha * np.sqrt(2 * np.log(t + 1) / (df_processed['recommend_count'] + 1))
    return scores + bonus

# Calculate initial scores (with normalized features)
initial_scores = calculate_scores_with_ucb(df_scaled, initial_weights, t=1, alpha=0.05)
df_scaled['ucb_initial_score'] = initial_scores
filtered_df['ucb_initial_score'] = initial_scores  # for indexing later

# ---- Select Top 5 Initial Recommendations ----
initial_recommendations = filtered_df.sort_values(by='ucb_initial_score', ascending=False).head(5).copy()
initial_recommendations_display = filtered_df.loc[initial_recommendations.index].copy()
initial_recommendations_display['ucb_initial_score'] = initial_recommendations['ucb_initial_score']

# ---- Compute Pure Scores (normalized features × initial weights, no UCB bonus) ----
df_scaled_top5 = df_scaled.loc[initial_recommendations.index].copy()
normalized_scores = sum(df_scaled_top5[attr] * initial_weights[attr] for attr in ranking_attributes)
initial_recommendations_display['pure_initial_score'] = normalized_scores

# ---- Print Initial Recommendations ----
print("\n--- Initial Recommended Products ---")
print(initial_recommendations_display[['product_name', 'brand', 'current_price', "unit_price",
                                       'star_rating', "review_counts", 'ucb_initial_score', 'pure_initial_score']])
print("-" * 30)

# ===================== STEP 7: User Feedback Re-Ranking + Multi-Round Weight Update =====================
print("\n--- Please re-rank the recommended products ---")
print(f"Please rank the {len(initial_recommendations)} recommended products by your preference (1 = most preferred):")
num_iterations = 3

original_df = filtered_df.copy()  # ✅ Keep original data for display

for iteration in range(num_iterations):
    print(f"\n====== Recommendation & Feedback Round {iteration+1} ======")

    # ✅ Normalize features for this round
    df_scaled = get_normalized_scaled_df(filtered_df, ranking_attributes, lower_is_better_attributes)

    # UCB score and recommendation
    df_scaled['current_score'] = calculate_scores_with_ucb(df_scaled, updated_weights, iteration + 1, alpha=0.05)
    top_indices = df_scaled.sort_values(by='current_score', ascending=False).head(5).index
    filtered_df.loc[top_indices, 'recommend_count'] += 1

    # Display recommended products (original view)
    print("\nRecommended Products for this round:")
    display_df = original_df.loc[top_indices].copy()
    display_df['current_score'] = df_scaled.loc[top_indices, 'current_score']
    print(display_df[['product_name', 'brand', 'current_price', "unit_price",
                      'star_rating', "review_counts", 'current_score']])

    # User feedback re-ranking
    user_reranking = {}
    recommended_items_list = list(top_indices)
    rank_counter = 1
    while recommended_items_list:
        print(f"\nPlease input the number of the product ranked #{rank_counter}:")
        for i, idx in enumerate(recommended_items_list):
            item = original_df.loc[idx]
            print(f"{i+1}. {item['product_name']} (Brand: {item['brand']}, Price: {item['current_price']:.2f}, "
                  f"Unit Price: {item['unit_price']:.2f}, Rating: {item['star_rating']}, "
                  f"Reviews: {item['review_counts']:.2f})")
        try:
            choice = int(input("Enter number: "))
            if 1 <= choice <= len(recommended_items_list):
                chosen_idx = recommended_items_list.pop(choice - 1)
                user_reranking[chosen_idx] = rank_counter
                rank_counter += 1
        except ValueError:
            pass

    # ✅ Calculate feature importance updates (delta w) based on user preference × feature values
    attribute_updates = {attr: 0.0 for attr in ranking_attributes}
    for idx, user_rank in user_reranking.items():
        if idx not in df_scaled.index:
            continue
        item_features = df_scaled.loc[idx, ranking_attributes]
        max_rank = len(user_reranking)
        preference = (max_rank - user_rank) / (max_rank - 1) if max_rank > 1 else 1.0
        for attr in ranking_attributes:
            attribute_updates[attr] += preference * item_features[attr]

    # ✅ Normalize delta w
    total_delta = sum(attribute_updates.values())
    if total_delta > 0:
        for attr in attribute_updates:
            attribute_updates[attr] = attribute_updates[attr] / total_delta
    else:
        avg = 1.0 / len(attribute_updates)
        for attr in attribute_updates:
            attribute_updates[attr] = avg

    # ✅ Momentum update of weights
    momentum = 0.9  # higher = more inertia
    for attr in ranking_attributes:
        updated_weights[attr] = momentum * updated_weights[attr] + (1 - momentum) * attribute_updates[attr]

    # ✅ Normalize and clip
    total = sum(updated_weights.values())
    if total > 0:
        for attr in updated_weights:
            updated_weights[attr] = max(0.0, updated_weights[attr]) / total
    else:
        avg = 1.0 / len(updated_weights)
        for attr in updated_weights:
            updated_weights[attr] = avg

    print(f"Updated weights after round {iteration+1}:")
    print(updated_weights)
    print("-" * 40)

# ===================== STEP 8: Final Recommendation =====================
print("\n====== Final Recommendation Results ======")

# ---- Compute final scores (normalized features × updated weights + UCB) ----
final_scores = calculate_scores_with_ucb(df_scaled, updated_weights, t=num_iterations + 1, alpha=0.05)
df_scaled['ucb_final_score'] = final_scores
filtered_df['ucb_final_score'] = final_scores

# ---- Select Top 5 Final Recommendations ----
final_recommendations = filtered_df.sort_values(by='ucb_final_score', ascending=False).head(5).copy()
final_recommendations_display = df.loc[final_recommendations.index].copy()
final_recommendations_display['ucb_final_score'] = final_recommendations['ucb_final_score']

# ---- Compute pure final scores (normalized features × final weights) ----
df_scaled_top5 = df_scaled.loc[final_recommendations.index].copy()
normalized_final_scores = sum(df_scaled_top5[attr] * updated_weights[attr] for attr in ranking_attributes)
final_recommendations_display['pure_final_score'] = normalized_final_scores

# ---- Display Final Recommendations ----
print("\nFinal Recommended Products:")
print(final_recommendations_display[['product_name', 'brand', 'current_price', "unit_price",
                                     'star_rating', "review_counts", 'ucb_final_score', 'pure_final_score']])

# ===================== STEP 9: Compare Initial vs Final Recommendation =====================
print("\n--- Initial Recommendation vs Final Recommendation (Product Names) ---")
print("Initial Recommendation:")
print(initial_recommendations_display['product_name'])
print("Final Recommendation:")
print(final_recommendations_display['product_name'])

# ===================== STEP 10: Save Final Weights =====================
def save_final_weights(ordering, weights, csv_path='another_final_trained_weights.csv', json_path='another_final_trained_weights.json'):
    """
    Save the final weights for each ranking order. Ensure normalization before saving.
    """
    weights = normalize_weights(weights)

    row = {
        'ranking_order': ' > '.join([k for k, _ in sorted(ordering.items(), key=lambda item: item[1])])
    }
    row.update(weights)

    # Save to CSV
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
        df_new.to_csv(csv_path, index=False)
    else:
        pd.DataFrame([row]).to_csv(csv_path, index=False)

    # Save to JSON
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            json_list = json.load(f)
    else:
        json_list = []

    json_list.append(row)
    with open(json_path, 'w') as f:
        json.dump(json_list, f, indent=2)

# ✅ Save this round's result
save_final_weights(user_attribute_ranking, updated_weights)
