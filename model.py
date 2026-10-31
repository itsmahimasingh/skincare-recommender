"""The AI part: learn which ingredients suit each skin type, then recommend and explain."""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

SKIN_TYPES = ['Dry', 'Oily', 'Combination', 'Normal', 'Sensitive']

# Well-known ingredients for common skin concerns (looked for by name inside the ingredient list)
CONCERNS = {
    'Acne & breakouts': ['salicylic acid', 'niacinamide', 'zinc pca', 'tea tree', 'benzoyl peroxide', 'sulfur',
                         'azelaic acid'],
    'Dark spots': ['ascorbic acid', 'ascorbyl', 'niacinamide', 'licorice', 'glycyrrhiza', 'arbutin', 'kojic acid',
                   'tranexamic acid', 'azelaic acid'],
    'Dullness': ['glycolic acid', 'lactic acid', 'mandelic acid', 'ascorbic acid', 'niacinamide', 'papain'],
    'Fine lines': ['retinol', 'retinyl', 'retinal', 'peptide', 'bakuchiol', 'ascorbic acid'],
    'Redness': ['centella asiatica', 'allantoin', 'panthenol', 'avena sativa', 'oatmeal', 'ceramide', 'bisabolol',
                'aloe barbadensis'],
    'Dehydration': ['hyaluronic acid', 'sodium hyaluronate', 'ceramide', 'squalane', 'shea butter',
                    'butyrospermum parkii'],
}


def load_products(path='data/cosmetics.csv'):
    """Read the data and keep only products with a real ingredient list."""
    df = pd.read_csv(path)
    df = df[df['Ingredients'].str.count(',') >= 4]          # drops "No Info" and similar
    df = df.drop_duplicates(subset=['Brand', 'Name']).reset_index(drop=True)
    return df


def split_ingredients(text):
    """'Water, Glycerin, Niacinamide' -> ['water', 'glycerin', 'niacinamide']"""
    # split at commas, but not inside names like "1,2-Hexanediol" (comma followed by a digit)
    parts = re.split(r',(?!\d)', text.lower())
    return [p.strip(' .*') for p in parts if p.strip(' .*')]


def build(df):
    """Turn ingredients into numbers (TF-IDF) and train one model per skin type."""
    vectorizer = TfidfVectorizer(tokenizer=split_ingredients, lowercase=False,
                                 token_pattern=None, min_df=3)   # ignore ingredients used by < 3 products
    X = vectorizer.fit_transform(df['Ingredients'])
    models = {}
    for skin in SKIN_TYPES:
        model = LogisticRegression(max_iter=2000, class_weight='balanced')
        model.fit(X, df[skin])
        models[skin] = model
    return vectorizer, X, models


def evaluate(df, X):
    """Accuracy on products the model has NOT seen (5-fold cross-validation). 50% = random guessing."""
    rows = []
    for skin in SKIN_TYPES:
        model = LogisticRegression(max_iter=2000, class_weight='balanced')
        score = cross_val_score(model, X, df[skin], cv=5, scoring='balanced_accuracy').mean()
        rows.append({'Skin type': skin, 'Accuracy on unseen products': f'{score:.0%}', 'Random guess': '50%'})
    return pd.DataFrame(rows)


def concerns_helped(ingredients, concerns):
    """{concern: [ingredients in this product known to help with it]} for the chosen concerns."""
    names = split_ingredients(ingredients)
    found = {}
    for concern in concerns:
        hits = sorted({name for name in names for key in CONCERNS[concern] if key in name})
        if hits:
            found[concern] = hits
    return found


def recommend(df, X, models, skin, label, concerns=(), n=5, skip=()):
    """Products of one type that suit this skin type, best first.
    Score = the model's match for the skin type + a bonus of 0.1 for each chosen concern the product helps with."""
    match = models[skin].predict_proba(X)[:, 1]
    result = df.assign(match=match)
    result = result[(result['Label'] == label) & (result[skin] == 1) & ~result.index.isin(skip)]
    helps = result['Ingredients'].apply(lambda text: len(concerns_helped(text, concerns)))
    result = result.assign(score=result['match'] + 0.1 * helps)
    return result.sort_values(['score', 'Rank'], ascending=False).head(n)


def routine(df, X, models, skin, concerns):
    """A simple morning and night routine: the best product for each step."""
    steps = {'Morning': [('Cleanser', 'Cleanse'), ('Treatment', 'Treat'), ('Moisturizer', 'Moisturise'),
                         ('Sun protect', 'Protect')],
             'Night': [('Cleanser', 'Cleanse'), ('Treatment', 'Treat'), ('Moisturizer', 'Moisturise'),
                       ('Eye cream', 'Eyes')]}
    plan, used = {}, set()
    for time, items in steps.items():
        plan[time] = []
        for label, step in items:
            # the same cleanser and moisturiser can be used morning and night; pick a different treatment
            skip = used if label == 'Treatment' else ()
            best = recommend(df, X, models, skin, label, concerns, n=1, skip=skip)
            if not best.empty:
                used.add(best.index[0])
                plan[time].append((step, best.index[0]))
    return plan


def check_product(text, vectorizer, X, models, df, n_similar=3):
    """For a pasted ingredient list: match for every skin type, and the most similar products in the data."""
    from sklearn.metrics.pairwise import cosine_similarity
    v = vectorizer.transform([text])
    fit = {skin: models[skin].predict_proba(v)[0, 1] for skin in SKIN_TYPES}
    similarity = cosine_similarity(v, X).ravel()
    similar = df.assign(similarity=similarity).sort_values('similarity', ascending=False).head(n_similar)
    known = int((v > 0).sum())      # how many of the pasted ingredients the model knows
    return fit, similar, known


def price_vs_rating(df):
    """Average rating in four price groups (cheapest quarter ... most expensive quarter)."""
    groups = pd.qcut(df['Price'], 4, labels=['Cheapest 25%', 'Lower middle', 'Upper middle', 'Most expensive 25%'])
    rated = df[df['Rank'] > 0]
    return rated.groupby(groups[rated.index], observed=True)['Rank'].mean().round(2)


def top_ingredients(vectorizer, models, skin, n=8):
    """Ingredients the model links most with this skin type (biggest positive weights)."""
    weights = pd.Series(models[skin].coef_[0], index=vectorizer.get_feature_names_out())
    return weights.sort_values(ascending=False).head(n)


def explain(vectorizer, X, models, skin, index, n=6):
    """Which ingredients in this product push its score up (good for this skin type)
    and which push it down. Each ingredient's effect = model weight x how much of it there is."""
    model = models[skin]
    row = X[index].toarray()[0]
    names = vectorizer.get_feature_names_out()
    effects = pd.Series(model.coef_[0] * row, index=names)
    # skip basics found in more than half of all products (like water): they don't explain much
    share = np.asarray((X > 0).mean(axis=0)).ravel()
    effects = effects[(row > 0) & (share < 0.5)]
    good = effects[effects > 0].sort_values(ascending=False).head(n)
    less = effects[effects < 0].sort_values().head(3)
    return good, less


if __name__ == '__main__':
    df = load_products()
    vectorizer, X, models = build(df)
    print('Products:', len(df), '| ingredients the model uses:', X.shape[1])
    print('\nHow well do ingredients predict the skin type?')
    print(evaluate(df, X).to_string(index=False))

    print('\nTop 5 moisturisers for dry skin:')
    top = recommend(df, X, models, 'Dry', 'Moisturizer')
    for i, row in top.iterrows():
        print(f"  {row['match']:.0%} match | rating {row['Rank']} | {row['Brand']} - {row['Name']}")

    first = top.index[0]
    good, less = explain(vectorizer, X, models, 'Dry', first)
    print(f"\nWhy {top.loc[first, 'Name']}? Good for dry skin:", ', '.join(good.index))
